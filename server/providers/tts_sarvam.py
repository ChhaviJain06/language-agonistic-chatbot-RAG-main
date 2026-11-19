"""Sarvam Bulbul TTS (C3 — never Piper, archived Oct 2025). Free credit tier.

Two transports, same output (raw 16-bit mono PCM):

* **WebSocket streaming (default).** One persistent connection per concurrent synthesis,
  kept in a small pool. Measured with a reused connection: first audio ~0.25 s whatever the
  text length, versus the REST call's 0.3 s + 0.02 s per character (2 s for a 100-char
  sentence) because REST returns the whole utterance at once. A NEW connection costs
  0.25-0.5 s, which is why connections are pooled, pre-opened by warm() and pinged.
* **REST fallback.** Used when streaming is disabled or fails before any audio was produced.

Barge-in: Sarvam has no cancel message, so a cancelled synthesis closes its socket instead of
returning it to the pool (the server stops generating when the socket closes).

Each reply starts with 0.1-0.3 s of digital silence; it is trimmed so speech starts sooner.
"""

from __future__ import annotations

import array
import asyncio
import base64
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers._http import make_client
from server.providers._pcm import wav_bytes_to_pcm
from server.providers.base import TTSProvider

logger = logging.getLogger(__name__)

TTS_URL = "https://api.sarvam.ai/text-to-speech"
WS_URL = "wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v3&send_completion_event=true"
MODEL = "bulbul:v3"
SAMPLE_RATE = 22050  # verified live from a real Bulbul response's WAV header

# Sarvam wants BCP-47-with-region codes; our contract uses bare language families.
# "hi-en" (Hinglish) maps to hi-IN — Bulbul reads romanised Hindi with the Hindi voice.
LANG_MAP = {
    "en": "en-IN", "hi": "hi-IN", "hi-en": "hi-IN", "mr": "mr-IN",
    "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "gu": "gu-IN", "ml": "ml-IN", "pa": "pa-IN", "or": "od-IN",
}

RECV_TIMEOUT_S = 15.0          # max wait for the next message from the server
IDLE_MAX_S = 40.0              # the server closes sockets idle for ~60 s; stay well inside that
PING_EVERY_S = 20.0
POOL_SIZE = 4
AUDIBLE = 150                  # |sample| above this is speech (~ -47 dBFS)
LEAD_PAD_SAMPLES = SAMPLE_RATE // 50       # keep 20 ms before the first audible sample


class _Trimmer:
    """Drops the leading digital silence of one utterance, working chunk by chunk."""

    def __init__(self) -> None:
        self._started = False
        self._carry = b""        # odd trailing byte of the previous chunk (keeps samples aligned)

    def feed(self, data: bytes) -> bytes:
        data = self._carry + data
        usable = len(data) - (len(data) % 2)
        self._carry = data[usable:]
        data = data[:usable]
        if self._started or not data:
            return data
        samples = array.array("h")
        samples.frombytes(data)
        for i, s in enumerate(samples):
            if abs(s) > AUDIBLE:
                self._started = True
                return data[max(0, i - LEAD_PAD_SAMPLES) * 2:]
        return b""               # still silence: wait for the next chunk


class SarvamTTS(TTSProvider):
    name = "sarvam-bulbul"
    sample_rate = SAMPLE_RATE

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().sarvam_api_key
        self._client: httpx.AsyncClient | None = None
        self._idle: list[tuple[object, float]] = []     # (websocket, last used)
        self._keepalive: asyncio.Task | None = None
        self._closing: set[asyncio.Future] = set()    # strong refs until each close finishes

    # ------------------------------------------------------------------- REST fallback
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = make_client()
        return self._client

    async def _synthesize_rest(self, text: str, lang: str,
                               voice: str | None) -> AsyncIterator[bytes]:
        payload = {"inputs": [text], "target_language_code": LANG_MAP.get(lang, "en-IN"),
                   "model": MODEL}
        if voice:
            payload["speaker"] = voice
        resp = await self._get_client().post(
            TTS_URL, json=payload, headers={"api-subscription-key": self._api_key})
        resp.raise_for_status()
        for b64_audio in resp.json()["audios"]:
            # ABC contract is raw PCM — Bulbul returns a WAV container, so unwrap it.
            pcm, _sr, _ch = wav_bytes_to_pcm(base64.b64decode(b64_audio))
            yield pcm

    # -------------------------------------------------------------- WebSocket plumbing
    async def _open(self):
        import websockets  # SDK import stays inside providers/ (C1)

        return await websockets.connect(
            WS_URL, additional_headers={"Api-Subscription-Key": self._api_key},
            max_size=None, open_timeout=5, ping_interval=None)

    async def _acquire(self):
        while self._idle:
            ws, last_used = self._idle.pop()
            if time.monotonic() - last_used < IDLE_MAX_S:
                return ws
            await self._close_quietly(ws)
        return await self._open()

    async def _release(self, ws) -> None:
        if len(self._idle) >= POOL_SIZE:
            await self._close_quietly(ws)
            return
        self._idle.append((ws, time.monotonic()))
        if self._keepalive is None or self._keepalive.done():
            self._keepalive = asyncio.create_task(self._keepalive_loop())

    @staticmethod
    async def _close_quietly(ws) -> None:
        with contextlib.suppress(Exception):
            await ws.close()

    async def _keepalive_loop(self) -> None:
        """Sarvam closes a socket after a minute of inactivity; a JSON ping keeps it open."""
        while self._idle:
            await asyncio.sleep(PING_EVERY_S)
            for ws, last_used in list(self._idle):
                if time.monotonic() - last_used >= IDLE_MAX_S:
                    self._idle = [p for p in self._idle if p[0] is not ws]
                    await self._close_quietly(ws)
                    continue
                try:
                    await ws.send(json.dumps({"type": "ping"}))
                except Exception:  # noqa: BLE001 - a dead socket is simply dropped from the pool
                    self._idle = [p for p in self._idle if p[0] is not ws]

    async def close(self) -> None:
        if self._keepalive:
            self._keepalive.cancel()
        idle, self._idle = self._idle, []
        for ws, _ in idle:
            await self._close_quietly(ws)
        if self._client is not None:
            await self._client.aclose()

    async def _synthesize_ws(self, text: str, lang: str,
                             voice: str | None) -> AsyncIterator[bytes]:
        config = {"type": "config", "data": {
            "language_code": LANG_MAP.get(lang, "en-IN"),
            "speaker": voice or get_settings().sarvam_speaker,
            "speech_sample_rate": str(SAMPLE_RATE),
            "output_audio_codec": "linear16",      # raw PCM, content_type "audio/pcm"
            "min_buffer_size": 30,
        }}
        for attempt in (1, 2):
            ws = await self._acquire()
            produced = False
            finished = False
            released = False
            trimmer = _Trimmer()
            try:
                await ws.send(json.dumps(config))
                await ws.send(json.dumps({"type": "text", "data": {"text": text}}))
                await ws.send(json.dumps({"type": "flush"}))
                while not finished:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), RECV_TIMEOUT_S))
                    kind = msg.get("type")
                    if kind == "audio":
                        pcm = trimmer.feed(base64.b64decode(msg["data"]["audio"]))
                        if pcm:
                            produced = True
                            yield pcm
                    elif kind == "event" and msg["data"].get("event_type") == "final":
                        finished = True
                    elif kind == "error":
                        raise RuntimeError(f"Sarvam TTS error: {msg.get('data')}")
                    # anything else (e.g. a ping reply) is ignored
                await self._release(ws)
                released = True
                return
            except Exception as exc:
                stale = not produced and attempt == 1 and \
                    type(exc).__name__.startswith("ConnectionClosed")
                if stale:
                    continue        # a pooled socket the server had already closed: retry fresh
                raise
            finally:
                if not released:
                    # Error, barge-in cancellation, or the consumer stopped reading early. Sarvam
                    # has no cancel message: closing the socket is what stops the server working.
                    self._closing.add(t := asyncio.ensure_future(self._close_quietly(ws)))
                    t.add_done_callback(self._closing.discard)

    # ---------------------------------------------------------------------- public API
    async def synthesize(self, text: str, lang: str,
                         voice: str | None = None) -> AsyncIterator[bytes]:
        if get_settings().sarvam_tts_streaming:
            produced = False
            try:
                async for pcm in self._synthesize_ws(text, lang, voice):
                    produced = True
                    yield pcm
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - fall back to REST if nothing was spoken
                if produced:
                    raise
                logger.warning("Sarvam streaming TTS failed (%r) — using REST", exc)
        async for pcm in self._synthesize_rest(text, lang, voice):
            yield pcm

    async def warm(self) -> None:
        """Open one streaming connection now, so the first reply doesn't pay for it."""
        if not get_settings().sarvam_tts_streaming:
            with contextlib.suppress(httpx.HTTPError):
                await self._get_client().get("https://api.sarvam.ai/", timeout=5.0)
            return
        ws = await self._open()
        await self._release(ws)

    async def health(self) -> bool:
        """Key present. A real synthesis would spend free-tier credits on every check."""
        return bool(self._api_key)
