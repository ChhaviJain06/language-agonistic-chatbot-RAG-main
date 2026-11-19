"""Sarvam Saaras ASR — the Indic-speech leg of the dual-path router. Sarvam has a free
credit tier; the local alternative is faster-whisper.
"""

from __future__ import annotations

import time

import httpx

from server.config import get_settings
from server.providers._http import make_client
from server.providers._pcm import pcm_to_wav_bytes
from server.providers.base import ASRProvider, Transcript

ASR_URL = "https://api.sarvam.ai/speech-to-text"
MODEL = "saaras:v3"

# Saarika transcribes into each language's native script.
NATIVE_SCRIPT = {
    "hi": "Deva", "mr": "Deva", "ne": "Deva", "sa": "Deva",
    "bn": "Beng", "as": "Beng",
    "ta": "Taml", "te": "Telu", "kn": "Knda", "ml": "Mlym",
    "gu": "Gujr", "pa": "Guru", "or": "Orya", "en": "Latn",
}
# Sarvam wants region-qualified codes ("hi-IN"); our contract uses bare families.
SARVAM_CODES = {
    "hi": "hi-IN", "mr": "mr-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "gu": "gu-IN", "pa": "pa-IN", "or": "od-IN",
    "en": "en-IN",
}


class SarvamASR(ASRProvider):
    name = "sarvam-saarika"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().sarvam_api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = make_client()
        return self._client

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": MODEL}
        base = hint_lang.split("-")[0] if hint_lang else None
        if base in SARVAM_CODES:
            data["language_code"] = SARVAM_CODES[base]
        headers = {"api-subscription-key": self._api_key}

        start = time.monotonic()
        resp = await self._get_client().post(ASR_URL, files=files, data=data, headers=headers)
        resp.raise_for_status()
        body = resp.json()
        latency_ms = (time.monotonic() - start) * 1000

        lang = (body.get("language_code") or base or "hi").split("-")[0]
        return Transcript(
            text=body.get("transcript", "") or "", lang=lang,
            script=NATIVE_SCRIPT.get(lang, "Deva"),
            confidence=1.0,   # the API returns no per-utterance confidence
            is_final=True, latency_ms=latency_ms, asr_engine_used=self.name,
        )

    async def warm(self) -> None:
        try:
            await self._get_client().get("https://api.sarvam.ai/", timeout=5.0)
        except httpx.HTTPError:
            pass        # only the TLS handshake matters; the response is irrelevant

    async def health(self) -> bool:
        """Key present. A real call would spend free-tier credits on every check."""
        return bool(self._api_key)
