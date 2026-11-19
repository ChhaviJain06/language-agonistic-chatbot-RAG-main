"""Sarvam streaming TTS: pooling, trimming, barge-in cleanup and REST fallback — against a
fake WebSocket, so no network or credits are needed."""

from __future__ import annotations

import array
import asyncio
import base64
import json
import time

import pytest

from server.providers import tts_sarvam
from server.providers.tts_sarvam import SarvamTTS, _Trimmer


def pcm(silence: int = 0, speech: int = 0, level: int = 2000) -> bytes:
    a = array.array("h", [0] * silence + [level] * speech)
    return a.tobytes()


def audio_msg(data: bytes) -> dict:
    return {"type": "audio", "data": {"content_type": "audio/pcm",
                                      "audio": base64.b64encode(data).decode()}}


FINAL = {"type": "event", "data": {"event_type": "final"}}


class ConnectionClosedError(Exception):          # name-matches websockets' ConnectionClosed*
    pass


class FakeWS:
    def __init__(self, replies: list[dict], fail_send: bool = False):
        self.replies, self.sent, self.closed, self.fail_send = list(replies), [], False, fail_send

    async def send(self, message: str) -> None:
        if self.fail_send:
            raise ConnectionClosedError("server closed it")
        self.sent.append(json.loads(message))

    async def recv(self) -> str:
        await asyncio.sleep(0)
        if not self.replies:
            await asyncio.Event().wait()             # block, like a server still generating
        return json.dumps(self.replies.pop(0))

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
async def tts(monkeypatch):
    t = SarvamTTS(api_key="k")
    t.opened = []                                    # sockets handed out, in order
    t.queue = []                                     # sockets to hand out next

    async def fake_open():
        ws = t.queue.pop(0)
        t.opened.append(ws)
        return ws

    monkeypatch.setattr(t, "_open", fake_open)
    yield t
    await t.close()


async def collect(gen) -> bytes:
    return b"".join([c async for c in gen])


async def test_streams_pcm_and_sends_config_text_flush(tts):
    ws = FakeWS([audio_msg(pcm(speech=100)), audio_msg(pcm(speech=50)), FINAL])
    tts.queue.append(ws)
    out = await collect(tts.synthesize("Namaste duniya", "hi-en"))
    assert len(out) == 150 * 2
    config, text, flush = ws.sent
    assert config["type"] == "config"
    assert config["data"] == {"language_code": "hi-IN", "speaker": "shubh",
                              "speech_sample_rate": "22050", "output_audio_codec": "linear16",
                              "min_buffer_size": 30}
    assert text == {"type": "text", "data": {"text": "Namaste duniya"}}
    assert flush == {"type": "flush"}


async def test_connection_is_reused_across_requests(tts):
    ws = FakeWS([audio_msg(pcm(speech=10)), FINAL, audio_msg(pcm(speech=10)), FINAL])
    tts.queue.append(ws)
    await collect(tts.synthesize("one", "en"))
    await collect(tts.synthesize("two", "hi"))
    assert len(tts.opened) == 1                       # a second connect would cost 0.25-0.5 s
    assert [m["data"]["language_code"] for m in ws.sent if m["type"] == "config"] == \
        ["en-IN", "hi-IN"]                            # language switches on the same socket
    assert not ws.closed


async def test_leading_digital_silence_is_trimmed(tts):
    tts.queue.append(FakeWS([audio_msg(pcm(silence=5000, speech=300)), FINAL]))
    out = await collect(tts.synthesize("x", "en"))
    assert len(out) == (tts_sarvam.LEAD_PAD_SAMPLES + 300) * 2     # 20 ms pad + the speech
    assert out[-2:] == pcm(speech=1)


async def test_silence_spanning_several_chunks_is_trimmed_too(tts):
    tts.queue.append(FakeWS([audio_msg(pcm(silence=3000)), audio_msg(pcm(silence=3000)),
                             audio_msg(pcm(silence=10, speech=100)), FINAL]))
    out = await collect(tts.synthesize("x", "en"))
    assert len(out) == (10 + 100) * 2                 # < 20 ms of silence remained: all kept
    assert out.startswith(pcm(silence=10))


def test_trimmer_keeps_samples_aligned_when_a_chunk_splits_one():
    t = _Trimmer()
    data = pcm(speech=4)
    assert t.feed(data[:3]) + t.feed(data[3:]) == data       # odd split, nothing lost or shifted


async def test_barge_in_cancellation_closes_the_socket_and_does_not_pool_it(tts):
    ws = FakeWS([audio_msg(pcm(speech=10))])          # then blocks: the server "keeps generating"
    tts.queue.append(ws)
    got = asyncio.Event()

    async def consume():
        async for _ in tts.synthesize("a long answer", "en"):
            got.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(got.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)
    assert ws.closed and tts._idle == []


async def test_a_consumer_that_stops_early_does_not_leak_the_socket(tts):
    ws = FakeWS([audio_msg(pcm(speech=10)), audio_msg(pcm(speech=10)), FINAL])
    tts.queue.append(ws)
    gen = tts.synthesize("x", "en")
    await gen.__anext__()
    await gen.aclose()
    await asyncio.sleep(0.02)
    assert ws.closed and tts._idle == []


async def test_a_stale_idle_socket_is_discarded(tts):
    old = FakeWS([])
    tts._idle = [(old, time.monotonic() - 100)]
    fresh = FakeWS([audio_msg(pcm(speech=10)), FINAL])
    tts.queue.append(fresh)
    await collect(tts.synthesize("x", "en"))
    assert old.closed and tts.opened == [fresh]


async def test_a_pooled_socket_the_server_already_closed_is_retried_on_a_fresh_one(tts):
    dead = FakeWS([], fail_send=True)
    tts._idle = [(dead, time.monotonic())]
    fresh = FakeWS([audio_msg(pcm(speech=10)), FINAL])
    tts.queue.append(fresh)
    out = await collect(tts.synthesize("x", "en"))
    assert len(out) == 20 and dead.closed and tts.opened == [fresh]


async def _rest(self, text, lang, voice):
    yield b"rest-audio"


async def test_falls_back_to_rest_when_the_connection_cannot_be_opened(tts, monkeypatch):
    async def boom():
        raise OSError("no route")

    monkeypatch.setattr(tts, "_open", boom)
    monkeypatch.setattr(SarvamTTS, "_synthesize_rest", _rest)
    assert await collect(tts.synthesize("x", "en")) == b"rest-audio"


async def test_falls_back_to_rest_on_a_server_error_before_any_audio(tts, monkeypatch):
    tts.queue.append(FakeWS([{"type": "error", "data": {"message": "quota"}}]))
    monkeypatch.setattr(SarvamTTS, "_synthesize_rest", _rest)
    assert await collect(tts.synthesize("x", "en")) == b"rest-audio"


async def test_an_error_after_audio_was_produced_is_raised_not_masked(tts, monkeypatch):
    tts.queue.append(FakeWS([audio_msg(pcm(speech=10)),
                             {"type": "error", "data": {"message": "boom"}}]))
    monkeypatch.setattr(SarvamTTS, "_synthesize_rest", _rest)
    with pytest.raises(RuntimeError, match="boom"):
        await collect(tts.synthesize("x", "en"))      # re-speaking via REST would repeat words


async def test_streaming_can_be_switched_off(tts, monkeypatch):
    from server.config import Settings

    monkeypatch.setattr(tts_sarvam, "get_settings",
                        lambda: Settings(_env_file=None, sarvam_tts_streaming=False))
    monkeypatch.setattr(SarvamTTS, "_synthesize_rest", _rest)
    assert await collect(tts.synthesize("x", "en")) == b"rest-audio"
    assert tts.opened == []


def test_odia_uses_the_code_the_streaming_api_documents():
    assert tts_sarvam.LANG_MAP["or"] == "od-IN"
