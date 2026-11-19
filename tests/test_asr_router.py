"""Dual-path routing (§4 feature #1) is the one thing that must never guess —
these pin the routing table down as a pure function, no audio/network needed.
"""

from __future__ import annotations

import asyncio

from server.providers.asr_router import choose_route, route_transcribe
from server.providers.base import ASRProvider, Transcript


def test_devanagari_hindi_routes_indic():
    assert choose_route("hi") == "indic"


def test_english_routes_en():
    assert choose_route("en") == "en"


def test_hinglish_code_mixed_routes_en_not_indic():
    # "hi-en" is code-mixed and Latin-script — the English/code-mixed engine
    # is the one built to handle it, not the Indic-tuned one.
    assert choose_route("hi-en") == "en"


def test_no_hint_defaults_to_en():
    assert choose_route(None) == "en"


def test_other_indic_languages_route_indic():
    for lang in ("mr", "bn", "ta", "te", "kn", "ml", "gu", "pa"):
        assert choose_route(lang) == "indic"


class MockASR(ASRProvider):
    def __init__(self, name: str, should_fail: bool = False, delay: float = 0.0):
        self.name = name
        self.should_fail = should_fail
        self.delay = delay

    async def transcribe(
        self, pcm: bytes, sample_rate: int, hint_lang: str | None = None
    ) -> Transcript:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            raise RuntimeError(f"{self.name} failed")
        return Transcript(
            text="mock transcript",
            lang=hint_lang or "en",
            script="Latn",
            confidence=0.99,
            is_final=True,
            latency_ms=10.0,
            asr_engine_used=self.name,
        )

    async def health(self) -> bool:
        return True


async def test_route_transcribe_indic_fallback_on_error(monkeypatch):
    sarvam_mock = MockASR("sarvam-mock", should_fail=True)
    whisper_mock = MockASR("whisper-mock", should_fail=False)

    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: sarvam_mock)
    monkeypatch.setattr("server.providers.asr_router.get_asr_en", lambda: whisper_mock)

    res = await route_transcribe(b"\x00" * 320, 16000, hint_lang="hi")
    assert res.text == "mock transcript"
    assert res.asr_engine_used == "whisper-mock"


async def test_route_transcribe_indic_fallback_on_timeout(monkeypatch):
    sarvam_mock = MockASR("sarvam-mock", should_fail=False, delay=0.1)
    whisper_mock = MockASR("whisper-mock", should_fail=False, delay=0.0)

    monkeypatch.setattr("server.providers.asr_router.SARVAM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: sarvam_mock)
    monkeypatch.setattr("server.providers.asr_router.get_asr_en", lambda: whisper_mock)

    res = await route_transcribe(b"\x00" * 320, 16000, hint_lang="hi")
    assert res.text == "mock transcript"
    assert res.asr_engine_used == "whisper-mock"


class FixedLangASR(MockASR):
    def __init__(self, name, lang, text="transcript"):
        super().__init__(name)
        self._lang, self._text = lang, text

    async def transcribe(self, pcm, sample_rate, hint_lang=None):
        return Transcript(text=self._text, lang=self._lang, script="Latn", confidence=1.0,
                          is_final=True, latency_ms=1.0, asr_engine_used=self.name)


async def test_unhinted_indic_speech_is_rerouted_to_the_indic_engine(monkeypatch):
    whisper = FixedLangASR("whisper", "hi", "garbled")
    sarvam = FixedLangASR("sarvam", "hi", "सही पाठ")
    monkeypatch.setattr("server.providers.asr_router.get_asr_en", lambda: whisper)
    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: sarvam)
    res = await route_transcribe(b"\x00" * 320, 16000, hint_lang=None)
    assert res.asr_engine_used == "sarvam" and res.text == "सही पाठ"


async def test_urdu_labelled_hindi_audio_is_also_rerouted(monkeypatch):
    monkeypatch.setattr("server.providers.asr_router.get_asr_en",
                        lambda: FixedLangASR("whisper", "ur", "x"))
    monkeypatch.setattr("server.providers.asr_router.get_asr_indic",
                        lambda: FixedLangASR("sarvam", "hi", "y"))
    assert (await route_transcribe(b"\x00" * 320, 16000)).asr_engine_used == "sarvam"


async def test_english_speech_stays_on_whisper(monkeypatch):
    sarvam_called = False

    class Boom(FixedLangASR):
        async def transcribe(self, *a, **k):
            nonlocal sarvam_called
            sarvam_called = True
            return await super().transcribe(*a, **k)

    monkeypatch.setattr("server.providers.asr_router.get_asr_en",
                        lambda: FixedLangASR("whisper", "en"))
    monkeypatch.setattr("server.providers.asr_router.get_asr_indic", lambda: Boom("s", "hi"))
    res = await route_transcribe(b"\x00" * 320, 16000)
    assert res.asr_engine_used == "whisper" and not sarvam_called


async def test_english_hint_does_not_force_whisper_language_and_reroutes_hindi(monkeypatch):
    from server.providers import asr_router
    from server.providers.base import Transcript

    seen = {}

    class En:
        async def transcribe(self, pcm, sr, hint_lang=None):
            seen["hint"] = hint_lang
            return Transcript("x", "hi", "Deva", 1.0, True, 1.0)

    class Indic:
        async def transcribe(self, pcm, sr, hint_lang=None):
            return Transcript("अशोक कौन था", "hi", "Deva", 1.0, True, 1.0)

    monkeypatch.setattr(asr_router, "get_asr_en", lambda: En())
    monkeypatch.setattr(asr_router, "get_asr_indic", lambda: Indic())
    out = await asr_router.route_transcribe(b"\0\0", 16000, hint_lang="en")
    assert out.text == "अशोक कौन था"      # English then Hindi: rerouted, not forced to English
