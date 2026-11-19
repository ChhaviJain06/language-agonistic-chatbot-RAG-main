"""Dual-path ASR router: Indic speech and English/code-mixed speech go to different
engines, chosen once per utterance from a hint (the previous turn's detected language),
not from decoding audio twice.

First utterance of a session has no hint, so it goes to the English/Whisper engine. If
Whisper reports an Indic language, the utterance is transcribed ONCE more by the Indic
engine, and the session hint is set for every later turn.
"""

from __future__ import annotations

import asyncio
import logging

from server.providers.base import ASRProvider, Transcript
from server.providers.factory import get_asr_en, get_asr_indic

logger = logging.getLogger(__name__)

# Bare language families that route to the Indic-tuned engine. "en" and code-mixed
# "<lang>-en" both go to the English/code-mixed engine — it handles code-switching.
INDIC_LANGS = {"hi", "mr", "bn", "ta", "te", "kn", "ml", "gu", "pa", "or", "as", "ne", "sa"}
# Whisper labels Hindi audio as Urdu surprisingly often; treat it as Hindi for routing.
REROUTE_LANGS = INDIC_LANGS | {"ur"}

SARVAM_TIMEOUT_SECONDS = 4.0


def choose_route(hint_lang: str | None) -> str:
    """Returns "indic" or "en" — pure function so routing is unit-testable."""
    if not hint_lang:
        return "en"
    base = hint_lang.split("-")[0]
    return "indic" if base in INDIC_LANGS and not hint_lang.endswith("-en") else "en"


async def _transcribe_indic(pcm: bytes, sample_rate: int, hint: str | None) -> Transcript:
    provider: ASRProvider = get_asr_indic()
    try:
        async with asyncio.timeout(SARVAM_TIMEOUT_SECONDS):
            return await provider.transcribe(pcm, sample_rate, hint_lang=hint)
    except Exception as exc:  # noqa: BLE001 - any Indic-engine failure falls back to Whisper
        logger.warning("Indic ASR (%s) failed or timed out (>%.1fs): %s — falling back to "
                       "English/Whisper ASR", provider.name, SARVAM_TIMEOUT_SECONDS, exc)
        return await get_asr_en().transcribe(pcm, sample_rate, hint_lang=hint)


async def route_transcribe(pcm: bytes, sample_rate: int,
                           hint_lang: str | None = None) -> Transcript:
    route = choose_route(hint_lang)
    logger.info("asr_router: hint_lang=%r -> route=%s", hint_lang, route)

    if route == "indic":
        return await _transcribe_indic(pcm, sample_rate, hint_lang)

    first = await get_asr_en().transcribe(pcm, sample_rate, hint_lang=hint_lang)
    base = first.lang.split("-")[0]
    if hint_lang in (None, "en") and base in REROUTE_LANGS and first.text.strip():
        indic_hint = "hi" if base == "ur" else base
        logger.info("asr_router: Whisper heard %r on an unforced turn — re-routing to Indic",
                    first.lang)
        second = await _transcribe_indic(pcm, sample_rate, indic_hint)
        return second if second.text.strip() else first
    return first
