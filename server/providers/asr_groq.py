"""Groq-hosted whisper-large-v3-turbo — the English/code-mixed leg of the dual-path ASR
router (free tier). It has its own request-count rate limit, separate from the
chat-completions token pool.

Uses verbose_json so we get the detected language and per-segment no-speech
probabilities, which lets us drop Whisper's well-known "Thanks for watching!"
hallucinations on silence instead of sending them to the LLM.
"""

from __future__ import annotations

import math
import time

import httpx

from server.config import get_settings
from server.lang.detect import majority_script
from server.providers._http import make_client
from server.providers._pcm import pcm_to_wav_bytes
from server.providers.base import ASRProvider, Transcript

TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODELS_URL = "https://api.groq.com/openai/v1/models"
MODEL = "whisper-large-v3-turbo"
NO_SPEECH_DROP = 0.6

# Whisper returns language NAMES, not codes.
WHISPER_LANG_CODES = {
    "english": "en", "hindi": "hi", "marathi": "mr", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "kannada": "kn", "malayalam": "ml", "gujarati": "gu", "punjabi": "pa",
    "urdu": "ur", "nepali": "ne", "sanskrit": "sa", "assamese": "as",
}
# A romanised-Hindi sample biases Whisper toward Latin-script output for Hinglish speech.
HINGLISH_PROMPT = "Ashoka ne Kalinga ke baad kya kiya? Akbar aur Aurangzeb mein kya antar tha?"


def _confidence(segments: list[dict]) -> float:
    lps = [s["avg_logprob"] for s in segments if "avg_logprob" in s]
    return round(math.exp(sum(lps) / len(lps)), 3) if lps else 1.0


class GroqASR(ASRProvider):
    name = "groq-whisper"

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().groq_api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = make_client()
        return self._client

    async def transcribe(self, pcm: bytes, sample_rate: int,
                         hint_lang: str | None = None) -> Transcript:
        wav = pcm_to_wav_bytes(pcm, sample_rate)
        files = {"file": ("audio.wav", wav, "audio/wav")}
        data = {"model": MODEL, "response_format": "verbose_json", "temperature": "0"}
        if hint_lang == "hi-en":
            data["language"] = "en"           # keeps Latin script for Hinglish speakers
            data["prompt"] = HINGLISH_PROMPT
        elif hint_lang and hint_lang != "en":
            # "en" is only the LAST turn's language: forcing it would make Whisper write spoken
            # Hindi as English text. Let Whisper detect; the router reroutes Indic results.
            data["language"] = hint_lang.split("-")[0]  # whisper wants bare "en", not "en-US"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        start = time.monotonic()
        resp = await self._get_client().post(TRANSCRIBE_URL, files=files, data=data,
                                             headers=headers)
        resp.raise_for_status()
        body = resp.json()
        latency_ms = (time.monotonic() - start) * 1000

        segments = body.get("segments") or []
        text = (body.get("text") or "").strip()
        if segments and all(s.get("no_speech_prob", 0.0) > NO_SPEECH_DROP for s in segments):
            text = ""      # silence/noise: Whisper hallucinates here, so drop it
        detected = WHISPER_LANG_CODES.get(str(body.get("language", "")).lower(),
                                          (hint_lang or "en").split("-")[0])
        return Transcript(
            text=text, lang=hint_lang if hint_lang == "hi-en" else detected,
            script=majority_script(text) if text else "Latn",
            confidence=_confidence(segments), is_final=True, latency_ms=latency_ms,
            asr_engine_used=self.name,
        )

    async def warm(self) -> None:
        await self.health()

    async def health(self) -> bool:
        try:
            resp = await self._get_client().get(
                MODELS_URL, headers={"Authorization": f"Bearer {self._api_key}"}, timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
