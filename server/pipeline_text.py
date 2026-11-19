"""Pure text helpers shared by the voice pipeline and the HTTP layer (no Pipecat import)."""

from __future__ import annotations

import re


def clean_tts_text(text: str) -> str:
    """Strips citation tags/markers and markdown so TTS speaks natural prose."""
    from server.rag.citations import strip_tags

    cleaned = strip_tags(text)
    cleaned = re.sub(r"\[\d+\]", "", cleaned)           # numbered markers
    cleaned = cleaned.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
    cleaned = re.sub(r"^\s*[-•]\s+", "", cleaned)
    cleaned = re.sub(r"\s+([.,!?।॥])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()
