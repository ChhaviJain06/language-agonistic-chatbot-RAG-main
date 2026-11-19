"""Per-call state shared by the voice pipeline's processors."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

MAX_HISTORY_MESSAGES = 6   # sliding window = 3 exchanges

# Spoken when a turn fails, so the user is never left in silence (in their language).
APOLOGIES = {
    "en": "Sorry, something went wrong. Please ask again.",
    "hi": "क्षमा करें, कुछ गड़बड़ हो गई। कृपया फिर से पूछें।",
    "hi-en": "Sorry, kuch gadbad ho gayi. Kripya dobara poochhiye.",
    "mr": "क्षमस्व, काहीतरी चूक झाली. कृपया पुन्हा विचारा.",
    "bn": "দুঃখিত, কিছু ভুল হয়েছে। অনুগ্রহ করে আবার জিজ্ঞাসা করুন।",
    "ta": "மன்னிக்கவும், ஏதோ தவறு நடந்தது. தயவுசெய்து மீண்டும் கேளுங்கள்.",
}


def apology_for(lang: str) -> str:
    return APOLOGIES.get(lang) or APOLOGIES.get(lang.split("-")[0]) or APOLOGIES["en"]


@dataclass
class SessionState:
    room: str
    conversation_id: str | None = None
    last_lang: str | None = None          # previous turn's language -> ASR routing hint
    last_asr_ms: float | None = None      # ASR latency of the utterance that started the turn
    history: list[dict] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    def remember(self, question: str, answer: str | None) -> None:
        """Adds an exchange to the sliding history window. `answer` is what the user
        actually HEARD (after a barge-in), or None if nothing was heard."""
        self.history.append({"role": "user", "content": question})
        if answer and answer.strip():
            self.history.append({"role": "assistant", "content": answer})
        self.history = self.history[-MAX_HISTORY_MESSAGES:]
