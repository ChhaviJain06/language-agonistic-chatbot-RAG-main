"""Language + script detection.

fastText lid.176 gives a coarse language family guess (e.g. "hi", "en", "mr").
It is NOT trusted to tell "hi" apart from "hi-en" (Hinglish): romanized Hindi
gets misclassified by lid.176 constantly, because it was trained mostly on
native-script text. Instead, the same fastText guess flips between "hi" and
"hi-en" purely on which Unicode block the text is written in — script is the
ONLY signal for that distinction, by design.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    Path("data/models/lid.176.ftz"),
    Path("data/lid.176.bin"),
    Path("data/models/lid.176.bin"),
]


def _find_lid_model() -> Path:
    for c in _MODEL_CANDIDATES:
        if c.exists():
            return c
    return _MODEL_CANDIDATES[0]


DEFAULT_MODEL_PATH = _find_lid_model()
LOW_CONFIDENCE = 0.4

# Romanized-Hindi function words. English homographs ("the", "ne", "par", "tha"...)
# are deliberately excluded: they made plain English look like Hinglish.
HINGLISH_MARKERS = frozenset({
    "hai", "hain", "nahi", "nahin", "kya", "kaise", "kyun", "kyu", "thi", "tha",
    "ka", "ki", "ke", "mein", "hoon", "hun", "aur", "bhi", "karo", "kiya", "gaya",
    "gayi", "koi", "kuch", "yeh", "woh", "aap", "tum", "hum", "mera", "tera",
    "uska", "iska", "se", "ko", "kab", "kaun", "kahan", "kaunsa", "kitne", "batao",
    "bataiye", "samjhao", "baad", "pehle", "unhone", "uske", "iske", "wala", "wali",
})
_WORD_RE = re.compile(r"[a-zA-Z]+")
_MIN_MARKER_HITS = 1


def _looks_hinglish(text: str) -> bool:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if w in HINGLISH_MARKERS)
    # one marker is enough for short questions; longer text needs a visible share.
    return hits >= _MIN_MARKER_HITS and (len(words) <= 8 or hits / len(words) >= 0.12)


# Unicode block ranges for the scripts this project actually sees. "Common"
# codepoints (digits, punctuation, whitespace) never count toward any script,
# so a script histogram isn't diluted by numbers or "." on either side.
SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("Deva", 0x0900, 0x097F),   # Devanagari: Hindi, Marathi, Sanskrit, Nepali
    ("Beng", 0x0980, 0x09FF),   # Bengali, Assamese
    ("Guru", 0x0A00, 0x0A7F),   # Gurmukhi: Punjabi
    ("Gujr", 0x0A80, 0x0AFF),   # Gujarati
    ("Orya", 0x0B00, 0x0B7F),   # Odia
    ("Taml", 0x0B80, 0x0BFF),   # Tamil
    ("Telu", 0x0C00, 0x0C7F),   # Telugu
    ("Knda", 0x0C80, 0x0CFF),   # Kannada
    ("Mlym", 0x0D00, 0x0D7F),   # Malayalam
    ("Arab", 0x0600, 0x06FF),   # Arabic/Urdu — Whisper often renders Hindi audio this way
    ("Arab", 0x0750, 0x077F),
    ("Latn", 0x0041, 0x005A),   # Latin uppercase
    ("Latn", 0x0061, 0x007A),   # Latin lowercase
    ("Latn", 0x00C0, 0x024F),   # Latin-1 supplement / extended (IAST diacritics)
]

# Which script each language is written in when answering.
LANG_SCRIPT = {
    "en": "Latn", "hi-en": "Latn", "hi": "Deva", "mr": "Deva", "ne": "Deva", "sa": "Deva",
    "bn": "Beng", "as": "Beng", "pa": "Guru", "gu": "Gujr", "or": "Orya",
    "ta": "Taml", "te": "Telu", "kn": "Knda", "ml": "Mlym", "ur": "Arab",
}


def script_for_lang(lang: str) -> str:
    """Script a reply in `lang` must use. Unknown languages default to Latin."""
    return LANG_SCRIPT.get(lang) or LANG_SCRIPT.get(lang.split("-")[0], "Latn")


@dataclass(frozen=True)
class Detection:
    lang: str       # BCP-47-ish: "en", "hi", "hi-en", "mr", ...
    script: str     # ISO 15924: "Latn", "Deva", "Beng", ...
    confidence: float


def script_histogram(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ch in text:
        cp = ord(ch)
        if unicodedata.category(ch)[0] in ("N", "P", "Z", "C", "S"):
            continue
        for script, lo, hi in SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


def majority_script(text: str) -> str:
    hist = script_histogram(text)
    if not hist:
        return "Latn"  # no lettered chars at all — default rather than guess
    return max(hist, key=hist.get)


class LanguageDetector:
    def __init__(self, model_path: Path | str | None = None):
        self._model_path = Path(model_path) if model_path else _find_lid_model()
        self._model = None

    def load(self) -> None:
        if self._model is None:
            import fasttext

            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"fastText model not found at {self._model_path}. "
                    "Run: python scripts/download_models.py"
                )
            self._model = fasttext.load_model(str(self._model_path))

    async def _ensure_loaded(self) -> None:
        if self._model is None:
            await asyncio.to_thread(self.load)

    async def detect(self, text: str) -> Detection:
        text = text.strip()
        if not text:
            return Detection(lang="en", script="Latn", confidence=0.0)

        script = majority_script(text)
        try:
            await self._ensure_loaded()
        except (ImportError, FileNotFoundError) as exc:
            # No model: degrade to script-only detection instead of failing the turn.
            logger.warning("fastText unavailable (%s) — using script-only detection", exc)
            return self._script_only(text, script)

        # fastText chokes on embedded newlines; a query/utterance is one line anyway.
        # Calling the underlying pybind method directly, not FastText.predict():
        # that wrapper does `np.array(probs, copy=False)`, which raises under numpy>=2.0.
        line = text.replace("\n", " ") + "\n"
        predictions = self._model.f.predict(line, 1, 0.0, "strict")
        if not predictions:
            return self._script_only(text, script)
        prob, label = predictions[0]
        base_lang = label.removeprefix("__label__")
        confidence = float(prob)

        if script != "Latn":
            # Urdu-script output for a Hindi-family language: treat as Hindi (Whisper quirk).
            if script == "Arab" and base_lang not in {"ur", "ar", "fa", "ps", "sd"}:
                base_lang = "ur"
            return Detection(lang=base_lang, script=script, confidence=confidence)

        if base_lang == "en" and confidence >= LOW_CONFIDENCE and not _looks_hinglish(text):
            return Detection(lang="en", script="Latn", confidence=confidence)

        # lid.176 is unreliable on romanized Hindi (e.g. "Ashoka ne kaunsa yudh lada tha?"
        # gets guessed as "la"). A marker-word check decides the Hinglish case.
        if _looks_hinglish(text):
            return Detection(lang="hi-en", script="Latn", confidence=confidence)

        if base_lang == "en" or confidence < LOW_CONFIDENCE:
            return Detection(lang="en", script="Latn", confidence=confidence)
        return Detection(lang=f"{base_lang}-en", script="Latn", confidence=confidence)

    @staticmethod
    def _script_only(text: str, script: str) -> Detection:
        if script == "Latn":
            return Detection("hi-en" if _looks_hinglish(text) else "en", "Latn", 0.0)
        lang = {"Deva": "hi", "Beng": "bn", "Taml": "ta", "Telu": "te", "Knda": "kn",
                "Mlym": "ml", "Gujr": "gu", "Guru": "pa", "Orya": "or", "Arab": "ur"}.get(
                    script, "en")
        return Detection(lang, script, 0.0)


_detector: LanguageDetector | None = None


def get_detector() -> LanguageDetector:
    global _detector
    if _detector is None:
        _detector = LanguageDetector()
    return _detector
