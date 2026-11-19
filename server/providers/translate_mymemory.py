"""Translation via MyMemory (deep-translator wrapper). Free, no API key, open library.

Why this exists next to the Google provider: Google's unofficial endpoint answers
"TooManyRequests" from many home/college networks (it did on the dev machine), while
MyMemory worked. Verified live for hi/hinglish->en and en->hi/mr/bn/ta.

Differences from Google you must know about:
  * no "auto" source: we guess it from the text's Unicode script;
  * each request is limited to ~500 BYTES (Devanagari is 3 bytes/char), so long text is
    split into sentence chunks;
  * anonymous daily quota (~5000 chars/day); quota errors return None like any failure;
  * ~1.5-2 s per request, noticeably slower than Google.
Like every translation provider it sends text off-machine, so `onprem` forces `none`.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict

from server.lang.detect import majority_script
from server.providers.base import TranslationProvider

logger = logging.getLogger(__name__)

TIMEOUT_S = 6.0
MAX_CHUNK_BYTES = 400          # safely under MyMemory's 500-byte limit
_CACHE_MAX = 512

# Our language codes -> the region codes MyMemory requires.
CODES = {
    "en": "en-GB", "hi": "hi-IN", "hi-en": "hi-IN", "mr": "mr-IN", "bn": "bn-IN",
    "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN", "gu": "gu-IN",
    "pa": "pa-IN", "or": "or-IN", "ur": "ur-PK", "as": "as-IN", "ne": "ne-NP",
}
# What a text's script most likely is, when the caller says "auto".
SCRIPT_TO_LANG = {
    "Latn": "en", "Deva": "hi", "Beng": "bn", "Taml": "ta", "Telu": "te", "Knda": "kn",
    "Mlym": "ml", "Gujr": "gu", "Guru": "pa", "Orya": "or", "Arab": "ur",
}
_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")


def chunk_text(text: str, max_bytes: int = MAX_CHUNK_BYTES) -> list[str]:
    """Sentence-aligned chunks no larger than max_bytes (UTF-8); oversize sentences are
    hard-split so a request can never be rejected for length."""
    chunks: list[str] = []
    current = ""
    for sent in _SPLIT.split(text.strip()):
        while len(sent.encode("utf-8")) > max_bytes:
            cut = max(1, max_bytes // 3)
            while len(sent[:cut].encode("utf-8")) < max_bytes and cut < len(sent):
                cut += 1
            while len(sent[:cut].encode("utf-8")) > max_bytes:
                cut -= 1
            if current:
                chunks.append(current)
                current = ""
            chunks.append(sent[:cut])
            sent = sent[cut:]
        candidate = f"{current} {sent}".strip()
        if len(candidate.encode("utf-8")) > max_bytes and current:
            chunks.append(current)
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


class MyMemoryTranslator(TranslationProvider):
    name = "mymemory"

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()

    @staticmethod
    def _translate_sync(text: str, source: str, target: str) -> str:
        from deep_translator import MyMemoryTranslator as _MM

        return _MM(source=source, target=target).translate(text)

    async def translate(self, text: str, target_lang: str,
                        source_lang: str = "auto") -> str | None:
        text = (text or "").strip()
        if not text:
            return None
        src_code = source_lang if source_lang != "auto" else \
            SCRIPT_TO_LANG.get(majority_script(text), "en")
        source, target = CODES.get(src_code), CODES.get(target_lang)
        if not source or not target:
            logger.debug("mymemory: unsupported pair %s->%s", src_code, target_lang)
            return None
        if source == target:
            return None
        key = (text, source, target)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            async with asyncio.timeout(TIMEOUT_S):
                parts = await asyncio.gather(*(
                    asyncio.to_thread(self._translate_sync, c, source, target)
                    for c in chunk_text(text)))
        except Exception as exc:  # noqa: BLE001 - best-effort; callers have fallbacks
            logger.warning("mymemory translation failed (%s->%s): %r", source, target, exc)
            return None
        out = " ".join(p.strip() for p in parts if p).strip()
        if not out or out.lower() == text.lower():
            return None
        self._cache[key] = out
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
        return out

    async def health(self) -> bool:
        return await self.translate("hello", "hi", "en") is not None
