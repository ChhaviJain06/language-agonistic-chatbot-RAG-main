"""Translation via deep-translator's Google Translate web endpoint (free, open-source
library, NO API key). Idea taken from the author's FAQ-chatbot project.

Honest caveats — why this is optional and OFF for `onprem`:
  * it sends the user's text to Google (violates C5, so onprem forces TRANSLATION_PROVIDER=none);
  * it is an unofficial endpoint: it can rate-limit or change, so every caller has a fallback;
  * deep-translator is blocking `requests` code, so it runs in a thread (C6) with a timeout.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

from server.providers.base import TranslationProvider

logger = logging.getLogger(__name__)

TIMEOUT_S = 3.0
MAX_CHARS = 1500           # the free endpoint rejects very long inputs
_CACHE_MAX = 512

# Google uses language NAMES for a few codes where ours are ISO.
_CODE_MAP = {"or": "odia", "as": "assamese", "hi-en": "hi"}


class GoogleTranslator(TranslationProvider):
    name = "google-free"

    def __init__(self) -> None:
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._failures = 0

    @staticmethod
    def _translate_sync(text: str, source: str, target: str) -> str:
        from deep_translator import GoogleTranslator as _GT

        return _GT(source=source, target=target).translate(text)

    async def translate(self, text: str, target_lang: str,
                        source_lang: str = "auto") -> str | None:
        text = (text or "").strip()[:MAX_CHARS]
        if not text:
            return None
        target = _CODE_MAP.get(target_lang, target_lang)
        key = (text, source_lang, target)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        try:
            async with asyncio.timeout(TIMEOUT_S):
                out = await asyncio.to_thread(self._translate_sync, text, source_lang, target)
        except Exception as exc:  # noqa: BLE001 - translation is best-effort, callers fall back
            self._failures += 1
            logger.warning("translation failed (%s): %r", target, exc)
            return None
        out = (out or "").strip()
        if not out or out.lower() == text.lower():
            return None                       # nothing useful changed
        self._cache[key] = out
        if len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
        return out

    async def health(self) -> bool:
        return await self.translate("hello", "hi", "en") is not None


class NullTranslator(TranslationProvider):
    """TRANSLATION_PROVIDER=none (always for onprem): every caller takes its fallback path."""

    name = "none"

    async def translate(self, text: str, target_lang: str,
                        source_lang: str = "auto") -> str | None:
        return None

    async def health(self) -> bool:
        return False
