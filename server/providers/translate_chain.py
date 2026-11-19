"""Tries several translators in order and returns the first real answer.

This is what makes one codebase run everywhere: on a laptop with the local model installed
the chain is [local, mymemory]; on a small host without torch it is just [mymemory]; with
nothing available it is empty and every caller takes its no-translation fallback.
"""

from __future__ import annotations

import logging

from server.providers.base import TranslationProvider

logger = logging.getLogger(__name__)


class ChainTranslator(TranslationProvider):
    def __init__(self, providers: list[TranslationProvider]):
        self._providers = providers
        self.name = "chain(" + ",".join(p.name for p in providers) + ")" if providers else "none"

    async def translate(self, text: str, target_lang: str,
                        source_lang: str = "auto") -> str | None:
        for provider in self._providers:
            out = await provider.translate(text, target_lang, source_lang)
            if out:
                return out
            logger.debug("%s gave no translation, trying next", provider.name)
        return None

    async def health(self) -> bool:
        return any([await p.health() for p in self._providers])
