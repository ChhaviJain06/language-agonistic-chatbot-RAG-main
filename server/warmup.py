"""Startup warm-up: pay one-off costs (TLS handshakes, model loading, DB pool) before the
first user turn instead of during it. Everything here is best-effort and logged — a failed
warm-up never blocks startup; the same call simply happens lazily on first use.
"""

from __future__ import annotations

import asyncio
import logging
import time

from server.lang.detect import get_detector
from server.providers import factory
from server.rag.db import get_pool

logger = logging.getLogger(__name__)


async def _timed(name: str, coro) -> tuple[str, float, str | None]:
    start = time.monotonic()
    try:
        await coro
        return name, (time.monotonic() - start) * 1000, None
    except Exception as exc:  # noqa: BLE001 - warm-up must never take the app down
        return name, (time.monotonic() - start) * 1000, repr(exc)


async def warm_all() -> None:
    tasks = [
        _timed("database pool", get_pool()),
        _timed("language detector", get_detector()._ensure_loaded()),
        _timed("llm", factory.get_llm().warm()),
        _timed("asr (en)", factory.get_asr_en().warm()),
        _timed("asr (indic)", factory.get_asr_indic().warm()),
        _timed("tts", factory.get_tts().warm()),
        _timed("embedder", factory.get_embedder().warm()),
        _timed("reranker", factory.get_reranker().warm()),
    ]
    results = await asyncio.gather(*tasks)
    summary = ", ".join(f"{n} {ms:.0f}ms{' FAILED' if err else ''}" for n, ms, err in results)
    logger.info("warm-up done: %s", summary)
    for name, _ms, err in results:
        if err:
            logger.warning("warm-up of %s failed (will retry lazily): %s", name, err)
