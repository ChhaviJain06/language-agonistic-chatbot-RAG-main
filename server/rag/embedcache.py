"""One embedding per distinct question per process.

A single turn used to embed the same question up to three times (semantic cache,
dense retrieval, cache store). Every embedding is a network round-trip (Cohere) or
seconds of CPU (bge-m3), so they all go through this small LRU instead.
"""

from __future__ import annotations

from collections import OrderedDict

from server.providers.factory import get_embedder

_MAX = 512
_cache: OrderedDict[str, list[float]] = OrderedDict()


async def embed_query(text: str) -> list[float]:
    key = text.strip()
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    vec = (await get_embedder().embed([key], is_query=True))[0]
    _cache[key] = vec
    if len(_cache) > _MAX:
        _cache.popitem(last=False)
    return vec


def clear() -> None:
    _cache.clear()
