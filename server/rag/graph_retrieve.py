"""Multi-hop/comparative-only graph augmentation (C12), merged into retrieval by
orchestrate.py.

Entity extraction goes through the LLM, not a capitalisation regex: the golden set's
multi_hop/comparative questions span en, hi-deva, hinglish-latn and mixed script, but
the graph's node names are English Wikipedia titles. Bridging "गांधी" -> "Gandhi" is
cross-lingual entity linking, which needs a multilingual model.

Every failure path returns "nothing found" (C11); nothing here may take a turn down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

from server.config import get_settings
from server.providers.base import GraphHit, LLMProvider
from server.providers.factory import get_graph_provider, get_llm

logger = logging.getLogger(__name__)

ENTITY_EXTRACT_PROMPT = """Identify the key named entities (people, dynasties,
places, battles, events) this question is about, using their standard ENGLISH
Wikipedia article names, even if the question is in Hindi, Hinglish, or mixed
script. Output strict JSON: {"entities": [str, ...]}. Return at most 3.
"""

_HEALTH_TTL = 30.0
_health_cache: tuple[float, bool] | None = None


async def _graph_is_up(graph) -> bool:
    """Health check cached for 30s — it used to add a full round-trip to every turn."""
    global _health_cache
    now = time.monotonic()
    if _health_cache and now - _health_cache[0] < _HEALTH_TTL:
        return _health_cache[1]
    ok = await graph.health()
    _health_cache = (now, ok)
    return ok


def reset_health_cache() -> None:
    global _health_cache
    _health_cache = None


async def extract_entities(question: str, llm: LLMProvider | None = None) -> list[str]:
    llm = llm or get_llm()
    messages = [{"role": "system", "content": ENTITY_EXTRACT_PROMPT},
                {"role": "user", "content": question}]
    if hasattr(llm, "complete"):
        reply = await llm.complete(messages, json_mode=True, max_tokens=800, max_wait=4.0)
    else:
        reply = "".join([c async for c in llm.stream(messages, None, [])])
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        return []
    entities = parsed.get("entities", []) if isinstance(parsed, dict) else parsed
    if not isinstance(entities, list):
        return []
    return [e.strip() for e in entities if isinstance(e, str) and e.strip()][:3]


def _path_key(hit: GraphHit) -> tuple:
    return tuple(tuple(sorted(step.items())) for step in hit.path)


async def graph_retrieve(question: str, question_type: str,
                         entities: list[str] | None = None) -> list[GraphHit]:
    """[] immediately for any type outside GRAPH_TRIGGER_TYPES (C12). [] (never raises)
    if the graph is unreachable, slow or the LLM linking step fails (C11)."""
    s = get_settings()
    if question_type not in s.graph_trigger_types_list:
        return []

    graph = get_graph_provider()
    if not await _graph_is_up(graph):
        logger.warning("graph unreachable — skipping graph augmentation (C11)")
        return []

    try:
        async with asyncio.timeout(s.graph_timeout_s + 3.0):
            if entities is None:
                entities = await extract_entities(question)
            if not entities:
                return []
            results = await asyncio.gather(
                *(asyncio.wait_for(graph.query_entity(e, max_hops=s.graph_max_hops),
                                   s.graph_timeout_s) for e in entities),
                return_exceptions=True)
    except Exception as exc:  # noqa: BLE001 - C11: graph trouble never fails the answer
        logger.warning("graph augmentation skipped: %r", exc)
        return []

    hits: list[GraphHit] = []
    seen: set[tuple] = set()
    for res in results:
        if isinstance(res, BaseException):
            logger.warning("graph query failed: %r", res)
            continue
        for hit in res:
            key = _path_key(hit)
            if key not in seen:
                seen.add(key)
                hits.append(hit)
    hits.sort(key=lambda h: h.confidence or 0.0, reverse=True)
    return hits[:10]
