"""orchestrate.py — the single retrieval entry point.

Vector retrieval always runs: it must produce a valid answer on its own (C11). Graph
augmentation only runs for multi_hop/comparative questions (C12). The English entity
names the graph step extracts are reused as keywords for the sparse search, so one LLM
call improves both retrievers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from server.config import get_settings
from server.providers.base import GraphHit
from server.providers.factory import get_translator
from server.rag.embedcache import embed_query
from server.rag.graph_retrieve import extract_entities, graph_retrieve
from server.rag.retrieve import Hit, chunks_by_ids, retrieve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestrationResult:
    hits: list[Hit]              # vector hits, plus any graph-cited chunks not already present
    graph_paths: list[GraphHit]  # raw entity-relation chains, for citing the multi-hop reasoning
    vector_only_hits: list[Hit]  # kept separate so eval can report the vector-only baseline
    entities: tuple[str, ...] = ()


async def _safe_entities(question: str) -> list[str]:
    try:
        async with asyncio.timeout(get_settings().graph_timeout_s + 2.0):
            return await extract_entities(question)
    except Exception as exc:  # noqa: BLE001 - C11
        logger.warning("entity extraction skipped: %r", exc)
        return []


async def _safe_graph(question: str, question_type: str, entities: list[str]) -> list[GraphHit]:
    try:
        return await graph_retrieve(question, question_type, entities=entities or None)
    except Exception as exc:  # noqa: BLE001 - C11: never let the graph fail a turn
        logger.warning("graph retrieval failed: %r", exc)
        return []


async def _english_query(question: str, source_lang: str) -> str | None:
    """English version of a non-English question (None if translation is off/failed).
    Its words become keywords for the sparse search, so a Hindi question can hit the
    English keyword index without waiting for the LLM entity step."""
    return await get_translator().translate(question, "en", source_lang)


async def orchestrate(question: str, question_type: str = "factoid",
                      top_k: int | None = None, rerank_top_k: int | None = None,
                      mode: str | None = None,
                      translate_from: str | None = None) -> OrchestrationResult:
    s = get_settings()
    if mode is None:
        # Plain factoid questions skip the ~0.5 s rerank when configured; harder types keep it.
        mode = "hybrid_norerank" if question_type == "factoid" and s.skip_rerank_factoid \
            else "hybrid"
    wants_graph = question_type in s.graph_trigger_types_list
    need_vec = mode != "sparse"

    # Independent network calls run together: embedding, entity linking and translation.
    entities, vec, english = await asyncio.gather(
        _safe_entities(question) if wants_graph else _none_list(),
        embed_query(question) if need_vec else _none(),
        _english_query(question, translate_from)
        if translate_from and get_settings().translate_queries else _none(),
    )
    keywords = [*entities, *([english] if english else [])]

    vector_hits, graph_paths = await asyncio.gather(
        retrieve(question, top_k=top_k, rerank_top_k=rerank_top_k, mode=mode,
                 keywords=keywords, query_vec=vec),
        _safe_graph(question, question_type, entities) if wants_graph else _none_list(),
    )

    seen = {h.chunk_id for h in vector_hits}
    graph_chunk_ids = [cid for gh in graph_paths for cid in gh.source_chunk_ids if cid not in seen]
    try:
        graph_hits = await chunks_by_ids(list(dict.fromkeys(graph_chunk_ids))[:5])
    except Exception as exc:  # noqa: BLE001 - C11
        logger.warning("could not load graph-cited chunks: %r", exc)
        graph_hits = []

    return OrchestrationResult(
        hits=[*vector_hits, *graph_hits],
        graph_paths=graph_paths,
        vector_only_hits=vector_hits,
        entities=tuple(entities),
    )


async def _none():
    return None


async def _none_list() -> list:
    return []
