"""Hybrid retrieval: dense + sparse, fused with RRF, then cross-encoder reranked.

Why hybrid. Dense (Cohere / bge-m3) is what makes a Hindi question find an English chunk,
but it is fuzzy about rare proper nouns and dates. Postgres full-text is exact on
"Plassey" or "1757" and useless on Devanagari. They fail in different places, so fusing
them beats either alone — that claim is what scripts/eval_retrieval.py exists to test.

The sparse side uses OR semantics over the query's Latin tokens plus `keywords` (English
entity names the graph step already extracted). That second part is what lets a Hindi
question like "अशोक ने कलिंग…" hit the keyword index at all.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

from server.config import get_settings
from server.providers.factory import get_reranker
from server.rag.db import get_pool
from server.rag.embedcache import embed_query

logger = logging.getLogger(__name__)

RRF_K = 60   # standard damping constant; keeps rank-1 from dominating the fusion

_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its of on or "
    "that the their this to was were what when where which who whom why with about after "
    "before into than then there these they those during also not but".split())
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{2,}")


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    article: str
    section: str
    text: str
    score: float
    url: str = ""


_DENSE_SQL = """
SELECT chunk_id, article, section, text, url
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1
LIMIT $2
"""

# ts_rank_cd over a weighted tsvector, not textbook BM25 — Postgres ships this, and
# pulling in pg_search just for Okapi weighting is not worth a new service.
_SPARSE_SQL = """
SELECT chunk_id, article, section, text, url, ts_rank_cd(tsv, q) AS rank
FROM chunks, to_tsquery('english', $1) q
WHERE tsv @@ q
ORDER BY rank DESC
LIMIT $2
"""


def build_or_tsquery(query: str, keywords: list[str] | None = None) -> str | None:
    """'a | b | c' tsquery string from alphanumeric tokens only — injection-safe because no
    tsquery operator characters can survive the tokenizer. None if nothing usable."""
    tokens: list[str] = []
    for text in [query, *(keywords or [])]:
        for t in _TOKEN_RE.findall(text):
            tl = t.lower()
            if tl not in _STOPWORDS and tl not in tokens:
                tokens.append(tl)
    return " | ".join(tokens[:24]) if tokens else None


def _hit(c: dict, score: float | None = None) -> Hit:
    return Hit(c["chunk_id"], c["article"], c["section"], c["text"],
               float(score if score is not None else c.get("score", c.get("rank", 0.0))),
               c.get("url", "") or "")


async def _dense(query: str, k: int, vec: list[float] | None = None) -> list[dict]:
    vec = vec if vec is not None else await embed_query(query)
    pool = await get_pool()
    return [dict(r) for r in await pool.fetch(_DENSE_SQL, vec, k)]


async def _sparse(query: str, k: int, keywords: list[str] | None = None) -> list[dict]:
    tsq = build_or_tsquery(query, keywords)
    if not tsq:
        return []
    pool = await get_pool()
    return [dict(r) for r in await pool.fetch(_SPARSE_SQL, tsq, k)]


def rrf(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion. Uses rank position only, so the dense cosine score and the
    ts_rank_cd score never have to be made commensurable."""
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for pos, row in enumerate(ranking, start=1):
            entry = fused.setdefault(row["chunk_id"], {**row, "score": 0.0})
            entry["score"] += 1.0 / (k + pos)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


STATS = {"rerank_fallbacks": 0}   # the evaluation reads this to prove it measured the reranker


async def retrieve(query: str, top_k: int | None = None, rerank_top_k: int | None = None,
                   mode: str = "hybrid", keywords: list[str] | None = None,
                   query_vec: list[float] | None = None) -> list[Hit]:
    """mode: "hybrid" | "dense" | "sparse" | "hybrid_norerank" — the non-default modes
    exist so the evaluation can ablate them."""
    s = get_settings()
    top_k = top_k or s.retrieve_top_k
    rerank_top_k = rerank_top_k or s.rerank_top_k

    if mode == "dense":
        candidates = await _dense(query, top_k, query_vec)
    elif mode == "sparse":
        candidates = await _sparse(query, top_k, keywords)
    else:
        dense_c, sparse_c = await asyncio.gather(
            _dense(query, top_k, query_vec), _sparse(query, top_k, keywords))
        candidates = rrf([dense_c, sparse_c])

    if not candidates:
        return []

    if mode in ("dense", "sparse", "hybrid_norerank"):
        return [_hit(c) for c in candidates[:rerank_top_k]]

    try:
        to_rerank = candidates[: max(s.rerank_candidates, rerank_top_k)]
        async with asyncio.timeout(s.rerank_timeout_s):
            scores = await get_reranker().score(query, [c["text"] for c in to_rerank])
        ranked = sorted(zip(to_rerank, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [_hit(c, sc) for c, sc in ranked[:rerank_top_k]]
    except Exception as exc:  # noqa: BLE001 - rerank is an upgrade, RRF order is the fallback
        STATS["rerank_fallbacks"] += 1
        logger.warning("reranking failed or timed out (%s) — using RRF order", exc)
        return [_hit(c) for c in candidates[:rerank_top_k]]


async def chunks_by_ids(chunk_ids: list[str]) -> list[Hit]:
    """Fetches specific chunks by id (the ones cited on a graph path)."""
    if not chunk_ids:
        return []
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT chunk_id, article, section, text, url FROM chunks WHERE chunk_id = ANY($1)",
        chunk_ids)
    by_id = {r["chunk_id"]: r for r in rows}
    return [Hit(r["chunk_id"], r["article"], r["section"], r["text"], 0.0, r["url"] or "")
            for cid in chunk_ids if (r := by_id.get(cid))]
