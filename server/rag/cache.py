"""Semantic question cache (pgvector).

Only answers that are SAFE to replay are stored: they passed the fidelity contract, they
carry verified citations, and they were produced without conversation history (a
follow-up like "and his son?" must never be served from an unrelated cached answer).
Lookups are partitioned by language, script and embedding model, so a Hindi question is
never answered with a cached English reply.

If Postgres is unreachable this degrades to a cache miss (C11-style), never an error.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from server.config import get_settings
from server.rag.db import get_pool
from server.rag.embedcache import embed_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CacheHit:
    id: str
    query_text: str
    response_text: str
    audio_cache_path: str | None
    metadata: dict[str, Any]
    similarity: float


# ORDER BY the raw distance operator so the HNSW index is used; the threshold is
# applied in Python (filtering on a computed expression forced a sequential scan).
_MATCH_SQL = """
SELECT id, query_text, response_text, audio_cache_path, metadata,
       1 - (query_embedding <=> $1) AS similarity
FROM query_cache
WHERE lang = $2 AND script = $3 AND embed_model = $4
ORDER BY query_embedding <=> $1
LIMIT 1;
"""

_INSERT_SQL = """
INSERT INTO query_cache (query_text, query_embedding, response_text, audio_cache_path,
                         metadata, lang, script, embed_model)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING id;
"""


async def check_semantic_cache(query: str, lang: str, script: str,
                               similarity_threshold: float | None = None) -> CacheHit | None:
    s = get_settings()
    if not s.cache_enabled:
        return None
    threshold = similarity_threshold if similarity_threshold is not None else s.cache_similarity
    try:
        vec = await embed_query(query)
        pool = await get_pool()
        row = await pool.fetchrow(_MATCH_SQL, vec, lang, script, s.embed_model_id)
        if not row or float(row["similarity"]) < threshold:
            return None
        meta_raw = row["metadata"]
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})
        hit = CacheHit(str(row["id"]), row["query_text"], row["response_text"],
                       row["audio_cache_path"], meta, float(row["similarity"]))
        logger.info("semantic cache HIT (%.4f) for %r", hit.similarity, query)
        return hit
    except Exception as exc:  # noqa: BLE001 - a cache problem is a miss, never an error
        logger.warning("semantic cache lookup skipped: %s", exc)
        return None


async def store_semantic_cache(query: str, response_text: str, lang: str, script: str,
                               metadata: dict[str, Any] | None = None,
                               audio_cache_path: str | None = None) -> str | None:
    s = get_settings()
    if not s.cache_enabled:
        return None
    try:
        vec = await embed_query(query)
        pool = await get_pool()
        cache_id = await pool.fetchval(_INSERT_SQL, query, vec, response_text, audio_cache_path,
                                       json.dumps(metadata or {}), lang, script,
                                       s.embed_model_id)
        return str(cache_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not store in semantic cache: %s", exc)
        return None
