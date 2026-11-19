"""Shared asyncpg pool + schema/embedding-model safety checks.

asyncpg speaks the wire protocol directly, so the SQLAlchemy-style `+asyncpg` in
DATABASE_URL has to be stripped first.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from server.config import ConfigError, get_settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()


def dsn() -> str:
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


async def _init(conn: asyncpg.Connection) -> None:
    await register_vector(conn)      # lets us pass/receive python lists as vector


async def get_pool() -> asyncpg.Pool:
    global _pool
    async with _pool_lock:      # warm-up and the first request must not both create a pool
        if _pool is None:
            # The vector type must exist before register_vector runs on a pooled connection.
            boot = await asyncpg.connect(dsn(), timeout=5)
            try:
                await boot.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                await boot.close()
            _pool = await asyncpg.create_pool(dsn(), min_size=1, max_size=5, init=_init,
                                              command_timeout=15)
        return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def ensure_schema() -> None:
    """Applies schema.sql idempotently. Docker only runs it on a *fresh* volume, so an
    existing database would otherwise silently lack newer tables (e.g. query_cache)."""
    pool = await get_pool()
    await pool.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    await pool.execute("DELETE FROM query_cache WHERE created_at < now() - interval '30 days'")


async def get_meta(key: str) -> str | None:
    pool = await get_pool()
    return await pool.fetchval("SELECT value FROM meta WHERE key = $1", key)


async def set_meta(key: str, value: str) -> None:
    pool = await get_pool()
    await pool.execute(
        "INSERT INTO meta (key, value) VALUES ($1, $2) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", key, value)


async def check_embed_model() -> None:
    """Fail loudly if the stored vectors came from a different embedding model than the
    one now configured. Cohere and bge-m3 are both 1024-d, so nothing else would notice
    and retrieval would silently return garbage after a profile switch."""
    stored = await get_meta("embed_model")
    wanted = get_settings().embed_model_id
    if stored and stored != wanted:
        raise ConfigError(
            f"Database vectors were built with '{stored}' but the active profile uses "
            f"'{wanted}'. Re-index with: python scripts/index_corpus.py --force"
        )
