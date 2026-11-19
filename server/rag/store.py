"""Persists conversations and turns (C10: session state lives in Postgres).

Best-effort by design: a database hiccup is logged and must never break the live
conversation, so every function swallows errors — but logs them, never silently.
"""

from __future__ import annotations

import json
import logging

from server.rag.db import get_pool

logger = logging.getLogger(__name__)


async def start_conversation(room: str) -> str | None:
    try:
        pool = await get_pool()
        return str(await pool.fetchval(
            "INSERT INTO conversations (room) VALUES ($1) RETURNING id", room))
    except Exception as exc:  # noqa: BLE001 - persistence must not break the call
        logger.warning("could not start conversation row: %s", exc)
        return None


async def end_conversation(conversation_id: str | None) -> None:
    if not conversation_id:
        return
    try:
        pool = await get_pool()
        await pool.execute("UPDATE conversations SET ended_at = now() WHERE id = $1::uuid",
                           conversation_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not close conversation row: %s", exc)


async def log_exchange(conversation_id: str | None, *, question: str, answer: str,
                       spoken_text: str | None, lang: str, script: str,
                       cited_chunk_ids: list[str], timing: dict) -> None:
    """Writes the user turn and the agent turn together."""
    if not conversation_id:
        return
    try:
        pool = await get_pool()
        async with pool.acquire() as conn, conn.transaction():
            idx = await conn.fetchval(
                "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns "
                "WHERE conversation_id = $1::uuid", conversation_id)
            await conn.execute(
                "INSERT INTO turns (conversation_id, turn_index, role, transcript, "
                "detected_lang, detected_script) VALUES ($1::uuid, $2, 'user', $3, $4, $5)",
                conversation_id, idx, question, lang, script)
            await conn.execute(
                "INSERT INTO turns (conversation_id, turn_index, role, transcript, spoken_text, "
                "detected_lang, detected_script, cited_chunk_ids, latency_ms) "
                "VALUES ($1::uuid, $2, 'agent', $3, $4, $5, $6, $7, $8::jsonb)",
                conversation_id, idx + 1, answer, spoken_text, lang, script,
                cited_chunk_ids, json.dumps(timing))
            await conn.execute(
                "UPDATE conversations SET lang = $2, script = $3 WHERE id = $1::uuid",
                conversation_id, lang, script)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not log turn: %s", exc)
