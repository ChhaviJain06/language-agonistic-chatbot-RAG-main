"""BHASHA FastAPI app: health, LiveKit token, text chat (JSON + SSE) and the built UI."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from livekit import api
from pydantic import BaseModel, Field

from server.config import ConfigError, get_settings

# uvicorn only configures its own loggers; without this, every logger.info() in server.* (turn
# latency, warm-up, fidelity repairs) is silently dropped at the default WARNING level.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("bhasha")
VERSION = "0.2.0"

_active_agent_tasks: dict[str, asyncio.Task] = {}
_bg_tasks: set[asyncio.Task] = set()      # asyncio keeps only weak refs to running tasks


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)
_rate_hits: dict[str, deque[float]] = defaultdict(deque)


# --------------------------------------------------------------------------- guards
def rate_limit(request: Request) -> None:
    """Sliding-window limiter per client IP (in-memory: fine for a single-node deployment)."""
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    # Behind Caddy every request comes from the proxy; the real client is in X-Forwarded-For.
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    ip = fwd or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    hits = _rate_hits[ip]
    while hits and now - hits[0] > 60:
        hits.popleft()
    if len(hits) >= limit:
        raise HTTPException(429, "Too many requests — please slow down.")
    hits.append(now)
    for k in [k for k, v in _rate_hits.items() if not v or now - v[-1] > 60]:
        del _rate_hits[k]        # idle clients must not accumulate forever


def check_access_key(x_access_key: str | None = Header(default=None)) -> None:
    key = get_settings().access_key
    if key and not hmac.compare_digest(x_access_key or "", key):
        raise HTTPException(401, "Missing or invalid access key.")


def friendly_error(exc: BaseException) -> tuple[int, str]:
    """(HTTP status, user-facing message) — never leaks internals. A free-tier rate limit is
    the most common failure in development, so it gets its own clear message."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return 503, ("The AI service is busy (free-tier rate limit). "
                     "Please wait a few seconds and try again.")
    if isinstance(exc, httpx.HTTPError):
        return 502, "Could not reach an AI service. Please try again."
    return 500, "Something went wrong while answering. Please try again."


# ------------------------------------------------------------------------ lifespan
async def _run_agent_pipeline(room_name: str, token: str) -> None:
    try:
        from server.pipeline import run_pipeline

        await run_pipeline(room_name, token)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("pipeline task in room %s failed", room_name)
    finally:
        _active_agent_tasks.pop(room_name, None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validates the profile and air-gap guarantees immediately (C1, C5).
    s = get_settings()
    try:
        from server.rag.db import check_embed_model, ensure_schema

        await ensure_schema()
        await check_embed_model()
    except ConfigError:
        raise                        # wrong embedding model: fail loudly, never serve garbage
    except Exception as exc:  # noqa: BLE001 - the app can still serve /health without a DB
        logger.warning("database not ready (%s) — chat endpoints will fail until it is", exc)
    logger.info("BHASHA %s started, profile=%s providers=%s", VERSION, s.bhasha_profile,
                s.resolved)
    if not s.disable_agent:      # tests and API-only mode skip it
        from server.warmup import warm_all

        _spawn(warm_all())
    yield
    for task in list(_active_agent_tasks.values()):
        task.cancel()
    from server.providers import factory
    from server.rag.db import close_pool

    tts = factory._tts
    if tts is not None and hasattr(tts, "close"):
        await tts.close()
    graph = factory._graph
    if graph is not None and hasattr(graph, "close"):
        await graph.close()
    await close_pool()


app = FastAPI(title="BHASHA", version=VERSION, lifespan=lifespan)
if get_settings().cors_origins_list:
    app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins_list,
                       allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {"status": "ok", "profile": s.bhasha_profile, "providers": s.resolved,
            "version": VERSION}


# ---------------------------------------------------------------------- voice token
class TokenResponse(BaseModel):
    token: str
    url: str
    room: str


@app.post("/livekit/token", dependencies=[Depends(check_access_key), Depends(rate_limit)])
async def livekit_token(request: Request) -> TokenResponse:
    """Mints a join token. The SERVER always picks the room name: a client-chosen room
    would let anyone join (and listen to) someone else's conversation."""
    s = get_settings()
    if not s.livekit_api_key or not s.livekit_api_secret:
        raise HTTPException(503, "LiveKit is not configured (LIVEKIT_API_KEY / SECRET).")

    if not s.disable_agent:
        live = [t for t in _active_agent_tasks.values() if not t.done()]
        if len(live) >= s.max_concurrent_agents:
            raise HTTPException(503, "The assistant is busy — please try again shortly.")

    room_name = f"bhasha-{uuid.uuid4().hex[:16]}"
    identity = f"user-{uuid.uuid4().hex[:8]}"
    token = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
             .with_identity(identity).with_name(identity)
             .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())

    forwarded_proto = request.headers.get("x-forwarded-proto")
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    lk_url = f"wss://{forwarded_host}" if forwarded_proto == "https" and forwarded_host \
        else (s.livekit_public_url or s.livekit_url)

    if not s.disable_agent:
        agent_token = (api.AccessToken(s.livekit_api_key, s.livekit_api_secret)
                       .with_identity(f"agent-{uuid.uuid4().hex[:8]}").with_name("BHASHA Agent")
                       .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt())
        _active_agent_tasks[room_name] = asyncio.create_task(
            _run_agent_pipeline(room_name, agent_token))
    return TokenResponse(token=token, url=lk_url, room=room_name)


@app.get("/voice/turns", dependencies=[Depends(check_access_key), Depends(rate_limit)])
async def voice_turns(room: str = Query(min_length=8, max_length=64,
                                        pattern=r"^bhasha-[0-9a-f]+$")) -> dict:
    """Answers of a voice room that may not have reached the browser (dropped data channel).
    The room name is the unguessable id the server handed out with the token."""
    from server import voice_replay

    return {"turns": voice_replay.turns_for(room)}


# ------------------------------------------------------------------------ text chat
class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    question_type: str = "factoid"
    lang: str | None = Field(default=None, max_length=8)
    history: list[Msg] = Field(default_factory=list, max_length=20)


def _prepare(req: QueryRequest) -> tuple[str, list[dict]]:
    question = req.question.strip()
    limit = get_settings().max_question_chars
    if not question:
        raise HTTPException(422, "Question is empty.")
    if len(question) > limit:
        raise HTTPException(413, f"Question is too long (max {limit} characters).")
    history = [m.model_dump() for m in req.history][-6:]
    return question, history


@app.post("/query", dependencies=[Depends(check_access_key), Depends(rate_limit)])
async def text_query(req: QueryRequest) -> dict:
    """Non-streaming text query (fallback when SSE is unavailable)."""
    from server.rag.answer import answer_question, maybe_cache

    question, history = _prepare(req)
    try:
        res = await answer_question(question, history=history, forced_lang=req.lang,
                                    question_type=req.question_type)
    except Exception as exc:
        logger.exception("query failed")
        status, message = friendly_error(exc)
        raise HTTPException(status, message, headers={"Retry-After": "10"}) from exc
    _spawn(maybe_cache(res, history))
    return res.to_payload()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/query/stream", dependencies=[Depends(check_access_key), Depends(rate_limit)])
async def stream_query(req: QueryRequest) -> StreamingResponse:
    """Server-Sent Events. Sentences are released only after passing the language
    fidelity contract, so the client never displays wrong-script text."""
    from server.lang.contract import build_contract
    from server.rag.answer import answer_question, maybe_cache

    question, history = _prepare(req)
    msg_id = uuid.uuid4().hex[:10]

    async def event_generator():
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def on_sentence(sent: str) -> None:
            await queue.put({"event": "sentence", "text": sent})

        async def work() -> None:
            try:
                res = await answer_question(question, history=history, contract=contract,
                                            on_sentence=on_sentence)
                payload = res.to_payload()
                payload.pop("question", None)
                await queue.put({"event": "done", **payload})
                await maybe_cache(res, history)
            except Exception as exc:
                logger.exception("stream query failed")
                await queue.put({"event": "error", "error": friendly_error(exc)[1]})
            finally:
                await queue.put(None)

        contract = await build_contract(question, forced_lang=req.lang)
        yield _sse({"event": "start", "lang": contract.lang, "script": contract.script,
                    "msg_id": msg_id})
        task = asyncio.create_task(work())
        try:
            while (item := await queue.get()) is not None:
                yield _sse(item)
        finally:
            task.cancel()          # client closed the stream: stop generating

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Mount the built React frontend if it exists
_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")
