"""One place that builds HTTP clients for provider calls.

httpx drops idle connections after 5 s by default. A voice turn is separated from the next by
more than that, so every provider call (ASR, embedding, rerank, LLM, TTS) was paying a fresh
TLS handshake. Measured against Groq: 753 ms for a request after 12 s idle with the default,
327 ms with a 90 s keep-alive — about 0.4 s saved per provider call.
"""

from __future__ import annotations

import httpx

KEEPALIVE_SECONDS = 90.0


def make_client(timeout: float = 30.0, connect: float = 5.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=connect),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=KEEPALIVE_SECONDS),
    )
