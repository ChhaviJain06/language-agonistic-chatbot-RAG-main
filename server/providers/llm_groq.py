"""Groq-hosted LLM (free tier is enough for development). Groq's chat endpoint is
plain OpenAI-compatible REST, and httpx is already a dependency, so a second SDK
for one HTTP call buys nothing.

`stream()` is the conversational path (contract + RAG context -> cited answer)
and must be genuinely cancellable: it's a plain async generator over SSE lines,
so `asyncio.CancelledError` raised while awaiting the next chunk closes the httpx
stream via the `async with` block, same as any other cancelled task.

`complete()` is a plain non-streaming call for one-shot work (graph extraction,
entity linking, judging) that has nothing to do with reply language or citations.

A single keep-alive httpx client is reused across turns, so each turn skips the
TLS handshake (worth ~100-200 ms of time-to-first-audio).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers._http import make_client
from server.providers._prompt import build_system_prompt
from server.providers.base import LangContract, LLMProvider

logger = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_ATTEMPTS = 6
MAX_WAIT_SECONDS = 60          # batch jobs (graph build, eval) may wait this long for a rate limit
LIVE_MAX_WAIT_SECONDS = 6      # a live conversation must fail fast instead of sleeping a minute
_DURATION_RE = re.compile(r"(?:(\d+)m)?([\d.]+)s")
_TRY_AGAIN_RE = re.compile(r"try again in ([\d.]+m?[\d.]*s)", re.IGNORECASE)


def _parse_duration(value: str) -> float:
    m = _DURATION_RE.match(value.strip())
    if not m:
        return 5.0
    minutes, seconds = m.groups()
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


def _wait_seconds(resp: httpx.Response, body_text: str) -> float:
    """The x-ratelimit-reset-tokens header only reflects the per-minute window. A 429
    can also come from a tokens-PER-DAY cap, where that header is irrelevant — the
    error message's own "try again in Xs" is the authoritative wait, so prefer it."""
    m = _TRY_AGAIN_RE.search(body_text)
    if m:
        return _parse_duration(m.group(1))
    return _parse_duration(resp.headers.get("x-ratelimit-reset-tokens", "5s"))


class GroqLLM(LLMProvider):
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        s = get_settings()
        self._api_key = api_key or s.groq_api_key
        self._model = model or s.llm_model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = make_client()
        return self._client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _system_prompt(self, contract: LangContract, context: list[dict]) -> str:
        return build_system_prompt(contract, context)

    async def stream(self, messages: list[dict], contract: LangContract,
                     context: list[dict]) -> AsyncIterator[str]:
        s = get_settings()
        models = [self._model]
        if s.llm_fallback_model and s.llm_fallback_model != self._model:
            models.append(s.llm_fallback_model)
        client = self._get_client()
        limited: httpx.Response | None = None

        for model in models:
            payload = {
                "model": model,
                "messages": [{"role": "system",
                              "content": self._system_prompt(contract, context)}, *messages],
                "stream": True,
            }
            if s.llm_reasoning_effort != "default" and model.startswith("openai/gpt-oss"):
                payload["reasoning_effort"] = s.llm_reasoning_effort   # reasoning models only
            for attempt in range(1, MAX_ATTEMPTS + 1):
                async with client.stream("POST", GROQ_URL, json=payload,
                                         headers=self._headers()) as resp:
                    if resp.status_code == 429:
                        body_text = (await resp.aread()).decode("utf-8", "replace")
                        wait = _wait_seconds(resp, body_text)
                        limited = resp
                        if attempt == MAX_ATTEMPTS or wait > LIVE_MAX_WAIT_SECONDS:
                            logger.warning("groq %s rate-limited (wait %.1fs)%s", model, wait,
                                           " — trying fallback model" if model != models[-1]
                                           else " — giving up fast")
                            break                  # next model, or raise below
                        await asyncio.sleep(wait + 0.3)
                        continue                   # nothing yielded yet — safe to retry
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):]
                        if data == "[DONE]":
                            break
                        choices = json.loads(data).get("choices") or [{}]
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    return
        assert limited is not None
        raise httpx.HTTPStatusError("429 rate limited on every configured model",
                                    request=limited.request, response=limited)

    async def complete(self, messages: list[dict], *, json_mode: bool = False,
                       max_tokens: int | None = None,
                       max_wait: float = MAX_WAIT_SECONDS) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            # Reasoning models (gpt-oss-*) spend completion tokens on hidden reasoning
            # first; without headroom a call can return "" for the actual answer.
            payload["max_completion_tokens"] = max_tokens
        client = self._get_client()
        for attempt in range(1, MAX_ATTEMPTS + 1):
            resp = await client.post(GROQ_URL, json=payload, headers=self._headers(),
                                     timeout=60.0)
            if resp.status_code != 429:
                if resp.status_code >= 400:
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code} calling Groq: {resp.text[:500]}",
                        request=resp.request, response=resp,
                    )
                return resp.json()["choices"][0]["message"]["content"]
            wait = _wait_seconds(resp, resp.text)
            if attempt == MAX_ATTEMPTS or wait > max_wait:
                resp.raise_for_status()
            await asyncio.sleep(wait + 0.5)
        raise RuntimeError("unreachable")

    async def warm(self) -> None:
        await self.health()

    async def health(self) -> bool:
        """Cheap check: model-list endpoint, no tokens spent."""
        try:
            resp = await self._get_client().get(
                "https://api.groq.com/openai/v1/models", headers=self._headers(), timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
