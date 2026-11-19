"""Local Ollama LLM (free, open-source; onprem/hybrid fallback, C5).

Implemented against Ollama's documented /api/chat NDJSON streaming contract and
matching llm_groq.py's structure. Not verified against a live Ollama instance on the
primary dev machine (no GPU) — verify on capable hardware before relying on it.
The model comes from OLLAMA_MODEL (default qwen2.5:7b-instruct); `ollama pull` it first.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from server.config import get_settings
from server.providers._http import make_client
from server.providers._prompt import build_system_prompt
from server.providers.base import LangContract, LLMProvider


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str | None = None, model: str | None = None):
        s = get_settings()
        self._base_url = (base_url or s.ollama_base_url).rstrip("/")
        self._model = model or s.ollama_model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = make_client(timeout=120.0, connect=3.0)
        return self._client

    def _system_prompt(self, contract: LangContract, context: list[dict]) -> str:
        return build_system_prompt(contract, context)

    async def stream(self, messages: list[dict], contract: LangContract,
                     context: list[dict]) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "system", "content": self._system_prompt(contract, context)},
                        *messages],
            "stream": True,
        }
        url = f"{self._base_url}/api/chat"
        async with self._get_client().stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                delta = chunk.get("message", {}).get("content")
                if delta:
                    yield delta
                if chunk.get("done"):
                    break

    async def complete(self, messages: list[dict], *, json_mode: bool = False,
                       max_tokens: int | None = None, max_wait: float = 0.0) -> str:
        payload = {"model": self._model, "messages": messages, "stream": False}
        if json_mode:
            payload["format"] = "json"
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        resp = await self._get_client().post(f"{self._base_url}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    async def health(self) -> bool:
        try:
            resp = await self._get_client().get(f"{self._base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
