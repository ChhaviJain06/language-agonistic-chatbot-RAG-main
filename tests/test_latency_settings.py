"""Latency-related behaviour: reasoning effort, reranking only the top candidates, warm-up."""

from contextlib import asynccontextmanager
from unittest.mock import patch

from server.config import Settings
from server.providers.base import LangContract
from server.providers.llm_groq import GroqLLM


class _SSE:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
        yield "data: [DONE]"

    def raise_for_status(self):
        pass


async def _payload_for(model: str, effort: str) -> dict:
    seen: dict = {}

    @asynccontextmanager
    async def fake_stream(self, method, url, json=None, **kw):
        seen.update(json)
        yield _SSE()

    settings = Settings(_env_file=None, llm_reasoning_effort=effort)
    with patch("httpx.AsyncClient.stream", fake_stream), \
            patch("server.providers.llm_groq.get_settings", lambda: settings):
        llm = GroqLLM(api_key="k", model=model)
        contract = LangContract("en", "Latn")
        async for _ in llm.stream([{"role": "user", "content": "x"}], contract, []):
            pass
    return seen


async def test_low_reasoning_effort_is_sent_to_gpt_oss():
    assert (await _payload_for("openai/gpt-oss-120b", "low"))["reasoning_effort"] == "low"


async def test_reasoning_effort_is_omitted_for_other_models_and_when_default():
    assert "reasoning_effort" not in await _payload_for("llama-3.3-70b-versatile", "low")
    assert "reasoning_effort" not in await _payload_for("openai/gpt-oss-120b", "default")


async def test_only_the_top_candidates_are_reranked(monkeypatch):
    from server.rag import retrieve as R

    rows = [{"chunk_id": f"c{i}", "article": "A", "section": "S", "text": f"t{i}", "url": ""}
            for i in range(20)]
    seen: dict = {}

    class Reranker:
        async def score(self, query, passages):
            seen["n"] = len(passages)
            return [float(len(passages) - i) for i in range(len(passages))]

    async def dense(*a, **k):
        return rows

    async def sparse(*a, **k):
        return []

    settings = Settings(_env_file=None, rerank_candidates=8)
    monkeypatch.setattr(R, "get_settings", lambda: settings)
    monkeypatch.setattr(R, "get_reranker", lambda: Reranker())
    monkeypatch.setattr(R, "_dense", dense)
    monkeypatch.setattr(R, "_sparse", sparse)
    hits = await R.retrieve("q", rerank_top_k=5)
    assert seen["n"] == 8 and len(hits) == 5


async def test_rerank_candidates_never_drops_below_the_requested_top_k(monkeypatch):
    from server.rag import retrieve as R

    rows = [{"chunk_id": f"c{i}", "article": "A", "section": "S", "text": "t", "url": ""}
            for i in range(20)]
    seen: dict = {}

    class Reranker:
        async def score(self, query, passages):
            seen["n"] = len(passages)
            return [1.0] * len(passages)

    async def dense(*a, **k):
        return rows

    async def sparse(*a, **k):
        return []

    settings = Settings(_env_file=None, rerank_candidates=2)
    monkeypatch.setattr(R, "get_settings", lambda: settings)
    monkeypatch.setattr(R, "get_reranker", lambda: Reranker())
    monkeypatch.setattr(R, "_dense", dense)
    monkeypatch.setattr(R, "_sparse", sparse)
    assert len(await R.retrieve("q", rerank_top_k=5)) == 5 and seen["n"] == 5


async def test_a_failing_warm_up_is_reported_not_raised(monkeypatch, caplog):
    import logging

    from server import warmup

    class Boom:
        async def warm(self):
            raise RuntimeError("network down")

    class Ok:
        async def warm(self):
            return None

    async def fake_pool():
        return None

    class FakeDetector:
        async def _ensure_loaded(self):
            return None

    monkeypatch.setattr(warmup.factory, "get_llm", lambda: Boom())
    for name in ("get_asr_en", "get_asr_indic", "get_tts", "get_embedder", "get_reranker"):
        monkeypatch.setattr(warmup.factory, name, lambda: Ok())
    monkeypatch.setattr(warmup, "get_pool", fake_pool)
    monkeypatch.setattr(warmup, "get_detector", lambda: FakeDetector())
    with caplog.at_level(logging.INFO):
        await warmup.warm_all()
    assert "llm" in caplog.text and "FAILED" in caplog.text


# ------------------------------------------------------------- rate-limit fallback model
class _Resp:
    def __init__(self, status: int, body: str = "", lines: list[str] | None = None):
        self.status_code, self._body, self._lines = status, body, lines or []
        self.headers: dict = {}
        self.request = None

    async def aread(self) -> bytes:
        return self._body.encode()

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_stream(responses: dict[str, _Resp], seen: list[str]):
    @asynccontextmanager
    async def fake(self, method, url, json=None, **kw):
        seen.append(json["model"])
        yield responses[json["model"]]
    return fake


_OK = _Resp(200, lines=['data: {"choices":[{"delta":{"content":"fallback answer"}}]}',
                        "data: [DONE]"])
_LIMITED = _Resp(429, "Rate limit reached ... Please try again in 40.5s")


async def _run(settings: Settings, responses: dict[str, _Resp]) -> tuple[str, list[str]]:
    seen: list[str] = []
    with patch("httpx.AsyncClient.stream", _fake_stream(responses, seen)), \
            patch("server.providers.llm_groq.get_settings", lambda: settings):
        llm = GroqLLM(api_key="k", model="openai/gpt-oss-120b")
        out = [t async for t in llm.stream([], LangContract("en", "Latn"), [])]
    return "".join(out), seen


async def test_a_rate_limited_model_falls_back_to_the_second_model_immediately():
    settings = Settings(_env_file=None, llm_fallback_model="openai/gpt-oss-20b")
    text, seen = await _run(settings, {"openai/gpt-oss-120b": _LIMITED,
                                       "openai/gpt-oss-20b": _OK})
    assert text == "fallback answer"
    assert seen == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]     # no 40 s sleep in between


async def test_when_every_model_is_limited_the_error_is_raised_fast():
    import httpx
    import pytest

    settings = Settings(_env_file=None, llm_fallback_model="openai/gpt-oss-20b")
    with pytest.raises(httpx.HTTPStatusError):
        await _run(settings, {"openai/gpt-oss-120b": _LIMITED, "openai/gpt-oss-20b": _LIMITED})


async def test_fallback_can_be_disabled():
    import httpx
    import pytest

    settings = Settings(_env_file=None, llm_fallback_model="")
    with pytest.raises(httpx.HTTPStatusError):
        await _run(settings, {"openai/gpt-oss-120b": _LIMITED})
