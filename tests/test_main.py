"""API behaviour: health, token minting and its hardening, text chat (JSON + SSE)."""

import json

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.config import Settings
from server.lang.contract import LangContract
from server.rag.answer import AnswerResult

client = TestClient(main.app)


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, livekit_api_key="devkey",
                    livekit_api_secret="devsecret_change_me_at_least_32_chars",
                    disable_agent=True, rate_limit_per_minute=0, **kw)


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    main._rate_hits.clear()
    monkeypatch.setattr(main, "get_settings", lambda: _settings())


def _result(text="Ashoka was a Mauryan emperor [1].", **kw) -> AnswerResult:
    return AnswerResult(
        question="Who was Ashoka?", contract=LangContract("en", "Latn"),
        raw_text="Ashoka was a Mauryan emperor [ashoka::0001].", text=text,
        sources=[{"n": 1, "chunk_id": "ashoka::0001", "article": "Ashoka", "section": "Reign",
                  "url": "https://en.wikipedia.org/wiki/Ashoka", "text": "Ashoka was an emperor.",
                  "score": 0.9}],
        fidelity_passed=True, repairs=0, cached=False,
        visual_stage={"event": "HISTORICAL_CONTEXT", "matched": True},
        timing={"retrieval_ms": 10.0, "first_sentence_ms": 50.0, "total_ms": 90.0}, **kw)


def test_health_reports_resolved_providers():
    body = client.get("/health").json()
    assert body["status"] == "ok" and "llm" in body["providers"] and body["version"]


def test_token_mints_a_jwt_and_server_chooses_the_room():
    resp = client.post("/livekit/token", params={"room": "someone-elses-room"})
    body = resp.json()
    assert resp.status_code == 200 and body["token"].count(".") == 2
    assert body["room"].startswith("bhasha-") and body["room"] != "someone-elses-room"


def test_token_requires_access_key_when_configured(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: _settings(access_key="s3cret"))
    assert client.post("/livekit/token").status_code == 401
    assert client.post("/livekit/token", headers={"X-Access-Key": "wrong"}).status_code == 401
    assert client.post("/livekit/token", headers={"X-Access-Key": "s3cret"}).status_code == 200


def test_rate_limit_returns_429(monkeypatch):
    s = Settings(_env_file=None, livekit_api_key="devkey", disable_agent=True,
                 livekit_api_secret="devsecret_change_me_at_least_32_chars",
                 rate_limit_per_minute=2)
    monkeypatch.setattr(main, "get_settings", lambda: s)
    codes = [client.post("/livekit/token").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_question_length_limit(monkeypatch):
    monkeypatch.setattr(main, "get_settings", lambda: _settings(max_question_chars=20))
    assert client.post("/query", json={"question": "x" * 21}).status_code == 413
    assert client.post("/query", json={"question": "   "}).status_code in (413, 422)


def test_query_returns_sources_and_timing(monkeypatch):
    async def fake(question, **kw):
        assert kw["history"] == [{"role": "user", "content": "hi"}]
        return _result()

    async def noop(*a, **k):
        return None

    monkeypatch.setattr("server.rag.answer.answer_question", fake)
    monkeypatch.setattr("server.rag.answer.maybe_cache", noop)
    data = client.post("/query", json={"question": "Who was Ashoka?",
                                       "history": [{"role": "user", "content": "hi"}]}).json()
    assert data["answer"].endswith("[1].")
    assert data["sources"][0]["chunk_id"] == "ashoka::0001"
    assert data["timing"]["first_sentence_ms"] == 50.0 and "visual_stage" in data


def _events(text: str) -> list[dict]:
    return [json.loads(c[6:]) for c in text.split("\n\n") if c.startswith("data: ")]


def test_stream_sends_start_sentences_then_done(monkeypatch):
    async def fake_contract(q, forced_lang=None, **kw):
        return LangContract(forced_lang or "en", "Latn")

    async def fake(question, *, on_sentence=None, **kw):
        await on_sentence("Ashoka ruled well.")
        await on_sentence("He built stupas.")
        return _result()

    async def noop(*a, **k):
        return None

    monkeypatch.setattr("server.lang.contract.build_contract", fake_contract)
    monkeypatch.setattr("server.rag.answer.answer_question", fake)
    monkeypatch.setattr("server.rag.answer.maybe_cache", noop)
    resp = client.post("/query/stream", json={"question": "Who was Ashoka?"})
    assert "text/event-stream" in resp.headers["content-type"]
    ev = _events(resp.text)
    assert [e["event"] for e in ev] == ["start", "sentence", "sentence", "done"]
    assert ev[0]["msg_id"] and ev[-1]["sources"][0]["n"] == 1 and "question" not in ev[-1]


def test_stream_error_is_reported_without_leaking_internals(monkeypatch):
    async def fake_contract(q, forced_lang=None, **kw):
        return LangContract("en", "Latn")

    async def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("server.lang.contract.build_contract", fake_contract)
    monkeypatch.setattr("server.rag.answer.answer_question", boom)
    ev = _events(client.post("/query/stream", json={"question": "hi"}).text)
    assert ev[-1]["event"] == "error" and "secret" not in ev[-1]["error"]
    assert not any(e["event"] == "done" for e in ev)


def test_rate_limited_llm_gives_a_friendly_503_not_a_traceback(monkeypatch):
    import httpx

    async def limited(*a, **k):
        req = httpx.Request("POST", "https://api.groq.com/x")
        raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))

    monkeypatch.setattr("server.rag.answer.answer_question", limited)
    resp = client.post("/query", json={"question": "Who was Ashoka?"})
    assert resp.status_code == 503 and "busy" in resp.json()["detail"]
    assert resp.headers["retry-after"] == "10"
