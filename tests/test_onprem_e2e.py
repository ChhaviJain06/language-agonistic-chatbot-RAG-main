"""BHASHA_PROFILE=onprem end-to-end with a network guard: the real answer pipeline
(language contract -> Ollama provider -> citations -> visual stage) runs against a mocked
local Ollama, and ANY request to a cloud domain fails the test."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from server.config import ConfigError, Settings, get_settings
from server.providers.factory import get_asr_en, get_embedder, get_llm, get_tts, reset_providers
from server.rag.orchestrate import OrchestrationResult
from server.rag.retrieve import Hit

BANNED = ["groq.com", "sarvam.ai", "cohere.ai", "cohere.com", "openai.com", "anthropic.com",
          "huggingface.co", "openstreetmap.org"]
CLOUD_KEYS = ["GROQ_API_KEY", "SARVAM_API_KEY", "COHERE_API_KEY", "OPENAI_API_KEY"]


@pytest.fixture(autouse=True)
def onprem(monkeypatch):
    for k in CLOUD_KEYS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    monkeypatch.setenv("CACHE_ENABLED", "0")
    get_settings.cache_clear()
    reset_providers()
    yield
    get_settings.cache_clear()
    reset_providers()


@pytest.mark.parametrize("key", CLOUD_KEYS)
def test_fails_loud_on_leaked_key(key, monkeypatch):
    monkeypatch.setenv(key, "sk-leaked")
    get_settings.cache_clear()
    with pytest.raises(ConfigError, match=key):
        Settings(_env_file=None)


def test_every_provider_resolves_local():
    assert get_asr_en().name == "faster-whisper" and get_embedder().name == "bge-m3"
    assert get_llm().name == "ollama" and get_tts().name == "kokoro"
    assert get_settings().resolved["graph"] == "neo4j_local"


async def test_full_answer_turn_makes_no_cloud_request(monkeypatch):
    from server.rag.answer import answer_question

    async def guarded(self, request, *a, **k):
        url = str(request.url)
        assert not any(d in url for d in BANNED), f"cloud egress to {url}"
        assert "11434" in url, f"unexpected request {url}"
        tokens = ["Ashoka ", "ne ", "Kalinga ", "war ", "ke ", "baad ", "Buddhism ", "apnaya ",
                  "[ashoka::0000]."]
        body = b"".join(json.dumps({"message": {"content": t}, "done": False}).encode() + b"\n"
                        for t in tokens) + b'{"done": true}\n'
        return httpx.Response(200, stream=httpx.ByteStream(body), request=request)

    hit = Hit("ashoka::0000", "Ashoka", "Introduction",
              "Ashoka embraced Buddhism after the Kalinga war.", 0.95,
              "https://en.wikipedia.org/wiki/Ashoka")

    async def fake_orchestrate(q, question_type="factoid", **k):
        return OrchestrationResult([hit], [], [hit])

    monkeypatch.setattr(httpx.AsyncClient, "send", guarded)
    with patch("server.rag.answer.orchestrate", fake_orchestrate):
        res = await answer_question("Ashoka ne Kalinga ke baad kya kiya?")

    assert (res.contract.lang, res.contract.script) == ("hi-en", "Latn")
    assert res.fidelity_passed and res.text.endswith("[1].")
    assert res.sources[0]["chunk_id"] == "ashoka::0000"
    assert res.visual_stage["matched"] and res.visual_stage["citation"]["article"] == "Ashoka"
