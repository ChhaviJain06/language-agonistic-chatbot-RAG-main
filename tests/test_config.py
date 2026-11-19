"""Profile resolution is the one thing in Milestone 0 that can silently leak data."""

import pytest

from server.config import ConfigError, Settings


def make(**kw) -> Settings:
    # _env_file=None so a developer's real .env can never change a test outcome.
    return Settings(_env_file=None, **kw)


def test_hybrid_keeps_declared_providers():
    s = make(bhasha_profile="hybrid", llm_provider="groq", tts_provider="sarvam")
    assert s.resolved["llm"] == "groq"
    assert s.resolved["tts"] == "sarvam"


def test_onprem_overrides_every_cloud_provider():
    s = make(
        bhasha_profile="onprem",
        asr_en_provider="groq",
        asr_indic_provider="sarvam",
        llm_provider="groq",
        tts_provider="sarvam",
    )
    assert s.resolved == {
        "asr_en": "local",
        "asr_indic": "local",
        "llm": "ollama",
        "tts": "kokoro",
        "embed": "local",
        "rerank": "local",
        "graph": "neo4j_local",
        "translation": "local",
    }


@pytest.mark.parametrize(
    "key", ["groq_api_key", "sarvam_api_key", "cohere_api_key", "openai_api_key"]
)
def test_onprem_fails_loudly_on_any_cloud_key(key):
    with pytest.raises(ConfigError, match=key.upper()):
        make(bhasha_profile="onprem", **{key: "sk-not-a-real-key"})


def test_cloud_keys_are_fine_off_onprem():
    s = make(bhasha_profile="cloud", groq_api_key="sk-x")
    assert s.resolved["llm"] == "groq"


def test_default_profile_is_cloud():
    assert make().bhasha_profile == "cloud"


def test_onprem_rejects_service_urls_outside_the_private_network():
    with pytest.raises(ConfigError, match="OLLAMA_BASE_URL"):
        make(bhasha_profile="onprem", ollama_base_url="https://api.example.com")
    with pytest.raises(ConfigError, match="NEO4J_URI"):
        make(bhasha_profile="onprem", neo4j_uri="bolt://203.0.113.7:7687")


def test_onprem_accepts_docker_service_names_and_lan_hosts():
    make(bhasha_profile="onprem", ollama_base_url="http://ollama:11434",
         neo4j_uri="bolt://192.168.1.20:7687",
         database_url="postgresql+asyncpg://u:p@postgres:5432/db")


def test_onprem_blocks_huggingface_network_access(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    make(bhasha_profile="onprem")
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_embed_model_id_tracks_provider():
    assert make().embed_model_id.startswith("cohere")
    assert make(bhasha_profile="onprem").embed_model_id == "BAAI/bge-m3"
