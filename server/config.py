"""Settings + profile resolution.

The deployment story (C1) is "swap providers with an env var". That only works if
the swap is resolved exactly once, here, at startup — never with an `if provider ==
"groq"` sprinkled through business logic. Everything downstream reads
`settings.resolved` and nothing else.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["cloud", "hybrid", "onprem"]

# What each cloud-backed provider degrades to when the profile forbids the network.
LOCAL_EQUIVALENT = {
    "asr_en_provider": "local",
    "asr_indic_provider": "local",
    "llm_provider": "ollama",
    "tts_provider": "kokoro",
    "embed_provider": "local",
    "rerank_provider": "local",
    "graph_provider": "neo4j_local",
    "translation_provider": "local",   # MyMemory/Google send text off-machine (C5)
}

CLOUD_KEYS = ("groq_api_key", "sarvam_api_key", "cohere_api_key", "openai_api_key")

# Hosts that never leave the machine / private network.
_LOCAL_SUFFIXES = (".local", ".internal", ".lan", ".localhost")
_DOCKER_SERVICE_NAMES = {"postgres", "neo4j", "livekit", "ollama", "caddy", "app"}


class ConfigError(RuntimeError):
    """Raised at import/startup time. Deliberately fatal — see C5."""


def is_private_host(host: str) -> bool:
    """True for localhost, loopback, RFC1918 ranges and docker service names."""
    host = (host or "").lower()
    if host in {"localhost", "::1"} or host in _DOCKER_SERVICE_NAMES:
        return True
    if host.endswith(_LOCAL_SUFFIXES):
        return True
    parts = host.split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        return a == 127 or a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168)
    return False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Matches .env.example and the README: cloud is the safe default for development.
    bhasha_profile: Profile = "cloud"

    asr_en_provider: Literal["groq", "local"] = "groq"
    asr_indic_provider: Literal["sarvam", "local"] = "sarvam"
    llm_provider: Literal["groq", "ollama", "vllm"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    # gpt-oss models "think" before answering. Retrieval-grounded answers don't need much
    # thinking, and measured on Groq it cut time-to-first-token from ~0.76 s to ~0.4-0.6 s.
    # "default" leaves the model's own setting.
    # When the primary model is rate-limited (free tiers are, often), retry once on this model.
    # On Groq each model has its own quota bucket, so this turns an error into a fast answer.
    # Empty string disables it.
    llm_fallback_model: str = "openai/gpt-oss-20b"
    llm_reasoning_effort: Literal["low", "medium", "high", "default"] = "low"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    tts_provider: Literal["sarvam", "kokoro"] = "sarvam"
    # Streaming over WebSocket: first audio ~0.25 s instead of 0.3 s + 0.02 s per character.
    # False = the REST endpoint (also the automatic fallback if streaming fails).
    sarvam_tts_streaming: bool = True
    sarvam_speaker: str = "shubh"     # Bulbul v3 voice (documented default); lowercase
    embed_provider: Literal["cohere", "local"] = "cohere"
    rerank_provider: Literal["cohere", "local"] = "cohere"
    # Optional text translation (script-repair fallback, keyword search). Free, no key.
    translation_provider: Literal["auto", "local", "mymemory", "google", "none"] = "auto"
    translate_queries: bool = False     # also translate questions for keyword search (+~2 s/turn)

    groq_api_key: str = ""
    sarvam_api_key: str = ""
    cohere_api_key: str = ""
    openai_api_key: str = ""

    database_url: str = "postgresql+asyncpg://bhasha:bhasha@localhost:5432/bhasha"
    livekit_url: str = "ws://localhost:7880"
    livekit_public_url: str = ""      # what the BROWSER connects to; empty = livekit_url
    livekit_api_key: str = ""
    livekit_api_secret: str = ""

    vad_silence_ms: int = 600
    retrieve_top_k: int = 20
    rerank_top_k: int = 5
    rerank_timeout_s: float = 1.0
    skip_rerank_factoid: bool = False   # saves ~0.5 s on factoid questions; check recall first
    # Only the best N fused candidates are reranked: rerank time scales with the number of
    # passages (Cohere: ~0.66 s for 20, ~0.45 s for 10). Measured on the golden set
    # (scripts/eval_rerank_candidates.py, n=44): Recall@5 0.769 with 20, 0.754 with 12,
    # 0.750 with 10 — 12 saves ~0.15-0.2 s for ~1.5 points. The rest keep their RRF order.
    rerank_candidates: int = Field(default=12, ge=1)
    fidelity_script_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    graph_provider: Literal["neo4j_local", "neo4j_aura", "none"] = "none"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    graph_max_hops: int = 2
    graph_timeout_s: float = 2.0
    graph_trigger_types: str = "multi_hop,comparative"  # comma-separated question types (C12)

    # --- API hardening ------------------------------------------------------
    # Optional shared secret. When set, /livekit/token requires `X-Access-Key`.
    access_key: str = ""
    max_question_chars: int = 600
    rate_limit_per_minute: int = 30
    max_concurrent_agents: int = 5
    agent_join_timeout_s: int = 60
    voice_reconnect_grace_s: int = 20  # keep the agent alive this long after the browser drops
    cors_origins: str = ""            # comma-separated; empty = same-origin only
    disable_agent: bool = False       # tests / API-only mode: never start a voice agent

    # --- Semantic cache -----------------------------------------------------
    cache_enabled: bool = True
    cache_similarity: float = Field(default=0.96, ge=0.5, le=1.0)

    @property
    def graph_trigger_types_list(self) -> list[str]:
        return [t.strip() for t in self.graph_trigger_types.split(",") if t.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def embed_model_id(self) -> str:
        """Which embedding model produced the stored vectors. Stored in the DB so a
        profile switch (Cohere -> bge-m3) cannot silently corrupt retrieval: both are
        1024-d, so a dimension check alone would not catch it."""
        if self.embed_provider == "cohere":
            return "cohere/embed-multilingual-v3.0"
        return "BAAI/bge-m3"

    @model_validator(mode="after")
    def _resolve_profile(self) -> Settings:
        if self.bhasha_profile != "onprem":
            return self

        # Fail loud rather than quietly ignoring a key the operator believes is
        # in use: a leaked key in an "air-gapped" deployment is the worst outcome.
        leaked = [k.upper() for k in CLOUD_KEYS if getattr(self, k)]
        if leaked:
            raise ConfigError(
                f"BHASHA_PROFILE=onprem forbids cloud credentials, but these are set: "
                f"{', '.join(leaked)}. Unset them or switch profile."
            )

        # Every service URL must stay inside the private network too (C5).
        outside = []
        for name, url in (("OLLAMA_BASE_URL", self.ollama_base_url),
                          ("NEO4J_URI", self.neo4j_uri),
                          ("LIVEKIT_URL", self.livekit_url),
                          ("DATABASE_URL", self.database_url)):
            host = urlparse(url.replace("postgresql+asyncpg", "postgresql")).hostname or ""
            if not is_private_host(host):
                outside.append(f"{name}={host}")
        if outside:
            raise ConfigError(
                "BHASHA_PROFILE=onprem requires private-network hosts, but: "
                + ", ".join(outside)
            )

        for field, local in LOCAL_EQUIVALENT.items():
            object.__setattr__(self, field, local)

        # Block HuggingFace / transformers from phoning home; models must be
        # pre-downloaded with scripts/download_models.py.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        return self

    @property
    def resolved(self) -> dict[str, str]:
        """The provider actually in force for each pipeline stage."""
        return {
            "asr_en": self.asr_en_provider,
            "asr_indic": self.asr_indic_provider,
            "llm": self.llm_provider,
            "tts": self.tts_provider,
            "embed": self.embed_provider,
            "rerank": self.rerank_provider,
            "graph": self.graph_provider,
            "translation": self.translation_provider,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
