"""Test environment: never touch a real .env, network, or LiveKit agent."""

import os

os.environ.setdefault("DISABLE_AGENT", "1")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "devsecret_change_me_at_least_32_chars")
os.environ.setdefault("CACHE_ENABLED", "0")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")
os.environ.setdefault("TRANSLATION_PROVIDER", "none")   # tests never hit Google

# Tests must never read a developer's real .env (it may hold live API keys).
from server.config import Settings  # noqa: E402

Settings.model_config["env_file"] = None
