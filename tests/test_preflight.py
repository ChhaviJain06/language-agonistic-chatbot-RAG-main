"""scripts/preflight.py: air-gap preconditions and service reachability."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scripts.preflight import check_port, run_preflight
from server.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_check_port_success_and_failure():
    with patch("socket.socket") as mock_sock:
        inst = mock_sock.return_value.__enter__.return_value
        inst.connect_ex.return_value = 0
        assert check_port("PostgreSQL", "localhost", 5432) is True
        inst.connect_ex.return_value = 111
        assert check_port("PostgreSQL", "localhost", 5432) is False


def test_onprem_with_cloud_key_is_fatal(monkeypatch):
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_leaked_key")
    with pytest.raises(SystemExit) as exc:
        run_preflight()
    assert exc.value.code == 1


def test_onprem_with_missing_models_is_fatal(monkeypatch, tmp_path):
    monkeypatch.setenv("BHASHA_PROFILE", "onprem")
    for k in ("GROQ_API_KEY", "SARVAM_API_KEY", "COHERE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)                       # no data/models here
    with patch("scripts.preflight._hf_cached", return_value=False), \
            pytest.raises(SystemExit) as exc:
        run_preflight()
    assert exc.value.code == 1


def test_cloud_profile_passes_when_services_are_up(monkeypatch):
    monkeypatch.setenv("BHASHA_PROFILE", "cloud")
    with patch("scripts.preflight.check_port", return_value=True):
        run_preflight()


def test_offline_service_is_fatal(monkeypatch):
    monkeypatch.setenv("BHASHA_PROFILE", "cloud")
    with patch("scripts.preflight.check_port", return_value=False), pytest.raises(SystemExit):
        run_preflight()
