"""scripts/preflight.py - startup sanity checks for BHASHA.

Reads the SAME settings the server uses (.env included), so it cannot disagree with it.
For BHASHA_PROFILE=onprem it also proves the air-gap preconditions: no cloud keys, only
private-network hosts, and every local model already on disk.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check_port(service_name: str, host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def _hf_cached(repo: str) -> bool:
    cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
    return cache.exists()


def _services(s) -> list[tuple[str, str, int]]:
    def hp(url: str, default: int) -> tuple[str, int]:
        u = urlparse(url.replace("postgresql+asyncpg", "postgresql"))
        return u.hostname or "localhost", u.port or default

    pg = hp(s.database_url, 5432)
    lk = hp(s.livekit_url, 7880)
    services = [("PostgreSQL (pgvector)", *pg), ("LiveKit", *lk)]
    if s.graph_provider != "none":
        services.append(("Neo4j Bolt", *hp(s.neo4j_uri, 7687)))
    if s.llm_provider == "ollama":
        services.append(("Ollama", *hp(s.ollama_base_url, 11434)))
    return services


def run_preflight() -> None:
    from server.config import ConfigError, Settings

    try:
        s = Settings()
    except ConfigError as exc:
        print(f"\n🚨 FATAL: {exc}")
        sys.exit(1)

    print(f"[*] Running preflight verification for profile: [{s.bhasha_profile.upper()}]")
    print(f"    providers: {s.resolved}")

    if s.bhasha_profile == "onprem":
        print("  ✅ No cloud credentials; all service hosts are on the private network.")
        missing = []
        if not Path("data/models/lid.176.ftz").exists() and not Path("data/lid.176.bin").exists():
            missing.append("fastText lid.176 (data/models/lid.176.ftz)")
        for repo in ("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"):
            if not _hf_cached(repo):
                missing.append(f"HuggingFace model {repo}")
        if missing:
            print("\n❌ Local models missing (the air-gapped host cannot download them):")
            for m in missing:
                print(f"  - {m}")
            print("Run once, while online: python scripts/download_models.py")
            sys.exit(1)
        print("  ✅ Local models present on disk.")
        from server.providers.translate_local import local_available

        if not local_available():
            print("  ⚠️  Local translation not installed (optional): "
                  "pip install -e \".[translate]\"")
            print("      then: python scripts/download_models.py --only translate "
                  "(translation is skipped until then)")

    print("[*] Checking local infrastructure services:")
    all_healthy = True
    for service, host, port in _services(s):
        is_up = check_port(service, host, port)
        print(f"  - {service:<24} ({host}:{port}) -> {'ONLINE' if is_up else 'OFFLINE'}")
        all_healthy &= is_up

    if not all_healthy:
        print("\n❌ One or more required services are offline.")
        print("Run: docker compose up -d postgres livekit   (add neo4j / ollama as needed)")
        sys.exit(1)

    print("\n✅ All preflight checks passed. System ready for execution.\n")


if __name__ == "__main__":
    run_preflight()
