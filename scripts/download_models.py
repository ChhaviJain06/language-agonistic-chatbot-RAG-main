#!/usr/bin/env python3
"""download_models.py — fetch every open-source model BHASHA needs, ONCE, while online.

After this, `BHASHA_PROFILE=onprem` runs with no network (HF_HUB_OFFLINE=1 is enforced by
server/config.py). Everything downloaded is free and open-source:

  * fastText lid.176 (language ID)             ~1 MB  (compressed .ftz)
  * BAAI/bge-m3 (multilingual embeddings)      ~2.3 GB
  * BAAI/bge-reranker-v2-m3 (reranker)         ~2.2 GB
  * faster-whisper model (ASR)                 size via WHISPER_MODEL_SIZE (default large-v3-turbo)
  * Kokoro-82M (TTS)                           ~330 MB
  * IndicTrans2 en<->indic 200M x2 (translation) ~1.6 GB (optional, needs ".[translate]")

Usage:
    python scripts/download_models.py             # everything
    python scripts/download_models.py --only lid  # one of: lid embed rerank asr tts translate
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID_PATH = Path("data/models/lid.176.ftz")


def get_lid() -> None:
    LID_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LID_PATH.exists():
        print(f"[skip] {LID_PATH} already present")
        return
    print(f"[get ] fastText lid.176 -> {LID_PATH}")
    urllib.request.urlretrieve(LID_URL, LID_PATH)  # noqa: S310 - fixed, trusted URL


def get_hf(repo: str) -> None:
    from huggingface_hub import snapshot_download

    print(f"[get ] {repo}")
    snapshot_download(repo)


def get_asr() -> None:
    from faster_whisper import WhisperModel

    size = os.getenv("WHISPER_MODEL_SIZE", "large-v3-turbo")
    print(f"[get ] faster-whisper {size}")
    WhisperModel(size, device="cpu", compute_type="int8")


def get_tts() -> None:
    from kokoro import KPipeline

    print("[get ] Kokoro-82M (English + Hindi pipelines)")
    for code in ("a", "h"):
        KPipeline(lang_code=code)


def get_translate() -> None:
    from server.providers.translate_local import EN_INDIC, INDIC_EN

    for repo in (EN_INDIC, INDIC_EN):
        get_hf(repo)


TASKS = {
    "translate": get_translate,
    "lid": get_lid,
    "embed": lambda: get_hf("BAAI/bge-m3"),
    "rerank": lambda: get_hf("BAAI/bge-reranker-v2-m3"),
    "asr": get_asr,
    "tts": get_tts,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(TASKS), action="append")
    args = ap.parse_args()
    failed = []
    for name in args.only or list(TASKS):
        try:
            TASKS[name]()
        except Exception as exc:  # noqa: BLE001 - report all failures, then exit non-zero
            print(f"[fail] {name}: {exc!r}", file=sys.stderr)
            failed.append(name)
    print("\n[done]" if not failed else f"\n[done with failures: {failed}]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
