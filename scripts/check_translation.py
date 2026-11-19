#!/usr/bin/env python3
"""check_translation.py — prove the translation setup works on THIS machine.

Run after installing the optional local translator:
    pip install -e ".[translate]" && python scripts/download_models.py --only translate
    python scripts/check_translation.py
Prints which engine answered, the result and the time for each sample. Exits non-zero if
the configured provider could not translate anything.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLES = [
    ("अशोक ने कलिंग युद्ध के बाद क्या किया?", "en", "hi → en"),
    ("மௌரிய பேரரசை நிறுவியவர் யார்?", "en", "ta → en"),
    ("শিবাজী মহারাজ কে ছিলেন?", "en", "bn → en"),
    ("शिवाजी महाराजांचा राज्याभिषेक कधी झाला?", "en", "mr → en"),
    ("Ashoka embraced Buddhism after the Kalinga war.", "hi", "en → hi"),
    ("Ashoka embraced Buddhism after the Kalinga war.", "ta", "en → ta"),
]


async def main() -> int:
    from server.config import get_settings
    from server.providers.factory import get_translator

    t = get_translator()
    print(f"configured: {get_settings().translation_provider}  ->  {t.name}\n")
    ok = 0
    for text, target, label in SAMPLES:
        t0 = time.time()
        out = await t.translate(text, target)
        ok += bool(out)
        print(f"[{label}] {out!r}  ({time.time() - t0:.1f}s)")
    print(f"\n{ok}/{len(SAMPLES)} translated")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
