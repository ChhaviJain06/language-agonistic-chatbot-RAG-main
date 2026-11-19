"""scripts/eval_ragas.py — prints the viva summary FROM a real results/final_eval.json.

It never invents a number: a missing metric prints "n/a" (C7). Generate the file first
with `python scripts/evaluate.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def compile_viva_report(eval_path: str = "results/final_eval.json") -> None:
    from scripts.evaluate import format_markdown_report

    path = Path(eval_path)
    if not path.exists():
        print(f"Error: {eval_path} not found. Run `python scripts/evaluate.py` first.")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + "=" * 70)
    print("🕉  BHASHA — EVALUATION REPORT (measured values only)")
    print("=" * 70)
    print(format_markdown_report(data))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    compile_viva_report(*sys.argv[1:2])
