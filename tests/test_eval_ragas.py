"""The viva report must be built from a real results file and must never invent numbers."""

import json
from pathlib import Path

from scripts.eval_ragas import compile_viva_report
from tests.test_evaluation import _report


def test_report_comes_from_the_results_file(tmp_path: Path, capsys):
    f = tmp_path / "final_eval.json"
    f.write_text(json.dumps(_report(spr=0.877)), encoding="utf-8")
    compile_viva_report(str(f))
    out = capsys.readouterr().out
    assert "87.7%" in out and "measured values only" in out
    assert "412" not in out and "98 ms" not in out          # the old hardcoded figures


def test_missing_file_says_so(capsys):
    compile_viva_report("nonexistent.json")
    assert "not found" in capsys.readouterr().out
