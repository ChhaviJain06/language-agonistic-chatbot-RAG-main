"""The evaluation harness must never produce a number it did not measure."""

from __future__ import annotations

from pathlib import Path

from scripts.evaluate import (
    bootstrap_ci,
    calculate_percentiles,
    calculate_recall_at_k,
    calculate_spr,
    flavour_group,
    format_markdown_report,
    language_matches,
    load_partial_records,
    rate,
    save_partial_record,
    status,
    verify_citations,
)


def test_spr_per_flavour_and_regional_grouping():
    records = [
        {"flavour": "en", "expected_script": "Latn", "answer": "Ashoka renounced violence."},
        {"flavour": "hi-deva", "expected_script": "Deva", "answer": "चंद्रगुप्त मौर्य ने स्थापना की।"},
        {"flavour": "hi-deva", "expected_script": "Deva", "answer": "Wrong script."},
        {"flavour": "regional-mr", "expected_script": "Deva", "answer": "शिवाजी [a::1]"},
        {"flavour": "regional-bn", "expected_script": "Beng", "answer": "পলাশীর যুদ্ধ"},
    ]
    m = calculate_spr(records)
    assert m["spr_overall"] == 0.8 and m["per_flavour"]["hi-deva"] == 0.5
    assert m["per_flavour"]["regional"] == 1.0 and "regional-mr" not in m["per_flavour"]
    assert flavour_group("regional-ta") == "regional" and flavour_group("en") == "en"


def test_citation_verification():
    corpus, ctx = {"a::1", "a::2", "p::5"}, {"a::1", "p::5"}
    ok = verify_citations("x [a::1] y [p::5]", corpus, ctx)
    assert ok["valid_citation_rate"] == 1.0 and ok["grounded_in_context_rate"] == 1.0
    assert verify_citations("x [fake::9]", corpus, ctx)["valid_citation_rate"] == 0.0
    assert verify_citations("x [a::2]", corpus, ctx)["grounded_in_context_rate"] == 0.0
    assert verify_citations("none", corpus, ctx)["has_citations"] is False


def test_recall_and_percentiles():
    assert calculate_recall_at_k(["A", "B"], ["A", "x", "B"], 5) == 1.0
    assert calculate_recall_at_k(["A", "B"], ["A", "x"], 5) == 0.5
    assert calculate_recall_at_k([], ["A"], 5) == 1.0
    p = calculate_percentiles([100.0, 200.0, 300.0, 400.0, 1000.0])
    assert p["p50"] == 300.0 and p["count"] == 5
    assert calculate_percentiles([]) == {"p50": 0.0, "p90": 0.0, "mean": 0.0, "count": 0}


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    vals = [1.0] * 40 + [0.0] * 10
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.8 < hi and (lo, hi) == bootstrap_ci(vals)
    assert rate([])["value"] is None and rate([True, False])["n"] == 2


def test_language_match_requires_the_right_language_not_just_script():
    assert language_matches("mr", "…", "mr") and not language_matches("mr", "…", "hi")
    assert language_matches("hi-en", "Unhone Buddhism apnaya", "hi")
    assert not language_matches("hi-en", "अशोक", "hi")          # Devanagari is not Hinglish
    assert language_matches("en", "x", "en")


def test_status_reflects_targets_honestly():
    assert status("spr", 0.84) == "❌ Not met" and status("spr", 0.96) == "✅ Met"
    assert status("first_sentence_p50", 900.0) == "❌ Not met"       # lower is better
    assert status("lfr", None) == "n/a"


def test_checkpoint_roundtrip(tmp_path: Path):
    f = tmp_path / "p.jsonl"
    save_partial_record({"id": "G001", "answer": "a"}, f)
    save_partial_record({"id": "G002", "answer": "b"}, f)
    assert set(load_partial_records(f)) == {"G001", "G002"}


def _report(spr=0.84):
    return {
        "generated_at": "t", "answer_model": "m", "judge": "groq:x", "language_check": "fasttext",
        "n_scored": 50, "n_golden": 50, "errors": [], "skipped": ["G050"],
        "spr": {"spr_overall": spr, "per_flavour": {"en": 1.0}},
        "lfr": {"overall": {"value": 0.86, "ci95": [0.76, 0.94], "n": 49},
                "per_flavour": {"en": {"value": 1.0, "ci95": [1, 1], "n": 10}}},
        "citations": {"presence": {"value": 0.98, "ci95": [0.9, 1.0], "n": 49},
                      "validity": {"value": 1.0, "ci95": [1, 1], "n": 40},
                      "grounded": {"value": 0.8, "ci95": [0.7, 0.9], "n": 40}},
        "judge_metrics": {"false_premise_rejection": {"passed": 3, "n": 4},
                          "out_of_scope_refusal": {"passed": 2, "n": 2}},
        "retrieval": {"modes": {"dense": {"value": 0.77, "ci95": [0.6, 0.9], "n": 44},
                                "hybrid": {"value": 0.78, "ci95": [0.6, 0.9], "n": 44}},
                      "graph": {"n": 9, "lift": 0.0644,
                                "vector_only": {"value": 0.7, "ci95": [0.5, 0.9], "n": 9},
                                "vector_plus_graph": {"value": 0.76, "ci95": [0.5, 0.9], "n": 9}}},
        "latency_ms": {"first_sentence_ms": {"p50": 900.0, "p90": 1500.0},
                       "total_turn_ms": {"p50": 1200.0, "p90": 2000.0}},
    }


def test_report_marks_missed_targets_as_missed():
    md = format_markdown_report(_report())
    assert "### 📊 BHASHA System Evaluation Benchmark Report" in md
    for label in ("Script Preservation Rate (SPR)", "Recall@5", "Citation Grounding Rate"):
        assert label in md
    spr_row = next(line for line in md.splitlines() if "Script Preservation" in line)
    assert "❌ Not met" in spr_row and "✅" not in spr_row
    assert "95% CI" in md and "3 / 4" in md and "not measured" in md


def test_report_tolerates_missing_sections():
    md = format_markdown_report({"spr": {}, "lfr": {}, "citations": {}, "judge_metrics": {}})
    assert "n/a" in md
