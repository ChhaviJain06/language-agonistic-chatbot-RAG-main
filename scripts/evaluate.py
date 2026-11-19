#!/usr/bin/env python3
"""evaluate.py — full evaluation harness. Every number in the report is MEASURED by this
run (C7). Nothing is hardcoded, and a failed target is shown as failed.

Needs a live database (Postgres+pgvector with the indexed corpus) and an LLM. There is
deliberately NO offline fallback: the old one fed the gold-answer articles to the model
as context, which measured the harness, not the system.

Metrics
  * SPR  - answer script == expected script
  * LFR  - script AND language both match (language checked with fastText when available)
  * Citation validity / grounding - cited ids exist in the corpus / were in the context
  * Faithfulness / relevance - LLM judge (use a judge that differs from the answer model)
  * False-premise rejection and out-of-scope refusal - LLM judge on the trap questions
  * Retrieval Recall@5 per mode (dense, sparse, hybrid, hybrid+rerank) and +graph lift
  * Latency: retrieval, first sentence (the honest text-side TTFA proxy), total turn
  * 95% bootstrap confidence intervals (n=50 is small; report the uncertainty)

Usage:
    python scripts/evaluate.py                       # full run, resumable
    python scripts/evaluate.py --judge ollama        # local judge
    python scripts/evaluate.py --no-resume           # ignore the checkpoint
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from server.config import get_settings  # noqa: E402
from server.lang.detect import get_detector, majority_script  # noqa: E402
from server.rag.citations import strip_tags  # noqa: E402
from server.rag.db import close_pool, get_pool  # noqa: E402
from server.rag.question_type import detect_question_type  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")

RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_CHECKPOINT = RESULTS_DIR / "eval_partial.jsonl"
CHUNK_ID_RE = re.compile(r"\[([a-zA-Z0-9_\-]+::[0-9]+|graph)\]")
NOT_MEASURED = "not measured — needs a live-microphone run"

# Targets from the project plan; "up" means higher is better.
TARGETS = {
    "spr": (0.95, "up"), "lfr": (0.90, "up"), "citation_validity": (1.0, "up"),
    "citation_grounded": (0.90, "up"), "faithfulness": (0.85, "up"), "relevance": (0.85, "up"),
    "recall_hybrid": (0.90, "up"), "first_sentence_p50": (700.0, "down"),
}


# ------------------------------------------------------------------ pure metric helpers
def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as fh:
        return {r["chunk_id"]: r for r in (json.loads(line) for line in fh if line.strip())}


def flavour_group(flavour: str) -> str:
    """regional-mr / regional-bn / regional-ta report together as 'regional'."""
    return "regional" if flavour.startswith("regional") else flavour


def calculate_percentiles(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p90": 0.0, "mean": 0.0, "count": 0}
    s = sorted(latencies)
    n = len(s)
    return {"p50": round(s[int(0.50 * (n - 1))], 2), "p90": round(s[int(0.90 * (n - 1))], 2),
            "mean": round(sum(s) / n, 2), "count": n}


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(iters))
    return (round(means[int(0.025 * iters)], 4), round(means[int(0.975 * iters) - 1], 4))


def rate(values: list[bool | float]) -> dict[str, Any]:
    vals = [float(v) for v in values]
    if not vals:
        return {"value": None, "ci95": None, "n": 0}
    return {"value": round(sum(vals) / len(vals), 4), "ci95": bootstrap_ci(vals), "n": len(vals)}


def calculate_recall_at_k(gold: list[str], retrieved: list[str], k: int = 5) -> float:
    if not gold:
        return 1.0
    top = retrieved[:k]
    return sum(1 for g in gold if g in top) / len(gold)


def calculate_spr(records: list[dict]) -> dict[str, Any]:
    if not records:
        return {"spr_overall": 0.0, "per_flavour": {}, "count": 0}
    by_flavour: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        exp = r.get("expected_script") or r.get("script") or "Latn"
        by_flavour[flavour_group(r.get("flavour", "unknown"))].append(
            majority_script(strip_tags(r.get("answer", ""))) == exp)
    allv = [m for v in by_flavour.values() for m in v]
    return {"spr_overall": round(sum(allv) / len(allv), 4),
            "per_flavour": {f: round(sum(v) / len(v), 4) for f, v in by_flavour.items()},
            "count": len(records)}


def verify_citations(answer: str, valid_corpus_ids: set[str],
                     context_chunk_ids: set[str]) -> dict[str, Any]:
    """Checks [chunk_id] tags against the corpus and against the context actually shown."""
    raw = CHUNK_ID_RE.findall(answer)
    cited = [c for c in raw if c != "graph"]
    if not cited:
        return {"has_citations": bool(raw), "cited_chunks": [],
                "valid_citation_rate": 1.0, "grounded_in_context_rate": 1.0}
    return {"has_citations": True, "cited_chunks": cited,
            "valid_citation_rate": round(sum(c in valid_corpus_ids for c in cited) / len(cited), 4),
            "grounded_in_context_rate":
                round(sum(c in context_chunk_ids for c in cited) / len(cited), 4)}


def language_matches(expected_lang: str, answer: str, detected_lang: str) -> bool:
    """Reply language == question language. Hinglish accepts 'hi-en'/'en' in Latin script."""
    exp = expected_lang.split("-")[0]
    det = detected_lang.split("-")[0]
    if expected_lang == "hi-en":
        return majority_script(answer) == "Latn" and det in {"hi", "en"}
    return det == exp


def status(key: str, value: float | None) -> str:
    if value is None:
        return "n/a"
    target, direction = TARGETS[key]
    ok = value >= target if direction == "up" else value <= target
    return "✅ Met" if ok else "❌ Not met"


def load_partial_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in data:
                    records[data["id"]] = data
    return records


def save_partial_record(record: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _passed(records: list[dict]) -> int:
    return sum(bool(r["judge_results"]["refusal_match"]) for r in records)


# ------------------------------------------------------------------------------- judges
JUDGE_PROMPT = """You are grading whether an AI assistant handled a tricky question correctly.
Question: {question}
Gold behaviour: {gold}
Assistant's answer: {answer}

Reply with strictly one word: "yes" if the answer matches the gold behaviour (rejects the
false premise / refuses the out-of-scope question rather than confabulating or answering
anyway), or "no" otherwise."""

FAITHFULNESS_PROMPT = """Question: {question}
Sources given to the assistant:
{context}
Assistant's answer: {answer}

Is every factual claim in the answer supported by the sources (faithfulness), and does the
answer actually address the question (relevance)? Reply with strictly one line:
"faithful=yes/no relevant=yes/no"."""


def _retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "too many requests", "503", "timeout", "rate"))


async def with_retry(fn, attempts: int = 6):
    from tenacity import (
        AsyncRetrying,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    async for attempt in AsyncRetrying(stop=stop_after_attempt(attempts), reraise=True,
                                       wait=wait_exponential_jitter(initial=2.0, max=45.0),
                                       retry=retry_if_exception(_retryable)):
        with attempt:
            return await fn()


async def judge(judge_llm, prompt: str) -> str:
    return await with_retry(lambda: judge_llm.complete([{"role": "user", "content": prompt}],
                                                       max_wait=45.0))


def make_judge(kind: str, model: str | None):
    if kind == "ollama":
        from server.providers.llm_ollama import OllamaLLM

        return OllamaLLM(model=model)
    from server.providers.llm_groq import GroqLLM

    return GroqLLM(model=model or "openai/gpt-oss-20b")


# --------------------------------------------------------------------------- retrieval
RERANK_PACE_S = 7.0     # Cohere trial key: 10 rerank calls/min

async def retrieval_ablation(golden: list[dict], chunk_article: dict[str, str]) -> dict[str, Any]:
    """Article-level Recall@5 for every retrieval mode, plus the additive graph lift."""
    from server.rag.orchestrate import orchestrate

    questions = [q for q in golden if q.get("gold_articles")]
    modes = ["dense", "sparse", "hybrid_norerank", "hybrid"]
    recalls: dict[str, list[float]] = {m: [] for m in modes}
    graph_vec: list[float] = []
    graph_comb: list[float] = []
    router_agree = 0

    from server.rag import retrieve as retrieve_mod

    get_settings().rerank_timeout_s = 30.0      # never let a slow call silently become the fallback
    retrieve_mod.STATS["rerank_fallbacks"] = 0

    for q in questions:
        gold = q["gold_articles"]
        for mode in modes:
            if mode == "hybrid":
                await asyncio.sleep(RERANK_PACE_S)
            res = await orchestrate(q["question"], question_type="factoid", mode=mode,
                                    rerank_top_k=5)
            recalls[mode].append(calculate_recall_at_k(
                gold, [h.article for h in res.vector_only_hits], 5))
        if detect_question_type(q["question"]) == q["type"] or \
                (q["type"] not in ("multi_hop", "comparative")
                 and detect_question_type(q["question"]) == "factoid"):
            router_agree += 1
        if q["type"] in ("multi_hop", "comparative"):
            await asyncio.sleep(RERANK_PACE_S)
            res = await orchestrate(q["question"], question_type=q["type"], rerank_top_k=5)
            vec = {h.article for h in res.vector_only_hits}
            comb = vec | {chunk_article.get(c) for gh in res.graph_paths
                          for c in gh.source_chunk_ids}
            comb.discard(None)
            graph_vec.append(sum(g in vec for g in gold) / len(gold))
            graph_comb.append(sum(g in comb for g in gold) / len(gold))

    out: dict[str, Any] = {"n": len(questions), "modes": {m: rate(v) for m, v in recalls.items()}}
    out["rerank_fallbacks"] = retrieve_mod.STATS["rerank_fallbacks"]
    out["rerank_valid"] = out["rerank_fallbacks"] == 0   # False: "hybrid" numbers are not reranked
    out["graph"] = {"n": len(graph_vec), "vector_only": rate(graph_vec),
                    "vector_plus_graph": rate(graph_comb)}
    if graph_vec:
        out["graph"]["lift"] = round(sum(graph_comb) / len(graph_comb)
                                     - sum(graph_vec) / len(graph_vec), 4)
    out["router_agreement_with_labels"] = round(router_agree / len(questions), 4) \
        if questions else None
    out["graph_routing_in_ablation"] = "golden-set labels (production uses the keyword router)"
    return out


# -------------------------------------------------------------------------- evaluation
async def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from server.rag.answer import answer_question

    try:
        pool = await get_pool()
        async with asyncio.timeout(3.0):
            await pool.fetchval("SELECT 1")
    except Exception as exc:  # noqa: BLE001
        logger.error("Database unreachable (%s). Start it and index the corpus first: "
                     "docker compose up -d postgres && python scripts/index_corpus.py", exc)
        raise SystemExit(2) from exc

    golden = load_golden(args.golden)[: args.max_samples]
    corpus = load_corpus(args.corpus)
    corpus_ids = set(corpus)
    chunk_article = {cid: r["article"] for cid, r in corpus.items()}
    detector = get_detector()
    judge_llm = make_judge(args.judge, args.judge_model)

    cached = load_partial_records(args.checkpoint) if args.resume else {}
    records: list[dict] = []

    for idx, q in enumerate(golden, start=1):
        if q["id"] in cached:
            records.append(cached[q["id"]])
            continue
        if q["type"] == "coreference":     # needs earlier turns; a single-shot run can't score it
            records.append({**q, "skipped": "needs conversation history"})
            continue
        logger.info("[%d/%d] %s [%s]", idx, len(golden), q["id"], q["flavour"])

        try:
            res = await with_retry(lambda q=q: answer_question(
                q["question"], question_type=q["type"], use_cache=False))
        except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating
            logger.error("generation failed for %s: %r", q["id"], exc)
            records.append({**q, "error": repr(exc)})
            continue

        det = await detector.detect(strip_tags(res.text))
        cites = verify_citations(res.raw_text, corpus_ids, set(res.context_ids))
        context_text = "\n".join(f"- {s['text'][:250]}" for s in res.sources[:5])

        judged: dict[str, Any] = {}
        try:
            if q["type"] in ("negative", "out_of_scope"):
                raw = await judge(judge_llm, JUDGE_PROMPT.format(
                    question=q["question"], gold=q.get("gold_answer", ""), answer=res.text))
                judged["refusal_match"] = raw.strip().lower().startswith("yes")
            else:
                raw = (await judge(judge_llm, FAITHFULNESS_PROMPT.format(
                    question=q["question"], context=context_text or "(none cited)",
                    answer=res.text))).lower()
                judged["faithful"] = "faithful=yes" in raw
                judged["relevant"] = "relevant=yes" in raw
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge failed for %s: %r", q["id"], exc)

        record = {
            **q, "answer": res.text, "raw_answer": res.raw_text,
            "fidelity_passed": res.fidelity_passed, "repairs_used": res.repairs,
            "detected_lang": det.lang, "language_match":
                language_matches(q["expected_lang"], res.text, det.lang),
            "retrieval_ms": res.timing["retrieval_ms"],
            "first_sentence_ms": res.timing["first_sentence_ms"],
            "total_turn_ms": res.timing["total_ms"],
            "citation_info": cites, "invalid_citations": res.invalid_citations,
            "judge_results": judged,
        }
        save_partial_record(record, args.checkpoint)
        records.append(record)
        await asyncio.sleep(1.0)     # stay inside free-tier per-minute limits

    scored = [r for r in records if "answer" in r]
    errors = [r["id"] for r in records if "error" in r]
    skipped = [r["id"] for r in records if "skipped" in r]

    by_f_lfr: dict[str, list[bool]] = defaultdict(list)
    for r in scored:
        by_f_lfr[flavour_group(r["flavour"])].append(
            bool(r["fidelity_passed"] and r["language_match"]
                 and majority_script(strip_tags(r["answer"])) == r["expected_script"]))
    lfr_all = [v for vs in by_f_lfr.values() for v in vs]

    cited = [r for r in scored if r["citation_info"]["cited_chunks"]]
    faith = [r for r in scored if "faithful" in r["judge_results"]]
    neg = [r for r in scored if r["type"] == "negative" and "refusal_match" in r["judge_results"]]
    oos = [r for r in scored if r["type"] == "out_of_scope"
           and "refusal_match" in r["judge_results"]]

    retrieval = None if args.skip_retrieval_ablation else \
        await retrieval_ablation(golden, chunk_article)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "n_golden": len(golden), "n_scored": len(scored), "errors": errors, "skipped": skipped,
        "answer_model": get_settings().llm_model, "profile": get_settings().bhasha_profile,
        "judge": f"{args.judge}:{args.judge_model or 'default'}",
        "language_check": ("fasttext" if detector._model is not None
                           else "script-only (no fastText model)"),
        "spr": calculate_spr(scored),
        "lfr": {"overall": rate(lfr_all),
                "per_flavour": {f: rate(v) for f, v in by_f_lfr.items()}},
        "citations": {
            "presence": rate([bool(r["citation_info"]["has_citations"]) for r in scored]),
            "validity": rate([r["citation_info"]["valid_citation_rate"] for r in cited]),
            "grounded": rate([r["citation_info"]["grounded_in_context_rate"] for r in cited]),
            "invalid_tags_removed": sum(r["invalid_citations"] for r in scored)},
        "judge_metrics": {
            "faithfulness": rate([r["judge_results"]["faithful"] for r in faith]),
            "relevance": rate([r["judge_results"]["relevant"] for r in faith]),
            "false_premise_rejection": {"passed": _passed(neg), "n": len(neg)},
            "out_of_scope_refusal": {"passed": _passed(oos), "n": len(oos)}},
        "retrieval": retrieval,
        "latency_ms": {k: calculate_percentiles([r[k] for r in scored])
                       for k in ("retrieval_ms", "first_sentence_ms", "total_turn_ms")},
        "audio_metrics": {k: NOT_MEASURED for k in ("wer", "cer", "ttfa_p50_ms", "ttfa_p95_ms",
                                                    "barge_in_latency_ms")},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("report saved to %s", args.output)
    print("\n" + format_markdown_report(report))
    await close_pool()
    return report


# ----------------------------------------------------------------------------- report
def _pct(m: dict | None) -> tuple[float | None, str]:
    if not m or m.get("value") is None:
        return None, "n/a"
    lo, hi = m["ci95"]
    return m["value"], (f"{m['value'] * 100:.1f}% (95% CI {lo * 100:.0f}–{hi * 100:.0f}%, "
                        f"n={m['n']})")


def format_markdown_report(res: dict[str, Any]) -> str:
    spr = res.get("spr", {}).get("spr_overall")
    lfr_v, lfr_s = _pct(res.get("lfr", {}).get("overall"))
    cit, jm = res.get("citations", {}), res.get("judge_metrics", {})
    val_v, val_s = _pct(cit.get("validity"))
    gnd_v, gnd_s = _pct(cit.get("grounded"))
    _, pres_s = _pct(cit.get("presence"))
    fa_v, fa_s = _pct(jm.get("faithfulness"))
    re_v, re_s = _pct(jm.get("relevance"))
    ret = res.get("retrieval") or {}
    modes = ret.get("modes", {})
    hyb_v, hyb_s = _pct(modes.get("hybrid"))
    lat = res.get("latency_ms", {})
    fs50 = lat.get("first_sentence_ms", {}).get("p50")

    def trap(key: str) -> str:
        t = jm.get(key) or {}
        return f"{t.get('passed', 0)} / {t.get('n', 0)}" if t else "n/a"

    rows = [
        ("🔒 Script", "Script Preservation Rate (SPR)", "> 95%",
         f"{spr * 100:.1f}%" if spr is not None else "n/a", status("spr", spr)),
        ("🔒 Contract", "Language Fidelity Rate (LFR)", "> 90%", lfr_s, status("lfr", lfr_v)),
        ("📚 Grounding", "Citation Presence Rate", "—", pres_s, ""),
        ("📚 Grounding", "Citation Validity Rate (in corpus)", "100%", val_s,
         status("citation_validity", val_v)),
        ("📚 Grounding", "Citation Grounding Rate (in context)", "> 90%", gnd_s,
         status("citation_grounded", gnd_v)),
        ("⚖️ Judge", "Faithfulness", "> 85%", fa_s, status("faithfulness", fa_v)),
        ("⚖️ Judge", "Answer Relevance", "> 85%", re_s, status("relevance", re_v)),
        ("🪤 Traps", "False-Premise Rejection", "all", trap("false_premise_rejection"), ""),
        ("🚧 Traps", "Out-of-Scope Refusal", "all", trap("out_of_scope_refusal"), ""),
    ]
    for m in ("dense", "sparse", "hybrid_norerank"):
        rows.append(("🔎 Retrieval", f"Recall@5 — {m}", "baseline", _pct(modes.get(m))[1], ""))
    rows.append(("🔎 Retrieval", "Recall@5 — hybrid + rerank", "> 90%", hyb_s,
                 status("recall_hybrid", hyb_v)))
    g = ret.get("graph") or {}
    if g.get("n"):
        before = _pct(g["vector_only"])[1].split(" ")[0]
        after = _pct(g["vector_plus_graph"])[1].split(" ")[0]
        lift = g.get("lift", 0)
        rows.append(("🕸️ Graph", f"Recall@5 multi-hop: vector → +graph (n={g['n']})", "lift > 0",
                     f"{before} → {after} (Δ {lift * 100:+.1f} pts)",
                     "✅ Positive" if lift > 0 else "❌ No lift"))
    rows.append(("⚡ Latency", "First sentence P50 / P90 (text side)", "< 700 ms",
                 f"{fs50} ms / {lat.get('first_sentence_ms', {}).get('p90')} ms"
                 if fs50 is not None else "n/a", status("first_sentence_p50", fs50)))
    rows.append(("⚡ Latency", "Total turn P50 / P90", "—",
                 f"{lat.get('total_turn_ms', {}).get('p50', 'n/a')} ms / "
                 f"{lat.get('total_turn_ms', {}).get('p90', 'n/a')} ms", ""))

    lines = [
        "### 📊 BHASHA System Evaluation Benchmark Report",
        f"*Generated {res.get('generated_at', '')} · answer model `{res.get('answer_model')}` · "
        f"judge `{res.get('judge')}` · language check: {res.get('language_check')}*",
        f"*Scored {res.get('n_scored', '?')} of {res.get('n_golden', '?')} questions; "
        f"{len(res.get('errors', []))} errored, {len(res.get('skipped', []))} skipped.*", "",
        "| Category | Metric | Target | Measured | Status |", "|---|---|---|---|---|",
        *[f"| {c} | {m} | {t} | **{v}** | {s} |" for c, m, t, v, s in rows],
        "", "#### Per-flavour breakdown", "| Flavour | SPR | LFR |", "|---|---|---|",
    ]
    spr_f = res.get("spr", {}).get("per_flavour", {})
    lfr_f = res.get("lfr", {}).get("per_flavour", {})
    for f in sorted(set(spr_f) | set(lfr_f)):
        lv = (lfr_f.get(f) or {}).get("value")
        lines.append(f"| `{f}` | {spr_f.get(f, 0) * 100:.1f}% | "
                     f"{'n/a' if lv is None else f'{lv * 100:.1f}%'} |")
    lines += ["", "_Audio metrics (WER/CER, true TTFA, barge-in latency): "
              f"{NOT_MEASURED}._"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="BHASHA evaluation harness")
    ap.add_argument("--golden", type=Path, default=ROOT_DIR / "data" / "golden_set.jsonl")
    ap.add_argument("--corpus", type=Path, default=ROOT_DIR / "data" / "corpus.jsonl")
    ap.add_argument("--output", type=Path, default=RESULTS_DIR / "final_eval.json")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--judge", choices=["groq", "ollama"], default="groq")
    ap.add_argument("--judge-model", default=None,
                    help="should differ from the answer model to avoid self-grading")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip-retrieval-ablation", action="store_true")
    asyncio.run(run_evaluation(ap.parse_args()))


if __name__ == "__main__":
    main()
