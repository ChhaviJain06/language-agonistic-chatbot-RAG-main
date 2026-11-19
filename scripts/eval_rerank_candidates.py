#!/usr/bin/env python3
"""eval_rerank_candidates.py — does reranking fewer candidates hurt retrieval quality?

Rerank time grows with the number of passages. This measures article-level Recall@5 on the
golden set when only the top-N fused candidates are reranked, for several N, using ONE rerank
call per question (the cross-encoder scores each passage independently, so the scores for the
first N candidates are just a prefix of the scores for all of them). Use it to choose
RERANK_CANDIDATES from data instead of guessing.

    python scripts/eval_rerank_candidates.py [--sleep 7]   # 7 s = Cohere trial limit (10/min)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.providers.factory import get_reranker  # noqa: E402
from server.rag import retrieve as R  # noqa: E402
from server.rag.db import close_pool  # noqa: E402
from server.rag.embedcache import embed_query  # noqa: E402

SIZES = (5, 8, 10, 12, 15, 20)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden_set.jsonl"))
    ap.add_argument("--sleep", type=float, default=7.0)
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.golden.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    questions = [q for q in rows if q.get("gold_articles")]
    recall: dict[int, list[float]] = {n: [] for n in SIZES}
    rerank_ms: list[float] = []
    reranker = get_reranker()

    for i, q in enumerate(questions, 1):
        vec = await embed_query(q["question"])
        dense, sparse = await asyncio.gather(R._dense(q["question"], 20, vec),
                                             R._sparse(q["question"], 20))
        cands = R.rrf([dense, sparse])[:20]
        t = time.monotonic()
        scores = await reranker.score(q["question"], [c["text"] for c in cands])
        rerank_ms.append((time.monotonic() - t) * 1000)
        for n in SIZES:
            top = sorted(zip(cands[:n], scores[:n], strict=True), key=lambda p: p[1],
                         reverse=True)[: args.k]
            found = {c["article"] for c, _ in top}
            recall[n].append(sum(g in found for g in q["gold_articles"]) / len(q["gold_articles"]))
        print(f"[{i}/{len(questions)}] {q['id']}", end="\r", flush=True)
        await asyncio.sleep(args.sleep)

    print(f"\nRecall@{args.k} by number of candidates reranked "
          f"(n={len(questions)} questions with gold articles):")
    for n in SIZES:
        print(f"  top-{n:<2} candidates: {sum(recall[n]) / len(recall[n]):.3f}")
    print(f"rerank call (20 passages): median {sorted(rerank_ms)[len(rerank_ms) // 2]:.0f} ms")
    await close_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
