"""Turns the model's raw `[chunk_id]` tags into verified, numbered citations (C8).

The model is told to cite by chunk_id. Here every tag is checked against the chunks
that were actually given to it: tags for unknown ids are dropped (never shown), and
the remaining ones become `[1]`, `[2]` markers backed by a `sources` list the UI can
open. So "citations" always means "chunks the answer really cited", not "everything
we retrieved".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from server.rag.retrieve import Hit

# [ashoka::0003]  [ashoka::0003, maurya::0001]  【ashoka::0003】  [graph]
_TAG_RE = re.compile(r"[\[【]\s*([^\[\]【】]*?)\s*[\]】]")
_ID_RE = re.compile(r"[A-Za-z0-9_\-]+::\d+")


@dataclass(frozen=True)
class NumberedAnswer:
    text: str                 # clean answer with [n] markers
    sources: list[dict]       # [{n, chunk_id, article, section, url, text, score}]
    invalid: int              # tags that referenced ids not in the retrieved context


def strip_tags(text: str) -> str:
    """Removes every chunk-id / [graph] tag — for TTS and for text shown mid-stream."""
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        return "" if (_ID_RE.search(inner) or inner.strip().lower() == "graph") else m.group(0)

    cleaned = _TAG_RE.sub(repl, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?।॥;:])", r"\1", cleaned)
    return cleaned.strip()


def extract_ids(text: str) -> list[str]:
    """Every chunk id cited in `text`, in order of appearance, with repeats."""
    ids: list[str] = []
    for m in _TAG_RE.finditer(text):
        ids.extend(_ID_RE.findall(m.group(1)))
    return ids


def number_citations(answer: str, hits: list[Hit]) -> NumberedAnswer:
    by_id = {h.chunk_id: h for h in hits}
    order: dict[str, int] = {}
    invalid = 0

    def repl(m: re.Match) -> str:
        nonlocal invalid
        inner = m.group(1)
        ids = _ID_RE.findall(inner)
        if not ids:
            return "" if inner.strip().lower() == "graph" else m.group(0)
        markers: list[str] = []
        for cid in ids:
            if cid not in by_id:
                invalid += 1
                continue
            n = order.setdefault(cid, len(order) + 1)
            marker = f"[{n}]"
            if marker not in markers:
                markers.append(marker)
        return "".join(markers)

    text = _TAG_RE.sub(repl, answer)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,!?।॥;:])", r"\1", text)

    sources = []
    for cid, n in sorted(order.items(), key=lambda kv: kv[1]):
        h = by_id[cid]
        sources.append({
            "n": n, "chunk_id": cid, "article": h.article, "section": h.section,
            "url": getattr(h, "url", "") or "", "text": h.text,
            "score": round(h.score, 4) if h.score else None,
        })
    return NumberedAnswer(text=text.strip(), sources=sources, invalid=invalid)
