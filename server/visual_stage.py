"""Visual Stage telemetry (map, era, knowledge-graph, citation) for the React client.

Everything here is derived from the turn itself: the question/answer text, the graph
paths actually retrieved, and the sources actually cited. Nothing is invented — when
nothing matches, fields are null/empty and `matched` is False so the UI can say so
(C8 applies to the visual layer too; the old version fabricated an "ERA_OF" edge and a
"corpus::overview" citation).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from server.providers.base import GraphHit
from server.rag.retrieve import Hit

MAX_NODES = 24

# [latitude, longitude] of the best-known place for each entity.
HISTORICAL_GEO: dict[str, list[float]] = {
    "ashoka": [25.61, 85.14], "maurya": [25.61, 85.14], "chandragupta": [25.61, 85.14],
    "bindusara": [25.61, 85.14], "pataliputra": [25.61, 85.14], "kalinga": [20.19, 85.84],
    "akbar": [27.18, 78.01], "mughal": [28.61, 77.21], "aurangzeb": [19.89, 75.34],
    "shah jahan": [27.17, 78.04], "taj mahal": [27.17, 78.04], "babur": [28.61, 77.21],
    "humayun": [28.59, 77.25], "plassey": [23.80, 88.25], "panipat": [29.39, 76.97],
    "shivaji": [18.23, 73.44], "maratha": [18.52, 73.85], "chola": [10.78, 79.13],
    "rajaraja": [10.78, 79.13], "rajendra chola": [11.20, 79.45], "gupta": [25.61, 85.14],
    "samudragupta": [25.61, 85.14], "harappa": [30.63, 72.86], "mohenjo-daro": [27.32, 68.13],
    "indus valley": [30.63, 72.86], "jallianwala bagh": [31.62, 74.88], "amritsar": [31.63, 74.87],
    "chauri chaura": [26.65, 83.78], "gandhi": [23.02, 72.57], "constitution": [28.61, 77.21],
    "ambedkar": [28.61, 77.21], "vijayanagara": [15.34, 76.46], "hampi": [15.34, 76.46],
    "delhi sultanate": [28.61, 77.21], "tipu sultan": [12.42, 76.69], "1857": [28.66, 77.23],
    "sanchi": [23.48, 77.74], "nalanda": [25.14, 85.44], "buddha": [27.43, 83.27],
    "rani of jhansi": [25.45, 78.57], "bengal partition": [22.57, 88.36],
}

# (label, [start_year, end_year]) with BCE as negative numbers.
HISTORICAL_ERAS: dict[str, tuple[str, list[int]]] = {
    "maurya": ("Maurya Empire", [-322, -185]), "ashoka": ("Maurya Empire", [-268, -232]),
    "chandragupta": ("Maurya Empire", [-322, -298]), "kalinga": ("Kalinga War", [-261, -261]),
    "gupta": ("Gupta Empire", [319, 550]), "samudragupta": ("Gupta Empire", [335, 375]),
    "indus valley": ("Indus Valley Civilisation", [-3300, -1300]),
    "harappa": ("Indus Valley Civilisation", [-3300, -1300]),
    "mohenjo-daro": ("Indus Valley Civilisation", [-3300, -1300]),
    "chola": ("Chola Dynasty", [848, 1279]), "rajendra chola": ("Chola Dynasty", [1014, 1044]),
    "rajaraja": ("Chola Dynasty", [985, 1014]),
    "delhi sultanate": ("Delhi Sultanate", [1206, 1526]),
    "vijayanagara": ("Vijayanagara Empire", [1336, 1646]),
    "mughal": ("Mughal Empire", [1526, 1857]), "babur": ("Mughal Empire", [1526, 1530]),
    "humayun": ("Mughal Empire", [1530, 1556]), "akbar": ("Mughal Empire", [1556, 1605]),
    "shah jahan": ("Mughal Empire", [1628, 1658]), "taj mahal": ("Mughal Empire", [1632, 1653]),
    "aurangzeb": ("Mughal Empire", [1658, 1707]),
    "maratha": ("Maratha Empire", [1674, 1818]), "shivaji": ("Maratha Empire", [1674, 1680]),
    "panipat": ("Battles of Panipat", [1526, 1761]), "plassey": ("Battle of Plassey", [1757, 1757]),
    "tipu sultan": ("Kingdom of Mysore", [1782, 1799]),
    "1857": ("Revolt of 1857", [1857, 1858]),
    "jallianwala": ("British Raj", [1919, 1919]),
    "chauri chaura": ("Non-Cooperation", [1922, 1922]),
    "gandhi": ("Independence Movement", [1915, 1948]),
    "constitution": ("Republic of India", [1949, 1950]),
}


def _compile(table: dict) -> list[tuple[str, re.Pattern]]:
    # Longest key first so "rajendra chola" wins over "chola".
    return [(k, re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])"))
            for k in sorted(table, key=len, reverse=True)]


_GEO_PATTERNS = _compile(HISTORICAL_GEO)
_ERA_PATTERNS = _compile(HISTORICAL_ERAS)


def _first_match(patterns: list[tuple[str, re.Pattern]], *texts: str) -> str | None:
    for text in texts:                       # question wins over answer
        for key, pat in patterns:
            if pat.search(text):
                return key
    return None


@dataclass
class VisualPayload:
    event: str = "HISTORICAL_CONTEXT"
    matched: bool = False
    entity: str | None = None
    dynasty: str | None = None
    era_years: list[int] | None = None
    geo_coordinates: list[float] | None = None
    subgraph: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"nodes": [], "links": []})
    citation: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_subgraph(graph_paths: list[GraphHit]) -> dict[str, list[dict[str, Any]]]:
    """Nodes/links from real graph hits. Edge direction is preserved: a hop stored as
    'in' is flipped so every link reads source -[relation]-> target correctly."""
    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_links: set[tuple[str, str, str]] = set()

    def add_node(name: str, group: int) -> None:
        if name not in seen_nodes and len(nodes) < MAX_NODES:
            nodes.append({"id": name, "name": name, "group": group})
            seen_nodes.add(name)

    for gh in graph_paths:
        path = gh.path
        for i in range(0, len(path) - 2, 2):
            left, rel, right = path[i].get("entity"), path[i + 1], path[i + 2].get("entity")
            if not left or not right:
                continue
            src, tgt = (right, left) if rel.get("dir") == "in" else (left, right)
            relation = rel.get("rel", "RELATED_TO")
            add_node(src, 1 if i == 0 else 2)
            add_node(tgt, 2)
            key = (src, tgt, relation)
            if key not in seen_links and src in seen_nodes and tgt in seen_nodes:
                seen_links.add(key)
                links.append({"source": src, "target": tgt, "relation": relation})
    return {"nodes": nodes, "links": links}


def extract_visual_payload(question: str, answer: str, hits: list[Hit],
                           graph_paths: list[GraphHit],
                           sources: list[dict] | None = None) -> VisualPayload:
    q, a = question.lower(), answer.lower()
    geo_key = _first_match(_GEO_PATTERNS, q, a)
    era_key = _first_match(_ERA_PATTERNS, q, a)

    citation = None
    if sources:
        s = sources[0]
        citation = {"chunk_id": s["chunk_id"], "article": s["article"], "section": s["section"],
                    "text": s["text"][:400], "url": s.get("url", "")}

    subgraph = build_subgraph(graph_paths)
    payload = VisualPayload(
        entity=geo_key.title() if geo_key else None,
        dynasty=HISTORICAL_ERAS[era_key][0] if era_key else None,
        era_years=list(HISTORICAL_ERAS[era_key][1]) if era_key else None,
        geo_coordinates=list(HISTORICAL_GEO[geo_key]) if geo_key else None,
        subgraph=subgraph,
        citation=citation,
    )
    payload.matched = bool(geo_key or era_key or subgraph["nodes"] or citation)
    return payload
