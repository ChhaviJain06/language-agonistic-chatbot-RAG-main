"""Builds the messages+context fed into lang.contract.generate_with_contract().

Citation is forced two ways: the provider system prompt numbers every source chunk by
chunk_id, and this module's instruction makes citing non-optional and forbids inventing
one (C8). Graph paths ride along as a synthetic context entry so a multi-hop answer can
follow the actual reasoning chain (e.g. "Shah Jahan -FATHER_OF-> Aurangzeb").
"""

from __future__ import annotations

from server.providers.base import GraphHit
from server.rag.orchestrate import OrchestrationResult

ANSWER_INSTRUCTION = """Answer the user's question using ONLY the sources given to you.
Every factual claim must be followed by the chunk_id(s) that support it, like [chunk_id].
Keep the answer short and natural (2-5 sentences) because it may be read aloud.
If the sources do not contain the answer, or the question is not about Indian history,
say so plainly in the user's language instead of guessing — never invent a citation.
If the question's premise is factually wrong according to the sources, correct the
premise before anything else. Use earlier messages only to resolve words like "he"/"his"."""


def render_graph_paths(graph_paths: list[GraphHit]) -> str:
    if not graph_paths:
        return ""
    lines = ["Reasoning chains from the knowledge graph (for context; do not cite 'graph'):"]
    for hit in graph_paths[:10]:
        parts: list[str] = []
        for step in hit.path:
            if "entity" in step:
                parts.append(str(step["entity"]))
            else:
                rel = step.get("rel", "RELATED_TO")
                parts.append(f"<-{rel}-" if step.get("dir") == "in" else f"-{rel}->")
        sources = ", ".join(hit.source_chunk_ids)
        lines.append(f"{' '.join(parts)} (evidence: {sources})")
    return "\n".join(lines)


def build_messages(question: str, history: list[dict] | None = None) -> list[dict]:
    messages = [m for m in (history or []) if m.get("role") in ("user", "assistant")]
    messages.append({"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\nQuestion: {question}"})
    return messages


def build_context(result: OrchestrationResult) -> list[dict]:
    """[{"chunk_id", "text"}] — what the provider numbers into the system prompt."""
    context = [{"chunk_id": h.chunk_id, "text": h.text} for h in result.hits]
    graph_block = render_graph_paths(result.graph_paths)
    if graph_block:
        context.append({"chunk_id": "graph", "text": graph_block})
    return context
