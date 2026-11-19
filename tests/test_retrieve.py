"""RRF and the sparse-query builder are the retrieval pieces that can be wrong silently."""

from server.rag.retrieve import build_or_tsquery, rrf


def row(cid: str) -> dict:
    return {"chunk_id": cid, "article": "A", "section": "S", "text": cid}


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([[row("a"), row("b"), row("c")], [row("c"), row("b"), row("z")]])
    assert {f["chunk_id"] for f in fused[:2]} == {"b", "c"}
    assert [f["chunk_id"] for f in fused[2:]] == ["a", "z"]
    assert len(fused) == 4 and fused == sorted(fused, key=lambda r: r["score"], reverse=True)


def test_rrf_single_ranker_preserves_order_and_empty():
    assert [f["chunk_id"] for f in rrf([[row("a"), row("b")]])] == ["a", "b"]
    assert rrf([[], []]) == []


def test_sparse_query_is_or_of_content_words():
    q = build_or_tsquery("Which war made Ashoka give up violence?")
    assert q == "war | made | ashoka | give | up | violence"       # stopwords dropped, OR semantics


def test_sparse_query_uses_english_entities_for_devanagari_questions():
    assert build_or_tsquery("अशोक ने कलिंग युद्ध क्यों लड़ा?") is None
    assert build_or_tsquery("अशोक ने कलिंग युद्ध क्यों लड़ा?", ["Ashoka", "Kalinga War"]) == \
        "ashoka | kalinga | war"


def test_sparse_query_cannot_inject_tsquery_syntax():
    q = build_or_tsquery("a' | !(b) & c:* <-> d; DROP TABLE chunks")
    assert q is not None and not any(ch in q.replace(" | ", "") for ch in "'!&:*<>();")


async def test_skip_rerank_setting_only_affects_factoid(monkeypatch):
    from server.config import get_settings
    from server.rag import orchestrate as orch

    seen = []

    async def fake_retrieve(q, **kw):
        seen.append(kw["mode"])
        return []

    async def none_vec(_q):
        return [0.0]

    monkeypatch.setattr(orch, "retrieve", fake_retrieve)
    monkeypatch.setattr(orch, "embed_query", none_vec)
    monkeypatch.setattr(get_settings(), "skip_rerank_factoid", True)
    await orch.orchestrate("who was ashoka", question_type="factoid")
    await orch.orchestrate("compare a and b", question_type="comparative")
    await orch.orchestrate("x", question_type="factoid", mode="hybrid")   # explicit wins (ablation)
    assert seen == ["hybrid_norerank", "hybrid", "hybrid"]
