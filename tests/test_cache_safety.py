"""Only safe answers may be replayed from the semantic cache."""

from server.providers.base import LangContract
from server.rag.answer import AnswerResult, cacheable


def res(**kw) -> AnswerResult:
    base = dict(question="q", contract=LangContract("hi", "Deva"), raw_text="r", text="उत्तर [1]",
                sources=[{"chunk_id": "a::1"}], fidelity_passed=True, repairs=0, cached=False,
                visual_stage={}, timing={})
    base.update(kw)
    return AnswerResult(**base)


def test_clean_cited_standalone_answers_are_cacheable():
    assert cacheable(res(), history=[])


def test_follow_up_questions_are_never_cached():
    assert not cacheable(res(), history=[{"role": "user", "content": "earlier"}])


def test_answers_that_failed_fidelity_or_have_no_citations_are_not_cached():
    assert not cacheable(res(fidelity_passed=False), [])
    assert not cacheable(res(sources=[]), [])
    assert not cacheable(res(text="  "), [])


def test_cache_hits_are_not_recached():
    assert not cacheable(res(cached=True), [])
