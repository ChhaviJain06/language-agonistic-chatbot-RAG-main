"""The one place a question becomes an answer — used by /query, /query/stream and the
voice pipeline, so detect -> cache -> retrieve -> generate(+contract) -> cite -> visualise
is implemented exactly once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from server.lang.contract import build_contract, generate_with_contract
from server.lang.normalize import normalize_typography
from server.providers.base import LangContract
from server.providers.factory import get_llm
from server.rag.cache import check_semantic_cache, store_semantic_cache
from server.rag.citations import number_citations, strip_tags
from server.rag.orchestrate import OrchestrationResult, orchestrate
from server.rag.prompt import build_context, build_messages
from server.rag.question_type import detect_question_type
from server.visual_stage import extract_visual_payload

logger = logging.getLogger(__name__)

SentenceCallback = Callable[[str], Awaitable[None]]


@dataclass
class AnswerResult:
    question: str
    contract: LangContract
    raw_text: str                       # as generated, with [chunk_id] tags
    text: str                           # clean, with [n] markers
    sources: list[dict]
    fidelity_passed: bool
    repairs: int
    cached: bool
    visual_stage: dict[str, Any]
    timing: dict[str, float] = field(default_factory=dict)
    similarity: float | None = None
    question_type: str = "factoid"
    context_ids: list[str] = field(default_factory=list)   # chunks the model was shown
    invalid_citations: int = 0                              # cited ids that were not in context

    def to_payload(self) -> dict[str, Any]:
        return {
            "question": self.question, "lang": self.contract.lang,
            "script": self.contract.script, "answer": self.text, "sources": self.sources,
            "fidelity_passed": self.fidelity_passed, "repairs": self.repairs,
            "cached": self.cached, "visual_stage": self.visual_stage, "timing": self.timing,
        }


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def answer_question(
    question: str, *, history: list[dict] | None = None, forced_lang: str | None = None,
    question_type: str | None = None, on_sentence: SentenceCallback | None = None,
    use_cache: bool = True, contract: LangContract | None = None,
) -> AnswerResult:
    t0 = time.perf_counter()
    history = history or []
    contract = contract or await build_contract(question, forced_lang=forced_lang)
    q_type = question_type if question_type and question_type != "factoid" \
        else detect_question_type(question)

    # A follow-up ("and his son?") depends on history, so it must bypass the cache.
    if use_cache and not history:
        hit = await check_semantic_cache(question, contract.lang, contract.script)
        if hit:
            hit = replace(hit, response_text=normalize_typography(hit.response_text))
            sources = hit.metadata.get("sources", [])
            if on_sentence:
                for sent in _split(hit.response_text):
                    await on_sentence(sent)
            visual = extract_visual_payload(question, hit.response_text, [], [], sources)
            total = _ms(t0)
            return AnswerResult(question, contract, hit.response_text, hit.response_text,
                                sources, True, 0, True, visual.to_dict(),
                                {"retrieval_ms": 0.0, "first_sentence_ms": total,
                                 "total_ms": total}, hit.similarity, q_type)

    prep_ms = _ms(t0)                    # language detection + embedding + cache lookup
    t_ret = time.perf_counter()
    result: OrchestrationResult = await orchestrate(
        question, question_type=q_type,
        translate_from=contract.lang if contract.lang != "en" else None)
    retrieval_ms = _ms(t_ret)

    messages = build_messages(question, history=history)
    context = build_context(result)

    first_sentence_ms: list[float] = []
    llm_first_ms: list[float] = []
    t_llm = time.perf_counter()

    async def tracked(sent: str) -> None:
        if not first_sentence_ms:
            first_sentence_ms.append(_ms(t0))
            llm_first_ms.append(_ms(t_llm))
        if on_sentence:
            await on_sentence(sent)

    raw, passed, repairs = await generate_with_contract(
        get_llm(), messages, contract, context, on_sentence=tracked)

    numbered = number_citations(raw, result.hits)
    visual = extract_visual_payload(question, numbered.text, result.hits,
                                    result.graph_paths, numbered.sources)
    timing = {"prep_ms": prep_ms, "retrieval_ms": retrieval_ms,
              "llm_first_chunk_ms": llm_first_ms[0] if llm_first_ms else _ms(t_llm),
              "first_sentence_ms": first_sentence_ms[0] if first_sentence_ms else _ms(t0),
              "total_ms": _ms(t0)}

    return AnswerResult(question, contract, raw, numbered.text, numbered.sources, passed,
                        repairs, False, visual.to_dict(), timing, None, q_type,
                        [h.chunk_id for h in result.hits], numbered.invalid)


def cacheable(res: AnswerResult, history: list[dict] | None) -> bool:
    """Only replay answers that passed the fidelity contract, are cited and are standalone."""
    return (not history and not res.cached and res.fidelity_passed and bool(res.sources)
            and bool(res.text.strip()))


async def maybe_cache(res: AnswerResult, history: list[dict] | None) -> None:
    if cacheable(res, history):
        await store_semantic_cache(res.question, res.text, res.contract.lang,
                                   res.contract.script, metadata={"sources": res.sources})


def _split(text: str) -> list[str]:
    from server.lang.contract import chunk_text

    return chunk_text(strip_tags(text))
