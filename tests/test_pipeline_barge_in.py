"""BhashaBrain with REAL Pipecat frames and stubbed I/O. Covers the three situations the
old implementation got wrong: interrupt during generation, interrupt DURING PLAYBACK
(the common case), and a failing turn (must never leave the caller in silence)."""

from __future__ import annotations

import asyncio
import json

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from server.lang.contract import LangContract
from server.lang.contract import chunk_text as split_sentences
from server.pipeline import BhashaBrain, BhashaSpeakFrame, SentenceTimingFrame
from server.pipeline_text import clean_tts_text
from server.rag.answer import AnswerResult
from server.rag.question_type import detect_question_type
from server.session import SessionState

DOWN = FrameDirection.DOWNSTREAM
ANSWER = "Ashoka ruled well. He built many stupas. He also spread Buddhism widely."


def _result(text: str) -> AnswerResult:
    return AnswerResult("q", LangContract("en", "Latn"), text, text, [], True, 0, False,
                        {"matched": False}, {"total_ms": 1.0})


@pytest.fixture
def brain(monkeypatch):
    b = BhashaBrain(SessionState(room="r"))
    pushed: list = []

    async def fake_push(frame, direction=DOWN):
        pushed.append(frame)

    async def noop(*a, **k):
        return None

    async def contract(q, **k):
        return LangContract("en", "Latn")

    b.push_frame = fake_push
    b.broadcast_interruption = _Counter()
    monkeypatch.setattr("server.pipeline.build_contract", contract)
    monkeypatch.setattr("server.pipeline.maybe_cache", noop)
    monkeypatch.setattr("server.pipeline.store.log_exchange", noop)
    b._pushed = pushed
    return b


class _Counter:
    calls = 0

    async def __call__(self, *a, **k):
        self.calls += 1


def _fake_answer(text=ANSWER, gate: asyncio.Event | None = None):
    async def fake(question, *, on_sentence=None, **kw):
        if gate:
            await gate.wait()
        for sent in split_sentences(text):
            await on_sentence(sent)
        return _result(text)
    return fake


def _messages(brain) -> list[dict]:
    return [json.loads(f.message) for f in brain._pushed
            if isinstance(f, OutputTransportMessageUrgentFrame)]


async def _say(brain, text="Tell me about Ashoka"):
    await brain.process_frame(TranscriptionFrame(text=text, user_id="", timestamp="0"), DOWN)


async def test_full_turn_speaks_sentences_and_commits_when_playback_ends(brain, monkeypatch):
    monkeypatch.setattr("server.pipeline.answer_question", _fake_answer())
    await _say(brain)
    await brain._turn_task
    spoken = [f.text for f in brain._pushed if isinstance(f, BhashaSpeakFrame)]
    assert spoken == split_sentences(ANSWER)
    assert brain._turn is not None and not brain._turn.committed     # still playing

    await brain.process_frame(BotStartedSpeakingFrame(), DOWN)
    for s in spoken:
        await brain.process_frame(SentenceTimingFrame(text=s, seconds=1.0), FrameDirection.UPSTREAM)
    brain._turn.controller.tracker._segment_start -= 3.0                # 3 s of audio played
    await brain.process_frame(BotStoppedSpeakingFrame(), DOWN)

    assert brain._turn is None
    assert brain.state.history[-1] == {"role": "assistant", "content": ANSWER}


async def test_each_message_is_sent_once(brain, monkeypatch):
    monkeypatch.setattr("server.pipeline.answer_question", _fake_answer())
    await _say(brain)
    await brain._turn_task
    events = [m["event"] for m in _messages(brain)]
    assert events == ["status", "answer"]                 # no duplicates
    assert len({m["msg_id"] for m in _messages(brain)}) == 1


async def test_barge_in_during_generation_cancels_and_keeps_only_the_question(brain, monkeypatch):
    gate = asyncio.Event()
    monkeypatch.setattr("server.pipeline.answer_question", _fake_answer(gate=gate))
    await _say(brain)
    task = brain._turn_task
    assert not task.done()

    await brain.process_frame(VADUserStartedSpeakingFrame(), DOWN)
    assert task.cancelled() and brain._turn is None
    assert brain.broadcast_interruption.calls == 1
    assert any(m.get("event") == "barge_in" for m in _messages(brain))
    assert brain.state.history == [{"role": "user", "content": "Tell me about Ashoka"}]


async def test_barge_in_during_playback_rewinds_to_what_was_heard(brain, monkeypatch):
    """The common case: generation is long finished, the bot is mid-answer."""
    monkeypatch.setattr("server.pipeline.answer_question", _fake_answer())
    await _say(brain)
    await brain._turn_task                                    # generation done
    sentences = split_sentences(ANSWER)

    await brain.process_frame(BotStartedSpeakingFrame(), DOWN)
    for s in sentences:
        await brain.process_frame(SentenceTimingFrame(text=s, seconds=2.0), FrameDirection.UPSTREAM)
    brain._turn.controller.tracker._segment_start -= 3.0      # 3 s into 3 x 2 s of audio

    await brain.process_frame(VADUserStartedSpeakingFrame(), DOWN)

    assert brain.broadcast_interruption.calls == 1 and brain._turn is None
    heard = brain.state.history[-1]["content"]
    assert heard == " ".join(sentences[:2])                   # sentence 3 was never heard
    assert "Buddhism" not in heard
    barge = next(m for m in _messages(brain) if m.get("event") == "barge_in")
    assert barge["spoken_text"] == heard


async def test_failing_turn_apologises_in_the_users_language(brain, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr("server.pipeline.answer_question", boom)
    await _say(brain)
    await brain._turn_task
    err = next(m for m in _messages(brain) if m["event"] == "error")
    assert "Sorry" in err["error"]
    assert any(isinstance(f, BhashaSpeakFrame) and "Sorry" in f.text for f in brain._pushed)


async def test_history_is_sent_to_the_next_turn(brain, monkeypatch):
    seen = {}

    async def fake(question, *, on_sentence=None, history=None, **kw):
        seen["history"] = history
        await on_sentence("Ok.")
        return _result("Ok.")

    monkeypatch.setattr("server.pipeline.answer_question", fake)
    brain.state.remember("Who built the Taj Mahal?", "Shah Jahan.")
    await _say(brain, "And his son?")
    await brain._turn_task
    assert seen["history"][0]["content"] == "Who built the Taj Mahal?"




def test_clean_tts_text_removes_citations_and_markers():
    raw = "Ashoka ruled [ashoka::0001] well [2]. **Yes**"
    assert clean_tts_text(raw) == "Ashoka ruled well. Yes"


def test_detect_question_type_uses_whole_words():
    assert detect_question_type("Who was the grandfather of Aurangzeb?") == "multi_hop"
    assert detect_question_type("Ashoka ke pita ka naam kya tha?") == "multi_hop"
    assert detect_question_type("Compare Akbar and Aurangzeb's religious policies") == "comparative"
    assert detect_question_type("Dono samrajyon me kya antar tha?") == "comparative"
    # these used to be mis-routed to the graph by substring matching:
    assert detect_question_type("When was the Battle of Plassey fought?") == "factoid"
    assert detect_question_type("What was the reason for the Swaraj movement?") == "factoid"
    assert detect_question_type("Where is the Great Bath?") == "factoid"
