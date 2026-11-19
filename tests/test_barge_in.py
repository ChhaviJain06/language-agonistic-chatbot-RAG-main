"""Barge-in must leave exactly the HEARD prefix behind and no orphan tasks."""

import asyncio

import pytest

from server.barge_in import BargeInController, SpokenTracker


@pytest.mark.parametrize("acked_chars", [30, 60, 90])
async def test_char_offset_rewind(acked_chars):
    tracker = SpokenTracker()
    tracker.append_generated("A" * 100)
    controller = BargeInController(tracker=tracker)
    llm = asyncio.create_task(asyncio.Event().wait())
    controller.bind_llm_task(llm)
    tracker.mark_spoken(acked_chars)
    spoken = await controller.barge_in()
    assert spoken == "A" * acked_chars and llm.cancelled()


async def test_barge_in_leaves_no_orphan_tasks():
    before = {t for t in asyncio.all_tasks() if not t.done()}
    controller = BargeInController()
    controller.bind_llm_task(asyncio.create_task(asyncio.Event().wait()))
    controller.bind_tts_task(asyncio.create_task(asyncio.Event().wait()))
    await controller.barge_in()
    assert {t for t in asyncio.all_tasks() if not t.done()} - before == set()


def test_timing_rewind_counts_sentences_at_least_half_played():
    t = SpokenTracker()
    for text in ("One.", "Two.", "Three."):
        t.register_sentence(text, 2.0)
    assert t.heard_sentences(now=0) == []
    t.start_playback(now=100.0)
    assert t.spoken_text_at(now=100.9) == ""                  # < 1 s of sentence 1
    assert t.spoken_text_at(now=101.0) == "One."
    assert t.spoken_text_at(now=103.0) == "One. Two."         # 3 s: 2nd sentence half heard
    assert t.spoken_text_at(now=110.0) == "One. Two. Three."


def test_pause_and_resume_accumulates_played_time():
    t = SpokenTracker()
    t.register_sentence("A.", 1.0)
    t.register_sentence("B.", 1.0)
    t.start_playback(now=0.0)
    t.pause_playback(now=1.0)          # bot paused between sentences (synthesis gap)
    assert not t.fully_played(slack=0.0, now=50.0)
    t.start_playback(now=60.0)
    assert t.fully_played(slack=0.0, now=61.0)


async def test_freeze_locks_spoken_text():
    t = SpokenTracker()
    t.register_sentence("A.", 1.0)
    t.start_playback(now=0.0)
    c = BargeInController(tracker=t)
    t._segment_start = 0.0
    spoken = await c.barge_in()
    t.start_playback()                  # later playback must not change the frozen text
    assert t.spoken_text == spoken
