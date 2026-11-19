"""Replay buffer + /voice/turns, and the status events that drive the voice UI."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from pipecat.frames.frames import BotStartedSpeakingFrame, OutputTransportMessageUrgentFrame

from server import voice_replay
from server.main import app
from tests.test_pipeline_barge_in import (  # noqa: F401  (brain fixture)
    ANSWER,
    DOWN,
    _result,
    brain,  # noqa: F401
)


def test_record_and_fetch_keeps_order_and_caps():
    voice_replay._rooms.clear()
    for i in range(voice_replay.MAX_TURNS_PER_ROOM + 5):
        voice_replay.record("bhasha-aa", {"msg_id": f"m{i}", "answer": "x"})
    ids = [t["msg_id"] for t in voice_replay.turns_for("bhasha-aa")]
    assert len(ids) == voice_replay.MAX_TURNS_PER_ROOM
    assert ids[-1] == f"m{voice_replay.MAX_TURNS_PER_ROOM + 4}"
    assert voice_replay.turns_for("bhasha-unknown") == []


def test_turns_expire(monkeypatch):
    voice_replay._rooms.clear()
    voice_replay.record("bhasha-bb", {"msg_id": "m1"})
    t0 = voice_replay._rooms["bhasha-bb"][0]
    monkeypatch.setattr(voice_replay.time, "monotonic", lambda: t0 + voice_replay.TTL_S + 1)
    assert voice_replay.turns_for("bhasha-bb") == []


def test_endpoint_returns_recorded_turn_and_rejects_bad_room():
    voice_replay._rooms.clear()
    voice_replay.record("bhasha-0123abcd", {"msg_id": "m1", "event": "answer", "answer": "hi"})
    client = TestClient(app)
    ok = client.get("/voice/turns", params={"room": "bhasha-0123abcd"})
    assert ok.status_code == 200 and ok.json()["turns"][0]["answer"] == "hi"
    assert client.get("/voice/turns", params={"room": "../etc"}).status_code == 422


@pytest.mark.asyncio
async def test_brain_sends_speaking_status_and_records_answer(brain, monkeypatch):  # noqa: F811
    voice_replay._rooms.clear()

    async def fake_answer(q, **k):
        return _result(ANSWER)

    monkeypatch.setattr("server.pipeline.answer_question", fake_answer)
    await brain._start_turn("who was ashoka")
    await brain._turn_task
    await brain.process_frame(BotStartedSpeakingFrame(), DOWN)

    sent = [json.loads(f.message) for f in brain._pushed
            if isinstance(f, OutputTransportMessageUrgentFrame)]
    assert {"thinking", "speaking"} <= {m.get("status") for m in sent if m["event"] == "status"}
    answer = next(m for m in sent if m["event"] == "answer")
    # Same msg_id as the "thinking" status — the browser must not treat it as a duplicate.
    assert answer["msg_id"] == next(m for m in sent if m.get("status") == "thinking")["msg_id"]
    assert voice_replay.turns_for("r")[0]["msg_id"] == answer["msg_id"]
