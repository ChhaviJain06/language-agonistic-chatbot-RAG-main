"""Short-lived buffer of finished voice answers, keyed by room.

Why: a voice answer is pushed once over the LiveKit data channel. If the browser's
connection drops at that moment the answer would be lost. The agent keeps finishing the
turn (see pipeline.py), stores the answer payload here, and the browser can fetch it
from GET /voice/turns after it reconnects.

This is a transient delivery buffer, not conversation state: the conversation itself
(turns, spoken text) is still persisted in Postgres (C10). Payloads carry sources and
the Visual Stage, which the `turns` table has no columns for.
"""

from __future__ import annotations

import time
from collections import OrderedDict

MAX_TURNS_PER_ROOM = 20
MAX_ROOMS = 200
TTL_S = 600.0

# room -> (last_touched_monotonic, {msg_id: payload})
_rooms: OrderedDict[str, tuple[float, OrderedDict[str, dict]]] = OrderedDict()


def _prune(now: float) -> None:
    for room in [r for r, (t, _) in _rooms.items() if now - t > TTL_S]:
        del _rooms[room]
    while len(_rooms) > MAX_ROOMS:
        _rooms.popitem(last=False)


def record(room: str, payload: dict) -> None:
    msg_id = str(payload.get("msg_id") or "")
    if not room or not msg_id:
        return
    now = time.monotonic()
    _prune(now)
    turns = _rooms.pop(room, (now, OrderedDict()))[1]
    turns[msg_id] = payload
    while len(turns) > MAX_TURNS_PER_ROOM:
        turns.popitem(last=False)
    _rooms[room] = (now, turns)


def turns_for(room: str) -> list[dict]:
    _prune(time.monotonic())
    entry = _rooms.get(room)
    return list(entry[1].values()) if entry else []
