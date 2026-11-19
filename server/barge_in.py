"""Cancellation bus + conversation-state rewind for true barge-in.

The key correctness property: when the user interrupts, the turn's recorded
`spoken_text` must rewind to what was ACTUALLY played through the speaker, never to
how much text the LLM generated. Those two diverge constantly: generation is several
sentences ahead of the speaker.

How "actually played" is measured without a playback-position ack from the TTS vendor:
BhashaTTS reports the duration of audio it synthesised for every sentence
(`register_sentence`), and the output transport tells us when the bot starts/stops
speaking (`start_playback` / `pause_playback`). Played time -> how many whole sentences
were (mostly) heard. A sentence counts as heard once at least half of it has played.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

HEARD_FRACTION = 0.5


@dataclass
class SpokenTracker:
    # --- legacy character-offset API (kept for simple callers and tests) -----
    full_text: str = ""
    spoken_upto: int = 0
    # --- timing API --------------------------------------------------------
    sentences: list[tuple[str, float]] = field(default_factory=list)  # (text, audio seconds)
    _played: float = 0.0
    _segment_start: float | None = None
    _frozen: str | None = None

    def append_generated(self, delta: str) -> None:
        self.full_text += delta

    def mark_spoken(self, upto: int) -> None:
        self.spoken_upto = max(self.spoken_upto, upto)

    # -- timing ---------------------------------------------------------------
    def register_sentence(self, text: str, seconds: float) -> None:
        self.sentences.append((text, max(0.0, seconds)))

    def start_playback(self, now: float | None = None) -> None:
        if self._segment_start is None:
            self._segment_start = time.monotonic() if now is None else now

    def pause_playback(self, now: float | None = None) -> None:
        if self._segment_start is not None:
            now = time.monotonic() if now is None else now
            self._played += max(0.0, now - self._segment_start)
            self._segment_start = None

    def played_seconds(self, now: float | None = None) -> float:
        total = self._played
        if self._segment_start is not None:
            now = time.monotonic() if now is None else now
            total += max(0.0, now - self._segment_start)
        return total

    @property
    def total_audio_seconds(self) -> float:
        return sum(d for _, d in self.sentences)

    def fully_played(self, slack: float = 0.25, now: float | None = None) -> bool:
        return bool(self.sentences) and \
            self.played_seconds(now) + slack >= self.total_audio_seconds

    def heard_sentences(self, now: float | None = None) -> list[str]:
        played = self.played_seconds(now)
        heard: list[str] = []
        cursor = 0.0
        for text, dur in self.sentences:
            if played >= cursor + dur * HEARD_FRACTION:
                heard.append(text)
                cursor += dur
            else:
                break
        return heard

    def freeze(self, now: float | None = None) -> str:
        """Locks in the spoken text at the moment of interruption."""
        self._frozen = self.spoken_text_at(now)
        return self._frozen

    def spoken_text_at(self, now: float | None = None) -> str:
        if self.sentences:
            return " ".join(self.heard_sentences(now))
        return self.full_text[: self.spoken_upto]

    @property
    def spoken_text(self) -> str:
        return self._frozen if self._frozen is not None else self.spoken_text_at()


@dataclass
class BargeInController:
    """One instance per in-flight agent turn. Owns the cancellable generation task and
    the SpokenTracker (TTS flushing itself is done by Pipecat's interruption frame)."""

    tracker: SpokenTracker = field(default_factory=SpokenTracker)
    interrupted: bool = False
    _llm_task: asyncio.Task | None = None
    _tts_task: asyncio.Task | None = None

    def bind_llm_task(self, task: asyncio.Task) -> None:
        self._llm_task = task

    def bind_tts_task(self, task: asyncio.Task) -> None:
        self._tts_task = task

    async def barge_in(self) -> str:
        """Freezes the spoken text, cancels both tasks and waits for them to actually
        finish — no orphan tasks — then returns the rewound spoken_text."""
        self.interrupted = True
        spoken = self.tracker.freeze()
        tasks = [t for t in (self._tts_task, self._llm_task) if t is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return spoken
