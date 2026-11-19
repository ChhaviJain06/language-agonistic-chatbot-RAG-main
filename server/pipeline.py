"""Pipecat pipeline assembly: LiveKit transport -> Silero VAD -> our ASR router ->
RAG + fidelity-contract "brain" -> our TTS -> LiveKit output.

Barge-in (the hard part)
------------------------
A turn stays *active* from the moment a transcript arrives until the bot has actually
finished speaking it — not merely until generation ends (generation finishes in a few
hundred ms; speech lasts many seconds, which is when users interrupt). While a turn is
active, VADUserStartedSpeakingFrame triggers a real barge-in:

  1. cancel the generation task (no more sentences are produced),
  2. broadcast Pipecat's interruption so BhashaTTS and the output transport flush audio,
  3. rewind conversation history to what was HEARD, measured from the audio durations
     BhashaTTS reports per sentence and the bot-speaking frames from the transport
     (barge_in.SpokenTracker),
  4. tell the client (`barge_in` event) so the UI can show it.

Not run end-to-end in this repository's CI: that needs a live LiveKit room and real audio.
The unit tests drive BhashaBrain with real Pipecat frames and stubbed I/O.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputTransportMessageUrgentFrame,
    SystemFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.livekit.transport import (
    LiveKitInputTransportMessageFrame,
    LiveKitParams,
    LiveKitTransport,
)

from server import voice_replay
from server.barge_in import BargeInController
from server.config import get_settings
from server.lang.contract import build_contract
from server.pipeline_text import clean_tts_text
from server.providers._pcm import rms_pcm16, wav_bytes_to_pcm
from server.providers.asr_router import route_transcribe
from server.providers.factory import get_tts
from server.rag import store
from server.rag.answer import answer_question, maybe_cache
from server.rag.question_type import detect_question_type  # noqa: F401  (re-exported)
from server.session import SessionState, apology_for

logger = logging.getLogger(__name__)

MIN_SEGMENT_RMS = 250.0   # ~ -42 dBFS: quieter than this is room noise, not speech
# Whisper's classic silence hallucinations — never worth sending to the LLM.
_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching",
                   "you", "bye", "okay", "subtitles by the amara.org community"}


@dataclass
class SentenceTimingFrame(SystemFrame):
    """Pushed UPSTREAM by BhashaTTS: how many seconds of audio a sentence became. This is
    what lets the brain work out how much of an answer was actually heard."""

    text: str = ""
    seconds: float = 0.0


@dataclass
class BhashaSpeakFrame(TTSSpeakFrame):
    """TTSSpeakFrame doesn't carry a language — the contract's target lang has to ride
    along so BhashaTTS knows which voice to use."""

    lang: str = "en"


class BhashaSTT(SegmentedSTTService):
    """Wraps asr_router.route_transcribe(). VADProcessor upstream segments the audio; each
    segment arrives here as a WAV-wrapped buffer."""

    def __init__(self, state: SessionState | None = None, **kwargs):
        # Pipecat 1.16 wants every settings field initialised (None = "not applicable"): our
        # engines are chosen by asr_router, not by a model/language set on the service.
        kwargs.setdefault("settings", STTSettings(model=None, language=None))
        kwargs.setdefault("ttfs_p99_latency", 1.0)
        super().__init__(**kwargs)
        self._state = state or SessionState(room="")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        pcm, sample_rate, _channels = wav_bytes_to_pcm(audio)
        if rms_pcm16(pcm) < MIN_SEGMENT_RMS:
            logger.debug("dropping near-silent segment")
            return
        transcript = await route_transcribe(pcm, sample_rate, hint_lang=self._state.last_lang)
        self._state.last_asr_ms = transcript.latency_ms
        text = transcript.text.strip()
        if not text or text.lower().strip(" .!") in _HALLUCINATIONS:
            return
        yield TranscriptionFrame(text=text, user_id="", timestamp=str(time.time()),
                                 language=None)


@dataclass
class _Turn:
    id: str
    question: str
    controller: BargeInController
    task: asyncio.Task | None = None
    generation_done: bool = False
    committed: bool = False
    answer: str = ""
    lang: str = "en"
    script: str = "Latn"
    sources: list[dict] | None = None
    timing: dict | None = None
    started_at: float = 0.0          # monotonic time the transcript arrived
    asr_ms: float | None = None
    first_audio_ms: float | None = None   # transcript -> first audio actually playing


class BhashaBrain(FrameProcessor):
    """RAG + Language Fidelity Contract + barge-in. Forwards every frame unchanged."""

    def __init__(self, state: SessionState | None = None, **kwargs):
        super().__init__(**kwargs)
        self.state = state or SessionState(room="")
        self._turn: _Turn | None = None
        self._bg: set[asyncio.Task] = set()
        # Back-compat handles used by tests/tools.
        self._turn_task: asyncio.Task | None = None

    # -- helpers ---------------------------------------------------------------
    async def _send(self, payload: dict) -> None:
        """One data-channel message to the browser. Only the urgent frame is used —
        sending both used to deliver every message twice."""
        await self.push_frame(OutputTransportMessageUrgentFrame(message=json.dumps(payload)),
                              FrameDirection.DOWNSTREAM)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    def _commit(self, turn: _Turn, heard: str | None) -> None:
        """Writes the turn into history using what the user HEARD, and persists it."""
        if turn.committed:
            return
        turn.committed = True
        self.state.remember(turn.question, heard)
        if heard is not None or turn.answer:
            self._spawn(store.log_exchange(
                self.state.conversation_id, question=turn.question, answer=turn.answer,
                spoken_text=heard, lang=turn.lang, script=turn.script,
                cited_chunk_ids=[s["chunk_id"] for s in (turn.sources or [])],
                timing={**(turn.timing or {}), "asr_ms": turn.asr_ms,
                        "first_audio_ms": turn.first_audio_ms}))

    @staticmethod
    def _log_first_audio(turn: _Turn) -> None:
        """Real, measured latency of this turn. `end of speech -> audio` also includes the
        VAD silence wait (VAD_SILENCE_MS) and, for voice input, the ASR call."""
        turn.first_audio_ms = round((time.monotonic() - turn.started_at) * 1000, 1)
        parts = {"asr_ms": turn.asr_ms, "first_audio_ms": turn.first_audio_ms}
        parts.update({k: v for k, v in (turn.timing or {}).items()})
        logger.info("latency %s", {k: v for k, v in parts.items() if v is not None})

    def _active(self) -> bool:
        return self._turn is not None and not self._turn.committed

    # -- frame handling ----------------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        turn = self._turn

        if isinstance(frame, BotStartedSpeakingFrame) and turn:
            turn.controller.tracker.start_playback()
            if turn.first_audio_ms is None:
                self._log_first_audio(turn)
            # The server, not the audio level, is the source of truth for "speaking": TTS
            # chunks have gaps, and the level-based UI used to flip to "listening" in them.
            await self._send({"event": "status", "status": "speaking", "msg_id": turn.id})
        elif isinstance(frame, BotStoppedSpeakingFrame) and turn:
            tracker = turn.controller.tracker
            tracker.pause_playback()
            if turn.generation_done and tracker.fully_played():
                self._commit(turn, turn.answer)        # everything was heard
                self._turn = None
                await self._send({"event": "status", "status": "listening", "msg_id": turn.id})
        elif isinstance(frame, SentenceTimingFrame) and turn:
            turn.controller.tracker.register_sentence(frame.text, frame.seconds)
        elif isinstance(frame, VADUserStartedSpeakingFrame) and self._active():
            await self._barge_in()

        await self.push_frame(frame, direction)

        query = self._extract_query(frame)
        if query:
            await self._start_turn(query)

    @staticmethod
    def _extract_query(frame: Frame) -> str | None:
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            return frame.text.strip()
        if isinstance(frame, LiveKitInputTransportMessageFrame):
            msg = frame.message
            if isinstance(msg, dict) and str(msg.get("text", "")).strip():
                return str(msg["text"]).strip()
        return None

    async def _barge_in(self) -> None:
        turn = self._turn
        assert turn is not None
        spoken = await turn.controller.barge_in()      # cancels generation, freezes the heard text
        await self.broadcast_interruption()            # flushes TTS + transport audio
        self._commit(turn, spoken)
        await self._send({"event": "barge_in", "msg_id": turn.id, "spoken_text": spoken})
        self._turn = None
        self._turn_task = None

    async def _start_turn(self, question: str) -> None:
        # A new question while the previous answer is still playing counts as an interruption.
        if self._active():
            prev = self._turn
            assert prev is not None
            spoken = await prev.controller.barge_in()
            await self.broadcast_interruption()
            self._commit(prev, spoken)

        turn = _Turn(id=uuid.uuid4().hex[:10], question=question, controller=BargeInController(),
                     started_at=time.monotonic(), asr_ms=self.state.last_asr_ms)
        self.state.last_asr_ms = None            # text messages have no ASR step
        self._turn = turn
        turn.task = asyncio.create_task(self._run_turn_safe(turn))
        self._turn_task = turn.task
        turn.controller.bind_llm_task(turn.task)

    # -- one turn ----------------------------------------------------------------
    async def _run_turn_safe(self, turn: _Turn) -> None:
        try:
            await self._run_turn(turn)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never leave the caller in silence
            logger.exception("turn failed")
            await self._fail_turn(turn, exc)

    async def _fail_turn(self, turn: _Turn, exc: Exception) -> None:
        lang = turn.lang
        message = apology_for(lang)
        await self._send({"event": "error", "msg_id": turn.id, "error": message,
                          "detail": type(exc).__name__})
        await self.push_frame(BhashaSpeakFrame(text=message, lang=lang), FrameDirection.DOWNSTREAM)
        turn.answer = message
        turn.generation_done = True

    async def _run_turn(self, turn: _Turn) -> None:
        question = turn.question
        await self._send({"event": "status", "status": "thinking", "msg_id": turn.id,
                          "question": question})

        contract = await build_contract(question)
        turn.lang, turn.script = contract.lang, contract.script
        self.state.last_lang = contract.lang            # routing hint for the next utterance

        async def on_sentence(sent: str) -> None:
            clean = clean_tts_text(sent)
            if clean:
                turn.controller.tracker.append_generated(clean + " ")
                await self.push_frame(BhashaSpeakFrame(text=clean, lang=contract.lang),
                                      FrameDirection.DOWNSTREAM)

        history = list(self.state.history)
        result = await answer_question(question, history=history, contract=contract,
                                       on_sentence=on_sentence)
        turn.answer = clean_tts_text(result.text)
        turn.sources, turn.timing = result.sources, result.timing
        logger.info("latency (answer ready) %s", result.timing)

        payload = result.to_payload()
        payload.update({"event": "answer", "msg_id": turn.id, "transcript": question})
        # Recorded BEFORE sending so a browser that dropped mid-turn can still fetch it.
        voice_replay.record(self.state.room, payload)
        await self._send(payload)
        self._spawn(maybe_cache(result, history))
        turn.generation_done = True


class BhashaTTS(TTSService):
    """Wraps providers/factory.get_tts(). Reads the target language off BhashaSpeakFrame
    (stashed in process_frame, since TTSService.run_tts's signature is fixed to
    (text, context_id) — it doesn't get the frame)."""

    def __init__(self, **kwargs):
        provider = get_tts()
        kwargs.setdefault("settings", TTSSettings(model=None, voice=None, language=None))
        super().__init__(sample_rate=provider.sample_rate, **kwargs)
        self._provider = provider
        self._sample_rate = provider.sample_rate
        self._pending_lang = "en"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, BhashaSpeakFrame):
            self._pending_lang = frame.lang
        await super().process_frame(frame, direction)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        total_bytes = 0
        async for pcm_chunk in self._provider.synthesize(text, self._pending_lang):
            total_bytes += len(pcm_chunk)
            yield TTSAudioRawFrame(
                audio=pcm_chunk, sample_rate=self._provider.sample_rate, num_channels=1)
        seconds = total_bytes / (2 * self._provider.sample_rate)   # 16-bit mono
        await self.push_frame(SentenceTimingFrame(text=text, seconds=seconds),
                              FrameDirection.UPSTREAM)


def build_pipeline(room_name: str, livekit_token: str,
                   state: SessionState | None = None) -> tuple[PipelineTask, asyncio.Event]:
    s = get_settings()
    state = state or SessionState(room=room_name)
    tts = BhashaTTS()
    transport = LiveKitTransport(
        url=s.livekit_url, token=livekit_token, room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True,
                             audio_out_sample_rate=tts.sample_rate),
    )
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        params=VADParams(stop_secs=s.vad_silence_ms / 1000, min_volume=0.05)))
    pipeline = Pipeline([
        transport.input(), vad, BhashaSTT(state), BhashaBrain(state), tts, transport.output(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams())
    joined = asyncio.Event()

    @transport.event_handler("on_first_participant_joined")
    async def _on_first_joined(transport, participant_id: str):
        joined.set()

    grace: dict[str, asyncio.Task] = {}

    async def _cancel_after_grace() -> None:
        # A network blip must not kill an answer that is being generated: the turn keeps
        # running (and lands in voice_replay) for a while, then the agent is freed.
        await asyncio.sleep(s.voice_reconnect_grace_s)
        logger.info("room %s: browser did not return within %ss — stopping agent",
                    room_name, s.voice_reconnect_grace_s)
        await task.cancel()

    @transport.event_handler("on_participant_disconnected")
    async def _on_participant_disconnected(transport, participant_id: str):
        if "g" not in grace or grace["g"].done():
            grace["g"] = asyncio.create_task(_cancel_after_grace())

    @transport.event_handler("on_participant_connected")
    async def _on_participant_connected(transport, participant_id: str):
        pending = grace.pop("g", None)
        if pending and not pending.done():
            pending.cancel()

    return task, joined


async def run_pipeline(room_name: str, livekit_token: str) -> None:
    s = get_settings()
    state = SessionState(room=room_name)
    state.conversation_id = await store.start_conversation(room_name)
    task, joined = build_pipeline(room_name, livekit_token, state)

    async def watchdog() -> None:
        """Nobody joined? Free the agent instead of leaking it (and its API quota)."""
        try:
            await asyncio.wait_for(joined.wait(), s.agent_join_timeout_s)
        except TimeoutError:
            logger.info("no participant joined room %s in %ss — stopping agent",
                        room_name, s.agent_join_timeout_s)
            await task.cancel()

    guard = asyncio.create_task(watchdog())
    try:
        await PipelineRunner().run(task)
    finally:
        guard.cancel()
        await store.end_conversation(state.conversation_id)
