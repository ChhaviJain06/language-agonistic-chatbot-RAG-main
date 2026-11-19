"""Shared PCM->WAV wrapping. ASRProvider.transcribe() takes raw PCM bytes
(§4), but every cloud ASR REST API wants an actual audio file with headers —
stdlib `wave` builds that in memory, no extra dependency for one struct write.
"""

from __future__ import annotations

import io
import wave


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1,
                     sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_bytes_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Strips a WAV container down to raw PCM — TTSProvider.synthesize() (§4)
    yields PCM chunks, not audio files, so a provider whose API returns WAV
    (e.g. Sarvam) unwraps it here rather than leaking the container format."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()


def rms_pcm16(pcm: bytes) -> float:
    """Root-mean-square level of 16-bit mono PCM, 0..32768. Used to drop near-silent
    segments before ASR, because Whisper invents text for silence."""
    import array
    import math

    if len(pcm) < 2:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return math.sqrt(sum(s * s for s in samples) / len(samples))
