"""The Language Fidelity Contract: detect -> constrain -> validate -> repair.

Constraining generation via the system prompt does NOT guarantee compliance —
models drift, especially mid-answer when citing English source text. So every
SENTENCE is validated before it is released to TTS / the client. When a sentence
drifts, it and everything after it are held back and rewritten (bounded, never an
unbounded loop — C6); sentences already released were valid and stay as spoken.

This is what makes the contract real for voice: the user never hears a wrong-script
sentence, because release happens only after validation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from server.config import get_settings
from server.lang.detect import get_detector, script_for_lang, script_histogram
from server.lang.normalize import normalize_typography
from server.providers.base import LangContract, LLMProvider

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 2
# A ~45-char chunk with one Latin name ("शिवाजी ने Maratha साम्राज्य") is 27% foreign; a chunk truly
# in the wrong script is ~100%. The strict whole-answer threshold stays in settings.
CHUNK_SCRIPT_THRESHOLD = 0.5
MIN_LETTERS_TO_JUDGE = 4   # "Yes." or "1757" says nothing about script drift
_CITATION_RE = re.compile(r"\[[a-zA-Z0-9_:\.\s,\-]+\]|【[a-zA-Z0-9_:\.\s,\-]+】")
_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")
_CLAUSE_RE = re.compile(r"(?<=[,;:])\s+")

# Speech is released in chunks, not only whole sentences. Measured on Sarvam TTS, synthesis
# takes ~0.3 s + ~0.02 s per character (20 chars 0.45 s, 45 chars 0.7 s, 65 chars 1.2 s,
# 100 chars 2.0 s) while the audio itself lasts ~0.075 s per character — so synthesis is
# faster than playback, and only the FIRST chunk's length decides when audio starts.
# Hence: a short first chunk (<= ~45 chars), then chunks of up to a sentence / ~110 chars.
FIRST_CHUNK_MIN_CHARS = 30        # a clause boundary (", ; :") is accepted from here...
FIRST_CHUNK_MAX_CHARS = 45        # ...and a word boundary is forced by here
LATER_CHUNK_MIN_CHARS = 80
LATER_CHUNK_MAX_CHARS = 110

# Where a forced cut sounds least awkward: right before a connective or preposition.
_BREAK_BEFORE = frozenset(
    "and but or who which that where when while by in of on at for with from to under after "
    "before during through into as because so then also".split())


def _word_cut(buffer: str, min_len: int, max_len: int) -> tuple[int, int] | None:
    """A forced cut at a space within [min_len, max_len], or None if the text hasn't grown
    far enough. Prefers a space right before a connective; otherwise the last space."""
    if len(buffer) <= max_len + 1:
        return None
    spaces = [i for i in range(min_len, min(max_len, len(buffer) - 1) + 1) if buffer[i] == " "]
    if not spaces:
        return None
    for i in reversed(spaces):
        nxt = buffer[i + 1:].split(" ", 1)[0].lower().strip(",.;:()[]")
        if nxt in _BREAK_BEFORE:
            return i, i + 1
    return spaces[-1], spaces[-1] + 1


def next_cut(buffer: str, first: bool) -> tuple[int, int] | None:
    """(end_of_chunk, start_of_rest) for the first releasable chunk in `buffer`, or None.

    A sentence ending always cuts. A clause boundary cuts once the chunk is long enough, and a
    word boundary is forced once it is too long. Never cuts inside a token ("1,000", "[1]")."""
    sentence = _SPLIT_RE.search(buffer)
    min_len = FIRST_CHUNK_MIN_CHARS if first else LATER_CHUNK_MIN_CHARS
    max_len = FIRST_CHUNK_MAX_CHARS if first else LATER_CHUNK_MAX_CHARS
    for m in _CLAUSE_RE.finditer(buffer):
        if min_len <= m.start() <= max_len and (sentence is None or m.start() < sentence.start()):
            return m.start(), m.end()
    if sentence and sentence.start() <= max_len + 1:
        return sentence.start(), sentence.end()
    forced = _word_cut(buffer, min_len, max_len)
    if forced and (sentence is None or forced[0] < sentence.start()):
        return forced
    return (sentence.start(), sentence.end()) if sentence else None


def chunk_text(text: str) -> list[str]:
    """Splits finished text into speech chunks exactly like live generation does, so a cached
    answer also starts with a short first chunk instead of one long sentence."""
    chunks: list[str] = []
    buffer = text.strip()
    cut = next_cut(buffer, first=True)
    while cut:
        piece = buffer[:cut[0]].strip()
        if piece:
            chunks.append(piece)
        buffer = buffer[cut[1]:]
        cut = next_cut(buffer, first=not chunks)
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks


async def build_contract(user_text: str, register: str = "neutral",
                         forced_lang: str | None = None) -> LangContract:
    """detect + constrain: the reply must mirror the user's language/script.
    `forced_lang` (UI language selector) skips detection but still maps to the right script."""
    if forced_lang:
        return LangContract(lang=forced_lang, script=script_for_lang(forced_lang),
                            register=register)
    det = await get_detector().detect(user_text)
    return LangContract(lang=det.lang, script=det.script, register=register)


def foreign_script_ratio(text: str, contract: LangContract) -> float:
    """Fraction of the reply's lettered characters NOT in the contract's script.

    Citations (e.g. [ashoka::0001]) are stripped first so their Latin chunk ids do not
    falsely penalise Hindi/Devanagari answers."""
    clean_text = _CITATION_RE.sub("", text)
    hist = script_histogram(clean_text)
    total = sum(hist.values())
    if total == 0:
        return 0.0
    foreign = sum(n for script, n in hist.items() if script != contract.script)
    return foreign / total


def validate(text: str, contract: LangContract) -> bool:
    return foreign_script_ratio(text, contract) <= get_settings().fidelity_script_threshold


def sentence_ok(sentence: str, contract: LangContract) -> bool:
    """Per-sentence check. Very short sentences are never judged."""
    clean = _CITATION_RE.sub("", sentence)
    if sum(script_histogram(clean).values()) < MIN_LETTERS_TO_JUDGE:
        return True
    return foreign_script_ratio(sentence, contract) <= max(
        CHUNK_SCRIPT_THRESHOLD, get_settings().fidelity_script_threshold)


async def _rewrite(bad_text: str, contract: LangContract, llm: LLMProvider) -> str:
    """One bounded re-prompt that rewrites ONLY the drifted text, keeping citation tags."""
    prompt = {
        "role": "user",
        "content": (
            f"Rewrite the text below entirely in language '{contract.lang}', using ONLY the "
            f"'{contract.script}' script. Keep any [chunk_id] citation tags exactly as they "
            f"are. Output only the rewritten text, nothing else.\n\n{bad_text}"
        ),
    }
    chunks: list[str] = []
    async for tok in llm.stream([prompt], contract, []):
        chunks.append(tok)
    return "".join(chunks).strip()


async def generate_with_contract(
    llm: LLMProvider,
    messages: list[dict],
    contract: LangContract,
    context: list[dict],
    on_sentence: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, bool, int]:
    """Streams a reply, releasing each validated sentence through `on_sentence`.

    Returns (final_text, passed_validation, repair_attempts_used).
    """
    released: list[str] = []      # sentences that passed and were handed out
    held: list[str] = []          # drifted sentence + everything generated after it
    buffer = ""

    async def release(sent: str) -> None:
        sent = normalize_typography(sent)
        if not sent:
            return
        if held:                       # once one sentence drifted, hold everything after it
            held.append(sent)
            return
        if sentence_ok(sent, contract):
            released.append(sent)
            if on_sentence is not None:
                await on_sentence(sent)
        else:
            held.append(sent)

    async for tok in llm.stream(messages, contract, context):
        buffer += tok
        cut = next_cut(buffer, first=not released and not held)
        while cut:
            sent = buffer[:cut[0]].strip()
            buffer = buffer[cut[1]:]
            await release(sent)
            cut = next_cut(buffer, first=not released and not held)
    await release(buffer.strip())

    attempts = 0
    if held:
        bad = " ".join(held)
        best, best_ratio = bad, foreign_script_ratio(bad, contract)
        while best_ratio > get_settings().fidelity_script_threshold and \
                attempts < MAX_REPAIR_ATTEMPTS:
            attempts += 1
            candidate = await _rewrite(best, contract, llm)
            ratio = foreign_script_ratio(candidate, contract) if candidate else 1.0
            logger.info("fidelity repair %d: ratio %.2f -> %.2f", attempts, best_ratio, ratio)
            if ratio < best_ratio:
                best, best_ratio = candidate, ratio
        for sent in chunk_text(best):
            released.append(sent)
            if on_sentence is not None:
                await on_sentence(sent)

    text = " ".join(released)
    return text, validate(text, contract), attempts
