"""The fidelity contract must be exactly right: it is checked per SENTENCE, before
anything is released to TTS."""

import pytest

from server.lang.contract import (
    MAX_REPAIR_ATTEMPTS,
    foreign_script_ratio,
    generate_with_contract,
    sentence_ok,
    validate,
)
from server.providers.base import LangContract

EN = LangContract(lang="en", script="Latn")
HI = LangContract(lang="hi", script="Deva")


class FakeLLM:
    """Yields `replies` in order, one per stream() call, a character at a time."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    async def stream(self, messages, contract, context):
        self.calls += 1
        for ch in self._replies.pop(0):
            yield ch


@pytest.fixture(autouse=True)
def _threshold(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.15})())


def test_ratio_pure_match_is_zero():
    assert foreign_script_ratio("Ashoka ruled the Maurya empire.", EN) == 0.0


def test_ratio_full_mismatch_is_one():
    assert foreign_script_ratio("Ashoka ruled the empire", HI) == 1.0


def test_ratio_ignores_latin_citation_tags():
    text = "मौर्य साम्राज्य की स्थापना चन्द्रगुप्त मौर्य ने की थी [maurya_empire::0000]"
    assert foreign_script_ratio(text, HI) == 0.0


def test_short_sentences_are_never_judged():
    assert sentence_ok("Yes.", HI) and sentence_ok("1757.", HI)


def test_urdu_script_counts_as_foreign_for_hindi():
    assert not validate("یہ اشوک کے بارے میں ہے", HI)


async def test_passes_first_try_and_streams_sentence_by_sentence():
    llm, got = FakeLLM(["Ashoka ruled well. He built stupas."]), []

    async def on_sentence(s):
        got.append(s)

    text, passed, attempts = await generate_with_contract(llm, [], EN, [], on_sentence)
    assert got == ["Ashoka ruled well.", "He built stupas."]
    assert passed and attempts == 0 and llm.calls == 1


async def test_wrong_script_sentence_is_never_released_and_is_rewritten():
    """The user must not HEAR the Devanagari sentence."""
    llm = FakeLLM(["अशोक ने राज किया। उसने स्तूप बनवाए।", "Ashoka ruled. He built stupas."])
    got = []

    async def on_sentence(s):
        got.append(s)

    text, passed, attempts = await generate_with_contract(llm, [], EN, [], on_sentence)
    assert all("अ" not in s for s in got)
    assert got == ["Ashoka ruled.", "He built stupas."]
    assert passed and attempts == 1 and text == "Ashoka ruled. He built stupas."


async def test_valid_prefix_is_kept_and_only_the_drifted_tail_is_rewritten():
    llm = FakeLLM(["Ashoka ruled well. अशोक ने स्तूप बनवाए।", "He built stupas."])
    got = []

    async def on_sentence(s):
        got.append(s)

    text, passed, attempts = await generate_with_contract(llm, [], EN, [], on_sentence)
    assert got == ["Ashoka ruled well.", "He built stupas."]
    assert text == "Ashoka ruled well. He built stupas." and passed and attempts == 1


async def test_gives_up_after_max_attempts_without_looping_forever():
    llm = FakeLLM(["अशोक एक महान राजा थे।"] * (MAX_REPAIR_ATTEMPTS + 1))
    text, passed, attempts = await generate_with_contract(llm, [], EN, [])
    assert not passed and attempts == MAX_REPAIR_ATTEMPTS
    assert llm.calls == MAX_REPAIR_ATTEMPTS + 1


# ------------------------------------------------------------------ early chunk release
def test_first_chunk_is_short_and_cut_before_a_connective():
    from server.lang.contract import FIRST_CHUNK_MAX_CHARS, next_cut

    text = "The Taj Mahal was commissioned and built for Emperor Shah Jahan, who directed it"
    end, rest = next_cut(text, first=True)
    assert text[:end] == "The Taj Mahal was commissioned and built"      # cut before "for"
    assert end <= FIRST_CHUNK_MAX_CHARS and text[rest:].startswith("for Emperor")


def test_a_comma_inside_the_first_chunk_window_is_preferred():
    from server.lang.contract import next_cut

    text = "The Delhi Sultanate was founded in 1206, and it lasted until 1526"
    end, rest = next_cut(text, first=True)
    assert text[:end] == "The Delhi Sultanate was founded in 1206,"
    assert text[rest:].startswith("and")


def test_nothing_is_cut_until_enough_text_has_arrived_and_numbers_are_safe():
    from server.lang.contract import next_cut

    assert next_cut("The Qutub Minar was begun und", first=True) is None   # still streaming
    assert next_cut("Short, clause", first=True) is None
    text = "Founded in 1,000 BCE by the kings of the north and more"
    end, _ = next_cut(text, first=True)
    assert not text[:end].endswith("1,") and text[end] == " "


def test_a_sentence_end_always_cuts_and_later_chunks_are_longer():
    from server.lang.contract import next_cut

    assert next_cut("It was big. Then", first=False) == (11, 12)
    text = "A clause that is forty or so characters long, then more words follow here"
    assert next_cut(text, first=True) is not None      # first chunk: cut at the comma
    assert next_cut(text, first=False) is None         # later chunks may run longer


async def test_first_chunk_reaches_tts_before_the_sentence_finishes():
    reply = ("The Taj Mahal was commissioned and built for Emperor Shah Jahan, who directed its "
             "construction through a board of architects.")
    llm, got = FakeLLM([reply]), []

    async def on_sentence(s):
        got.append(s)

    text, passed, _ = await generate_with_contract(llm, [], EN, [], on_sentence)
    assert len(got) >= 2 and len(got[0]) <= 45         # a short first chunk goes out first
    assert " ".join(got) == reply and text == reply and passed


async def test_a_wrong_script_clause_is_still_held_back():
    bad = "अशोक ने कलिंग का युद्ध लड़ा था और बहुत दुखी हुए, फिर बौद्ध बने।"
    llm = FakeLLM([bad, "Ashoka fought, then converted."])
    got = []

    async def on_sentence(s):
        got.append(s)

    await generate_with_contract(llm, [], EN, [], on_sentence)
    assert all("अ" not in s and "ब" not in s for s in got)


def test_chunk_text_splits_cached_answers_like_live_generation():
    from server.lang.contract import chunk_text

    text = ("The Taj Mahal was commissioned and built for Emperor Shah Jahan, who directed its "
            "construction. After he was deposed, his son Aurangzeb imprisoned him at Agra Fort.")
    chunks = chunk_text(text)
    assert len(chunks[0]) <= 45 and len(chunks) >= 3      # short first chunk, then the rest
    assert " ".join(chunks) == text                       # nothing lost or reordered
    assert chunk_text("") == [] and chunk_text("Hi.") == ["Hi."]


async def test_exotic_spaces_and_hyphens_are_normalised_before_release():
    llm, got = FakeLLM(["The Taj Mahal was built by Shah Jahan in a non‑violent age."]), []

    async def on_sentence(s):
        got.append(s)

    text, _, _ = await generate_with_contract(llm, [], EN, [], on_sentence)
    assert text == "The Taj Mahal was built by Shah Jahan in a non-violent age."
    assert all(" " not in s and " " not in s and "‑" not in s for s in got)


def test_normalize_typography_never_touches_letters():
    from server.lang.normalize import normalize_typography

    assert normalize_typography("अशोक ने  [1]​।") == "अशोक ने  [1]।"
    assert normalize_typography("Qutb‑ud‑Din") == "Qutb-ud-Din"


def test_hindi_chunk_with_one_latin_name_is_not_flagged():
    from server.lang.contract import sentence_ok
    from server.providers.base import LangContract

    hi = LangContract(lang="hi", script="Deva")
    assert sentence_ok("कलिंग युद्ध के बाद Ashoka ने", hi)
    assert not sentence_ok("After the Kalinga war Ashoka turned", hi)   # truly wrong script
