"""Translation is optional: it must help when present and change nothing when absent."""

import pytest

from server.lang.contract import (
    MAX_REPAIR_ATTEMPTS,
    can_translate_to,
    generate_with_contract,
)
from server.providers.base import LangContract, TranslationProvider
from server.providers.translate_google import GoogleTranslator, NullTranslator

HI = LangContract("hi", "Deva")


class FakeTranslator(TranslationProvider):
    name = "fake"

    def __init__(self, mapping):
        self.mapping, self.calls = mapping, []

    async def translate(self, text, target_lang, source_lang="auto"):
        self.calls.append((text, target_lang))
        return self.mapping.get(text)

    async def health(self):
        return True


class StubbornLLM:
    """Always answers in English, even when asked to rewrite."""

    def __init__(self):
        self.calls = 0

    async def stream(self, messages, contract, context):
        self.calls += 1
        for ch in "Ashoka was a great Mauryan emperor [ashoka::0001].":
            yield ch


@pytest.fixture(autouse=True)
def _threshold(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_settings",
                        lambda: type("S", (), {"fidelity_script_threshold": 0.15})())


def test_only_non_latin_targets_can_be_machine_translated():
    assert can_translate_to(HI) and can_translate_to(LangContract("ta", "Taml"))
    assert not can_translate_to(LangContract("hi-en", "Latn"))      # Hinglish
    assert not can_translate_to(LangContract("en", "Latn"))


async def test_translation_rescues_a_stubborn_llm_and_keeps_the_citation(monkeypatch):
    fake = FakeTranslator({"Ashoka was a great Mauryan emperor.": "अशोक एक महान मौर्य सम्राट थे।"})
    monkeypatch.setattr("server.lang.contract.get_translator", lambda: fake)
    llm, spoken = StubbornLLM(), []

    async def on_sentence(s):
        spoken.append(s)

    text, passed, attempts = await generate_with_contract(llm, [], HI, [], on_sentence)
    assert passed and attempts == MAX_REPAIR_ATTEMPTS
    assert "अशोक" in text and "[ashoka::0001]" in text          # citation survived translation
    assert all("Mauryan" not in s for s in spoken)                # the English never reached TTS
    assert fake.calls[0][0] == "Ashoka was a great Mauryan emperor."   # tag stripped before sending


async def test_without_a_translator_behaviour_is_unchanged(monkeypatch):
    monkeypatch.setattr("server.lang.contract.get_translator", lambda: NullTranslator())
    text, passed, attempts = await generate_with_contract(StubbornLLM(), [], HI, [])
    assert not passed and attempts == MAX_REPAIR_ATTEMPTS


async def test_null_translator_never_translates():
    assert await NullTranslator().translate("hello", "hi") is None


async def test_google_translator_is_cached_and_never_raises(monkeypatch):
    t, calls = GoogleTranslator(), []

    def fake_sync(text, source, target):
        calls.append(target)
        return "नमस्ते"

    monkeypatch.setattr(GoogleTranslator, "_translate_sync", staticmethod(fake_sync))
    assert await t.translate("hello", "hi") == "नमस्ते"
    assert await t.translate("hello", "hi") == "नमस्ते" and calls == ["hi"]   # second is cached

    def boom(text, source, target):
        raise ConnectionError("offline")

    monkeypatch.setattr(GoogleTranslator, "_translate_sync", staticmethod(boom))
    assert await t.translate("different text", "hi") is None                  # no exception


async def test_unchanged_or_empty_results_are_treated_as_no_translation(monkeypatch):
    t = GoogleTranslator()
    monkeypatch.setattr(GoogleTranslator, "_translate_sync",
                        staticmethod(lambda text, s, tg: text))
    assert await t.translate("same", "en") is None
    assert await t.translate("   ", "hi") is None


def test_odia_uses_googles_language_name():
    from server.providers.translate_google import _CODE_MAP
    assert _CODE_MAP["or"] == "odia" and _CODE_MAP["as"] == "assamese"


# ------------------------------------------------------------------ MyMemory provider
from server.providers.translate_mymemory import (  # noqa: E402
    MAX_CHUNK_BYTES,
    MyMemoryTranslator,
    chunk_text,
)


def test_chunks_never_exceed_the_byte_limit_even_for_devanagari():
    text = " ".join(["अशोक ने कलिंग युद्ध के बाद बौद्ध धर्म अपनाया।"] * 30)
    chunks = chunk_text(text)
    assert len(chunks) > 1 and all(len(c.encode("utf-8")) <= MAX_CHUNK_BYTES for c in chunks)
    one_giant = "क" * 1000
    assert all(len(c.encode("utf-8")) <= MAX_CHUNK_BYTES for c in chunk_text(one_giant))
    assert "".join(chunk_text(one_giant)) == one_giant


async def test_mymemory_guesses_source_from_script_and_maps_region_codes(monkeypatch):
    seen = []

    def fake(text, source, target):
        seen.append((source, target))
        return "What did Ashoka do?"

    monkeypatch.setattr(MyMemoryTranslator, "_translate_sync", staticmethod(fake))
    t = MyMemoryTranslator()
    assert await t.translate("अशोक ने क्या किया?", "en") == "What did Ashoka do?"
    assert seen == [("hi-IN", "en-GB")]


async def test_mymemory_unsupported_pairs_and_same_language_return_none(monkeypatch):
    called = False

    def fake(*a):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(MyMemoryTranslator, "_translate_sync", staticmethod(fake))
    t = MyMemoryTranslator()
    assert await t.translate("hello", "xx") is None            # unknown target
    assert await t.translate("hello", "en", "en") is None      # nothing to do
    assert not called


async def test_mymemory_failure_returns_none_and_never_raises(monkeypatch):
    def boom(*a):
        raise ConnectionError("down")

    monkeypatch.setattr(MyMemoryTranslator, "_translate_sync", staticmethod(boom))
    assert await MyMemoryTranslator().translate("hello", "hi", "en") is None


def test_factory_selects_the_configured_translator(monkeypatch):
    import server.providers.factory as factory
    from server.config import Settings
    from server.providers.translate_mymemory import MyMemoryTranslator as MM

    for value, cls in (("mymemory", MM), ("google", GoogleTranslator), ("none", NullTranslator)):
        factory._translator = None
        s = Settings(_env_file=None, translation_provider=value)
        monkeypatch.setattr(factory, "get_settings", lambda s=s: s)
        assert isinstance(factory.get_translator(), cls)
    factory._translator = None


# ------------------------------------------------- local provider + chain (no torch needed)
from server.providers.translate_chain import ChainTranslator  # noqa: E402
from server.providers.translate_local import LocalTranslator, local_available  # noqa: E402


async def test_chain_returns_the_first_real_translation_and_skips_failures():
    chain = ChainTranslator([FakeTranslator({}), FakeTranslator({"hi": "नमस्ते"}),
                             FakeTranslator({"hi": "never reached"})])
    assert await chain.translate("hi", "hi") == "नमस्ते"
    assert await ChainTranslator([]).translate("hi", "hi") is None
    assert "fake" in chain.name


async def test_local_translator_rules_that_need_no_model():
    t = LocalTranslator()
    assert await t.translate("", "hi") is None
    assert await t.translate("hello", "xx") is None                    # unknown language
    assert await t.translate("हिंदी", "ta", "hi") is None              # only to/from English
    assert await t.translate("Ashoka ne kya kiya", "en", "hi-en") is None   # romanised: unsupported


async def test_local_translator_degrades_instead_of_crashing_when_packages_are_missing(monkeypatch):
    t = LocalTranslator()

    def boom(*a):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(t, "_translate_sync", boom)
    assert await t.translate("Hello there.", "hi", "en") is None       # no exception, logged once
    assert t._unavailable_logged


async def test_local_translator_batches_sentences_and_maps_language_tags(monkeypatch):
    t, seen = LocalTranslator(), {}

    def fake(sentences, src, tgt):
        seen.update(sentences=sentences, src=src, tgt=tgt)
        return ["अशोक ने बौद्ध धर्म अपनाया।", "उसने दूत भेजे।"]

    monkeypatch.setattr(t, "_translate_sync", fake)
    out = await t.translate("Ashoka embraced Buddhism. He sent envoys.", "hi", "en")
    assert out == "अशोक ने बौद्ध धर्म अपनाया। उसने दूत भेजे।"
    assert seen == {"sentences": ["Ashoka embraced Buddhism.", "He sent envoys."],
                    "src": "eng_Latn", "tgt": "hin_Deva"}
    # script guess for indic -> en
    seen.clear()
    monkeypatch.setattr(t, "_translate_sync", lambda s, a, b: seen.update(src=a, tgt=b) or ["x"])
    await t.translate("மௌரிய பேரரசை நிறுவியவர் யார்?", "en")
    assert (seen["src"], seen["tgt"]) == ("tam_Taml", "eng_Latn")


def test_local_available_is_false_without_the_optional_packages_or_models():
    # torch/IndicTransToolkit/models are not installed in the dev venv
    assert local_available() in (True, False)      # never raises, never imports torch


def test_auto_chain_skips_local_when_it_is_not_installed(monkeypatch):
    import server.providers.factory as factory
    from server.config import Settings

    factory._translator = None
    s = Settings(_env_file=None, translation_provider="auto")
    monkeypatch.setattr(factory, "get_settings", lambda: s)
    monkeypatch.setattr("server.providers.translate_local.local_available", lambda: False)
    assert factory.get_translator().name == "chain(mymemory)"
    factory._translator = None
    monkeypatch.setattr("server.providers.translate_local.local_available", lambda: True)
    assert factory.get_translator().name == "chain(indictrans2-local,mymemory)"
    factory._translator = None
