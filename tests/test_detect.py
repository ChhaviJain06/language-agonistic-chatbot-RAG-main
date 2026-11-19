"""hi vs hi-en is decided by script (CLAUDE.md spec). Uses a fake fastText model so the
tests run without the 1 MB model file; scripts/download_models.py fetches the real one."""

from server.lang.detect import (
    LanguageDetector,
    _looks_hinglish,
    majority_script,
    script_for_lang,
)


class FakeFT:
    def __init__(self, label: str, prob: float):
        self._p = (prob, f"__label__{label}")

        class F:
            def predict(_s, line, k, thr, mode):
                return [self._p]

        self.f = F()


def det(label: str, prob: float) -> LanguageDetector:
    d = LanguageDetector(model_path="x")
    d._model = FakeFT(label, prob)
    return d


def test_majority_script_devanagari_latin_and_empty():
    assert majority_script("अशोक ने कौन सा युद्ध लड़ा था?") == "Deva"
    assert majority_script("Which war made Ashoka give up violence?") == "Latn"
    assert majority_script("42!") == "Latn"


def test_urdu_script_is_recognised():
    assert majority_script("یہ اشوک کے بارے میں ہے") == "Arab"


def test_hinglish_markers_exclude_english_homographs():
    assert _looks_hinglish("Ashoka ne kaunsa yudh lada tha?")
    assert not _looks_hinglish("Which war made Ashoka give up violence?")
    assert not _looks_hinglish("The war at Plassey was the turning point")   # 'the' != Hinglish


def test_script_for_lang():
    assert script_for_lang("mr") == "Deva" and script_for_lang("hi-en") == "Latn"
    assert script_for_lang("ta") == "Taml" and script_for_lang("zz") == "Latn"


async def test_devanagari_is_hindi():
    d = await det("hi", 0.97).detect("अशोक ने कौन सा युद्ध लड़ा था?")
    assert (d.lang, d.script) == ("hi", "Deva")


async def test_english_is_english():
    d = await det("en", 0.95).detect("Which war made Ashoka give up violence?")
    assert (d.lang, d.script) == ("en", "Latn")


async def test_romanised_hindi_with_low_confidence_is_hinglish():
    d = await det("la", 0.09).detect("Ashoka ne kaunsa yudh lada tha?")
    assert (d.lang, d.script) == ("hi-en", "Latn")


async def test_hinglish_markers_beat_a_confident_english_guess():
    d = await det("en", 0.8).detect("Ashoka ne Kalinga ke baad kya kiya?")
    assert d.lang == "hi-en"


async def test_missing_model_degrades_to_script_only(tmp_path):
    d = LanguageDetector(model_path=tmp_path / "nope.ftz")
    assert (await d.detect("अशोक कौन थे?")).lang == "hi"
    assert (await d.detect("Ashoka ne kya kiya?")).lang == "hi-en"
    assert (await d.detect("Who was Ashoka?")).lang == "en"
