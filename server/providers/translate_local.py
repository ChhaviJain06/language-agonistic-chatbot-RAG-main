"""Offline translation with AI4Bharat IndicTrans2 (MIT licence) — no API, no rate limits,
no network after the one-time model download. Covers every Indian language this project
supports (Hindi, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi,
Odia, Assamese, Urdu, Nepali, Sanskrit) to and from English.

OPTIONAL dependency: `pip install -e ".[translate]"` (torch + transformers, ~1 GB) and
`python scripts/download_models.py --only translate` (~1.6 GB of models, once). Nothing here
is imported at module load, so a host without these packages simply skips this provider.

Hosting note: the two distilled 200M models need roughly 3 GB of RAM. On a small host leave
TRANSLATION_PROVIDER=auto — it falls back to an online engine, or to no translation, instead
of failing.

STATUS: written against the published IndicTrans2 model-card API but NOT yet run against the
real models on this machine (the packages were not installed). Verify with
`python scripts/check_translation.py` before relying on it.

Limits: IndicTrans2 reads native scripts only. Romanised input (Hinglish) is unsupported and
returns None; the rest of the pipeline already copes with that.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import re
from pathlib import Path

from server.lang.detect import majority_script
from server.providers.base import TranslationProvider

logger = logging.getLogger(__name__)

EN_INDIC = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_EN = "ai4bharat/indictrans2-indic-en-dist-200M"

# our code -> IndicTrans2 FLORES tag
TAGS = {
    "en": "eng_Latn", "hi": "hin_Deva", "mr": "mar_Deva", "ne": "npi_Deva", "sa": "san_Deva",
    "bn": "ben_Beng", "as": "asm_Beng", "ta": "tam_Taml", "te": "tel_Telu", "kn": "kan_Knda",
    "ml": "mal_Mlym", "gu": "guj_Gujr", "pa": "pan_Guru", "or": "ory_Orya", "ur": "urd_Arab",
}
SCRIPT_TO_LANG = {
    "Deva": "hi", "Beng": "bn", "Taml": "ta", "Telu": "te", "Knda": "kn", "Mlym": "ml",
    "Gujr": "gu", "Guru": "pa", "Orya": "or", "Arab": "ur",
}
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")
MAX_SENTENCES = 12
MAX_LENGTH = 200
NUM_BEAMS = 2          # 5 is the paper default; 2 is ~2x faster on CPU with small quality loss


def _hf_cached(repo: str) -> bool:
    root = Path.home() / ".cache" / "huggingface" / "hub"
    return (root / ("models--" + repo.replace("/", "--"))).exists()


def local_available() -> bool:
    """Cheap, import-free check used by the `auto` chain: packages present AND models cached."""
    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
            and importlib.util.find_spec("IndicTransToolkit") is not None
            and _hf_cached(EN_INDIC) and _hf_cached(INDIC_EN))


class LocalTranslator(TranslationProvider):
    name = "indictrans2-local"

    def __init__(self) -> None:
        self._models: dict[str, tuple] = {}          # repo -> (tokenizer, model)
        self._processor = None
        self._lock = asyncio.Lock()                  # one inference at a time; CPU-bound
        self._unavailable_logged = False

    # -- blocking helpers (always run via asyncio.to_thread, C6) ----------------------
    def _load(self, repo: str):
        if repo not in self._models:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                repo, trust_remote_code=True, torch_dtype=torch.float32)
            model.eval()
            self._models[repo] = (tok, model)
        if self._processor is None:
            from IndicTransToolkit.processor import IndicProcessor

            self._processor = IndicProcessor(inference=True)
        return self._models[repo]

    def _translate_sync(self, sentences: list[str], src_tag: str, tgt_tag: str) -> list[str]:
        import torch

        repo = EN_INDIC if src_tag == "eng_Latn" else INDIC_EN
        tok, model = self._load(repo)
        batch = self._processor.preprocess_batch(sentences, src_lang=src_tag, tgt_lang=tgt_tag)
        inputs = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                     return_attention_mask=True)
        with torch.no_grad():
            generated = model.generate(**inputs, use_cache=True, min_length=0,
                                       max_length=MAX_LENGTH, num_beams=NUM_BEAMS,
                                       num_return_sequences=1)
        decoded = tok.batch_decode(generated, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
        return self._processor.postprocess_batch(decoded, lang=tgt_tag)

    # -- public ---------------------------------------------------------------------
    async def translate(self, text: str, target_lang: str,
                        source_lang: str = "auto") -> str | None:
        text = (text or "").strip()
        if not text:
            return None
        if source_lang == "auto":
            script = majority_script(text)
            source_lang = "en" if script == "Latn" else SCRIPT_TO_LANG.get(script, "")
        src, tgt = TAGS.get(source_lang), TAGS.get(target_lang)
        if not src or not tgt or src == tgt:
            return None
        if "eng_Latn" not in (src, tgt):
            return None                  # only to/from English models are loaded (keeps RAM low)
        if source_lang == "hi-en" or (src != "eng_Latn" and majority_script(text) == "Latn"):
            return None                  # romanised input: unsupported
        sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()][:MAX_SENTENCES]
        try:
            async with self._lock:
                out = await asyncio.to_thread(self._translate_sync, sentences, src, tgt)
        except Exception as exc:  # noqa: BLE001 - optional component: report once, then degrade
            if not self._unavailable_logged:
                logger.warning("local translation unavailable: %r (see translate_local.py)", exc)
                self._unavailable_logged = True
            return None
        result = " ".join(o.strip() for o in out if o).strip()
        return result if result and result.lower() != text.lower() else None

    async def health(self) -> bool:
        return local_available() and await self.translate("hello", "hi", "en") is not None
