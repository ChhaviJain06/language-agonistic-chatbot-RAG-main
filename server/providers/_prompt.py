"""System prompt shared by every LLM provider, so Groq and Ollama behave identically."""

from __future__ import annotations

from server.providers.base import LangContract

_LANG_NAMES = {
    "en": "English", "hi": "Hindi", "hi-en": "Hinglish (Hindi written in the Latin script)",
    "mr": "Marathi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
    "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "ur": "Urdu",
}


def build_system_prompt(contract: LangContract | None, context: list[dict]) -> str:
    lines: list[str] = []
    if contract is not None:
        name = _LANG_NAMES.get(contract.lang, contract.lang)
        lines.append(
            f"Reply ONLY in {name} (language code '{contract.lang}'), written in the "
            f"'{contract.script}' script, register '{contract.register}'. "
            "Never switch language or script mid-reply, even when the sources are in English; "
            "translate facts into the reply language but keep proper names recognisable."
        )
    if context:
        lines.append(
            "Cite ONLY sources listed below, by their exact id in square brackets, e.g. "
            "[ashoka::0007]. Never cite 'graph'. Never invent an id. If the sources do not "
            "answer the question, say you do not know instead of guessing."
        )
        for c in context:
            lines.append(f"[{c['chunk_id']}] {c['text']}")
    return "\n".join(lines)
