"""Typographic clean-up of model output.

Reasoning models like to emit U+202F (narrow no-break space) inside names ("Taj Mahal",
"Shah Jahan") and U+2011 (non-breaking hyphen). Most UI fonts draw the former almost
invisibly, so words run together on screen, and TTS engines can read the latter oddly.
Everything is mapped to its plain ASCII equivalent; letters are never touched.
"""

from __future__ import annotations

import re

_SPACES_RE = re.compile("[       　]")
_HYPHENS_RE = re.compile("[‐‑‒]")
_INVISIBLE_RE = re.compile("[​⁠﻿]")      # zero-width: no visual or audible meaning


def normalize_typography(text: str) -> str:
    text = _INVISIBLE_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    return _HYPHENS_RE.sub("-", text)
