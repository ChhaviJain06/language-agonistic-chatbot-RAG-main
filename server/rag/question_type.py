"""Heuristic question-type routing (C12): decide whether graph traversal is worth it.

Whole-word matching only. The old substring version fired on "son" inside "reason"
and "war" inside "Swaraj", sending nearly every question to Neo4j.
"""

from __future__ import annotations

import re

_COMPARATIVE_LATIN = re.compile(
    r"\b(compare|comparison|compared|difference|differences|differ|differs|versus|vs|"
    r"contrast|similarities|similarity|antar|tulna|farak|fark)\b", re.IGNORECASE)

_MULTI_HOP_LATIN = re.compile(
    r"\b(father|mother|sons?|daughters?|grandfather|grandmother|grandson|granddaughter|"
    r"successors?|succeeded|predecessors?|descendants?|ancestors?|lineage|dynasty|"
    r"imprisoned|overthrew|"
    r"(emperor|king|ruler|founder|queen|general)\s+who|"
    r"baap|pita|beta|betee|beti|pota|potaa|dada|dadaji|nana|vansh|parivar|"
    r"kiske\s+baad|kiske\s+pehle|kiska\s+beta|kiski\s+beti)\b", re.IGNORECASE)

# Non-Latin scripts have no \b word boundaries worth trusting; substring is fine here.
_NON_LATIN_COMPARATIVE = ("तुलना", "अंतर", "फर्क", "बनाम", "তুলনা", "পার্থক্য",
                          "ஒப்பிடு", "வேறுபாடு", "तुलना", "फरक")
_NON_LATIN_MULTI_HOP = ("पिता", "पुत्र", "बेटा", "बेटी", "दादा", "वंश", "उत्तराधिकारी",
                        "पूर्वज", "परिवार", "वडील", "मुलगा", "বাবা", "ছেলে", "বংশ",
                        "தந்தை", "மகன்", "வம்சம்")


def detect_question_type(question: str) -> str:
    """Returns "comparative", "multi_hop" or "factoid"."""
    if _COMPARATIVE_LATIN.search(question) or any(w in question for w in _NON_LATIN_COMPARATIVE):
        return "comparative"
    if _MULTI_HOP_LATIN.search(question) or any(w in question for w in _NON_LATIN_MULTI_HOP):
        return "multi_hop"
    return "factoid"
