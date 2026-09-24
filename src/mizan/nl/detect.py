"""Script/language detection for incoming questions.

Why not a statistical language detector
---------------------------------------
``langdetect``/``fasttext`` would add a dependency and a model file to answer a question we
can answer exactly. We do not need to distinguish Arabic from Farsi from Urdu — we need to
know which *prompt template* and which *glossary direction* to use, and that is decided by
script, which is a deterministic property of the code points. A rule you can read is worth
more here than a classifier you cannot explain.

The ``MIXED`` case is real and common: Gulf users routinely write
``كم عدد الـ orders المتأخرة؟`` — Arabic grammar with English schema nouns borrowed
verbatim. That input must take the Arabic path (RTL display, Arabic glossary) while still
letting the English tokens match schema names directly.
"""

from __future__ import annotations

from enum import Enum

from .normalize import arabic_ratio

#: Below this, treat as English. Above ``_MIXED_UPPER``, treat as Arabic. Between the two,
#: the text is genuinely code-switched.
_MIXED_LOWER = 0.15
_MIXED_UPPER = 0.85


class Script(str, Enum):
    """Which writing system a question is predominantly in."""

    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"

    @property
    def is_rtl(self) -> bool:
        """Whether the UI should render this question right-to-left."""
        return self in (Script.ARABIC, Script.MIXED)

    @property
    def prompt_language(self) -> str:
        """Which prompt template family to use. Mixed input takes the Arabic path."""
        return "en" if self is Script.ENGLISH else "ar"


def detect_script(text: str) -> Script:
    """Classify ``text`` by the proportion of Arabic-script letters it contains."""
    ratio = arabic_ratio(text)
    if ratio >= _MIXED_UPPER:
        return Script.ARABIC
    if ratio <= _MIXED_LOWER:
        return Script.ENGLISH
    return Script.MIXED
