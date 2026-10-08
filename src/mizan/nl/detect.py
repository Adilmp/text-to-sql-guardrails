"""Script/language detection for incoming questions.

Why not a statistical language detector
---------------------------------------
``langdetect``/``fasttext`` would add a dependency and a model file to answer a question we
can answer exactly from the code points. A rule you can read is worth more here than a
classifier you cannot explain.

Arabic and Urdu share a script but not an alphabet: Urdu adds letters Arabic never uses
(ٹ ڈ ڑ ں ے ہ ھ) and types its own forms of shared ones (ک ی for ك ي). So once the text is
known to be Arabic-script, counting those letters separates the two languages
(``urdu_score``). Persian would be read as Urdu; it is not a supported language.

The ``MIXED`` case is real and common: Gulf users routinely write
``كم عدد الـ orders المتأخرة؟`` — Arabic grammar with English schema nouns borrowed
verbatim. That input must take the Arabic path (RTL display, Arabic glossary) while still
letting the English tokens match schema names directly.
"""

from __future__ import annotations

from enum import Enum

from .normalize import arabic_ratio, urdu_score

#: Below this, treat as English. Above ``_MIXED_UPPER``, treat as Arabic. Between the two,
#: the text is genuinely code-switched.
_MIXED_LOWER = 0.15
_MIXED_UPPER = 0.85


class Script(str, Enum):
    """Which writing system a question is predominantly in."""

    ARABIC = "ar"
    ENGLISH = "en"
    MIXED = "mixed"
    #: Urdu, including Urdu with English loanwords in Latin letters. Code-switching is the
    #: norm in written Urdu ("کتنے orders pending ہیں؟"), so it is not split into a mixed
    #: class of its own. Roman Urdu (Urdu in Latin letters) reads as ENGLISH: telling it
    #: apart is a language-identification problem, not a script one.
    URDU = "ur"

    @property
    def is_rtl(self) -> bool:
        """Whether the UI should render this question right-to-left."""
        return self is not Script.ENGLISH

    @property
    def prompt_language(self) -> str:
        """The question's language for the prompt: mixed Arabic/English counts as Arabic."""
        if self is Script.ENGLISH:
            return "en"
        return "ur" if self is Script.URDU else "ar"


def detect_script(text: str) -> Script:
    """Classify ``text`` by the proportion of Arabic-script letters, then Arabic vs Urdu."""
    ratio = arabic_ratio(text)
    if ratio <= _MIXED_LOWER:
        return Script.ENGLISH
    if urdu_score(text) > 0:
        return Script.URDU
    if ratio >= _MIXED_UPPER:
        return Script.ARABIC
    return Script.MIXED
