"""Arabic and Latin text normalization.

The central design decision in this module
------------------------------------------
There are **two** normalization strengths, and conflating them is a real bug:

``clean_for_model``
    Meaning-preserving repairs only. Safe to apply to text that is about to be shown to a
    human or sent to an LLM. Fixes Unicode presentation forms, deletes *invisible* control
    characters, and converts Arabic-Indic digits to ASCII.

``normalize_for_matching``
    Aggressive, lossy folding used **only** as a lookup key — schema linking, glossary
    matching, cache keys. It folds letter variants and strips every diacritic. You must
    never send this to the model or display it: ``مُحَمَّد`` and ``محمد`` fold together, which is
    exactly what you want for matching and exactly what you do not want on screen.

Why folding rules follow Lucene's ``ArabicNormalizer``
------------------------------------------------------
Arabic orthography admits several spellings of the same word: hamza carriers (أ إ آ) are
routinely typed as bare alef, final ya (ي) and alef maqsura (ى) are used interchangeably in
Egyptian and Gulf typing, and ta marbuta (ة) is often typed as ha (ه). Rather than invent a
folding scheme, this module implements the same rules as Apache Lucene's
``ArabicNormalizer``, which is the long-standing reference implementation for Arabic IR.
Using an established scheme means the behaviour is explainable and comparable.

The invisible-character problem
--------------------------------
Bidirectional control characters (U+200E..U+200F, U+202A..U+202E, U+2066..U+2069) are
zero-width. Two strings that render identically on screen can differ by several of them,
because copying Arabic out of a browser, a PDF or a spreadsheet frequently embeds directional
marks. Any equality check, dictionary lookup or cache key that skips this step will fail
intermittently in a way that is invisible in a terminal and nearly impossible to eyeball.
Stripping them is the first thing both functions do.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- tables

#: Zero-width and directional formatting characters. All are deleted outright: none of them
#: carry meaning for either an LLM prompt or a string comparison.
_INVISIBLE_CHARS = (
    "­"  # SOFT HYPHEN
    "​"  # ZERO WIDTH SPACE
    "‌"  # ZERO WIDTH NON-JOINER
    "‍"  # ZERO WIDTH JOINER
    "‎"  # LEFT-TO-RIGHT MARK
    "‏"  # RIGHT-TO-LEFT MARK
    "‪"  # LEFT-TO-RIGHT EMBEDDING
    "‫"  # RIGHT-TO-LEFT EMBEDDING
    "‬"  # POP DIRECTIONAL FORMATTING
    "‭"  # LEFT-TO-RIGHT OVERRIDE
    "‮"  # RIGHT-TO-LEFT OVERRIDE
    "⁦"  # LEFT-TO-RIGHT ISOLATE
    "⁧"  # RIGHT-TO-LEFT ISOLATE
    "⁨"  # FIRST STRONG ISOLATE
    "⁩"  # POP DIRECTIONAL ISOLATE
    "﻿"  # ZERO WIDTH NO-BREAK SPACE / BOM
)
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE_CHARS}]")

#: Tatweel (kashida) — a purely decorative elongation, e.g. ``مــحــمــد`` for ``محمد``.
_TATWEEL = "ـ"

#: Harakat (short-vowel marks), tanween, shadda, sukun and Quranic annotation signs.
#: Deliberately does NOT include U+0640 (tatweel), which is handled separately because it
#: is a letter-joining character rather than a combining mark.
_DIACRITICS_RE = re.compile(
    "["
    "ؐ-ؚ"  # Arabic signs (e.g. sallallahou alayhe wassallam)
    "ً-ٟ"  # tanween, fatha/damma/kasra, shadda, sukun, extended marks
    "ٰ"  # superscript alef
    "ۖ-ۜ"  # small high Quranic marks
    "۟-ۨ"
    "۪-ۭ"
    "]"
)

#: U+0660..U+0669. Used across the Arab world. NFKC does **not** fold these to ASCII —
#: they have no compatibility decomposition — so an explicit table is mandatory.
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
#: U+06F0..U+06F9. Used in Persian and Urdu. Visually near-identical to the above for
#: several digits but a different code point range — relevant here because the user's own
#: locale (Pakistan) types these, not U+0660.
_EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_DIGIT_MAP = {
    **{ord(c): str(i) for i, c in enumerate(_ARABIC_INDIC_DIGITS)},
    **{ord(c): str(i) for i, c in enumerate(_EXTENDED_ARABIC_INDIC_DIGITS)},
}

#: Arabic punctuation → ASCII. The two numeric separators matter for Text-to-SQL: a
#: question containing ``١٢٬٥٠٠٫٧٥`` must become ``12,500.75`` or the literal the model
#: emits will not match anything in the database.
_PUNCT_MAP = {
    ord("،"): ",",  # ARABIC COMMA
    ord("؛"): ";",  # ARABIC SEMICOLON
    ord("؟"): "?",  # ARABIC QUESTION MARK
    ord("٪"): "%",  # ARABIC PERCENT SIGN
    ord("٫"): ".",  # ARABIC DECIMAL SEPARATOR
    ord("٬"): ",",  # ARABIC THOUSANDS SEPARATOR
    ord("‐"): "-",  # HYPHEN
    ord("–"): "-",  # EN DASH
    ord("—"): "-",  # EM DASH
    ord("“"): '"',
    ord("”"): '"',
    ord("‘"): "'",
    ord("’"): "'",
}

#: Lucene ArabicNormalizer letter folding.
_LETTER_FOLD_MAP = {
    ord("آ"): "ا",  # آ ALEF WITH MADDA      -> ا
    ord("أ"): "ا",  # أ ALEF WITH HAMZA ABOVE-> ا
    ord("إ"): "ا",  # إ ALEF WITH HAMZA BELOW-> ا
    ord("ٱ"): "ا",  # ٱ ALEF WASLA           -> ا
    ord("ى"): "ي",  # ى ALEF MAKSURA         -> ي
    ord("ة"): "ه",  # ة TEH MARBUTA          -> ه
}

_WHITESPACE_RE = re.compile(r"\s+")

#: Any character in the Arabic Unicode blocks, used for script detection.
_ARABIC_CHAR_RE = re.compile(
    "["
    "؀-ۿ"  # Arabic
    "ݐ-ݿ"  # Arabic Supplement
    "ࢠ-ࣿ"  # Arabic Extended-A
    "ﭐ-﷿"  # Arabic Presentation Forms-A
    "ﹰ-﻿"  # Arabic Presentation Forms-B
    "]"
)
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


# ------------------------------------------------------------------------ functions


def strip_invisible(text: str) -> str:
    """Remove zero-width and bidirectional control characters."""
    return _INVISIBLE_RE.sub("", text)


def fold_digits(text: str) -> str:
    """Map Arabic-Indic (U+0660..) and Extended Arabic-Indic (U+06F0..) digits to ASCII."""
    return text.translate(_DIGIT_MAP)


def fold_punctuation(text: str) -> str:
    """Map Arabic punctuation and numeric separators to their ASCII equivalents."""
    return text.translate(_PUNCT_MAP)


def strip_diacritics(text: str) -> str:
    """Remove harakat, tanween, shadda, sukun and Quranic annotation marks."""
    return _DIACRITICS_RE.sub("", text)


def strip_tatweel(text: str) -> str:
    """Remove the decorative kashida elongation character."""
    return text.replace(_TATWEEL, "")


def fold_letters(text: str) -> str:
    """Fold Arabic letter variants per Lucene's ``ArabicNormalizer``. Lossy by design."""
    return text.translate(_LETTER_FOLD_MAP)


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_for_model(text: str) -> str:
    """Meaning-preserving cleanup. Safe for display and for LLM prompts.

    Applies, in order: NFKC (folds Arabic presentation-form ligatures back to ordinary
    letters), invisible-character removal, tatweel removal, digit folding, punctuation
    folding, whitespace collapse.

    Deliberately does **not** strip diacritics or fold letter variants — both change what
    the text says, and the model is perfectly capable of reading ``أحمد`` as written.
    """
    text = unicodedata.normalize("NFKC", text)
    text = strip_invisible(text)
    text = strip_tatweel(text)
    text = fold_digits(text)
    text = fold_punctuation(text)
    return collapse_whitespace(text)


def normalize_for_matching(text: str) -> str:
    """Aggressive folding for use as a lookup key only. Lossy — never display this.

    Everything ``clean_for_model`` does, plus diacritic stripping, letter-variant folding
    and case folding. Two strings that a fluent reader would call "the same word spelled
    differently" compare equal after this.
    """
    text = clean_for_model(text)
    text = strip_diacritics(text)
    text = fold_letters(text)
    return text.casefold()


def arabic_ratio(text: str) -> float:
    """Fraction of *letter* characters that are Arabic-script.

    Digits, punctuation and whitespace are excluded from the denominator: "٥٠٠ ?" is not
    meaningfully Arabic text, and counting its digits would make every numeric question
    look Arabic.
    """
    arabic = len(_ARABIC_CHAR_RE.findall(text))
    latin = len(_LATIN_CHAR_RE.findall(text))
    total = arabic + latin
    return 0.0 if total == 0 else arabic / total
