"""Natural-language preprocessing: script detection and Arabic normalization."""

from __future__ import annotations

from .detect import Script, detect_script
from .normalize import (
    arabic_ratio,
    clean_for_model,
    collapse_whitespace,
    fold_digits,
    fold_letters,
    fold_punctuation,
    normalize_for_matching,
    strip_diacritics,
    strip_invisible,
    strip_tatweel,
)

__all__ = [
    "Script",
    "arabic_ratio",
    "clean_for_model",
    "collapse_whitespace",
    "detect_script",
    "fold_digits",
    "fold_letters",
    "fold_punctuation",
    "normalize_for_matching",
    "strip_diacritics",
    "strip_invisible",
    "strip_tatweel",
]
