"""Arabic normalization tests.

Every Arabic fixture carries an English gloss and a transliteration in a comment, so the
suite is readable and reviewable without fluent Arabic.
"""

from __future__ import annotations

import pytest

from mizan.nl import (
    Script,
    arabic_ratio,
    clean_for_model,
    detect_script,
    fold_digits,
    fold_letters,
    fold_punctuation,
    normalize_for_matching,
    strip_diacritics,
    strip_invisible,
    strip_tatweel,
)


class TestInvisibleCharacters:
    """The bug class that is impossible to see in a terminal."""

    def test_rtl_mark_removed(self) -> None:
        # U+200F RIGHT-TO-LEFT MARK, routinely prepended when copying from a browser.
        assert strip_invisible("‏مرحبا") == "مرحبا"  # marhaba / "hello"

    def test_visually_identical_strings_compare_equal_after_normalizing(self) -> None:
        # Both render as "طلبات" (talabat / "orders"). One carries embedded bidi controls.
        plain = "طلبات"
        with_controls = "‫طلبات‬"
        assert plain != with_controls
        assert normalize_for_matching(plain) == normalize_for_matching(with_controls)

    @pytest.mark.parametrize(
        "char", ["​", "‌", "‍", "‎", "‏", "﻿", "­"]
    )
    def test_each_invisible_char_stripped(self, char: str) -> None:
        assert strip_invisible(f"a{char}b") == "ab"


class TestDigits:
    def test_arabic_indic_digits(self) -> None:
        # ٥٠٠ is Arabic-Indic (U+0660..) for 500.
        assert fold_digits("٥٠٠") == "500"

    def test_extended_arabic_indic_digits(self) -> None:
        # ۵۰۰ is Extended Arabic-Indic (U+06F0..), used in Persian and Urdu.
        assert fold_digits("۵۰۰") == "500"

    def test_all_ten_digits_both_ranges(self) -> None:
        assert fold_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
        assert fold_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"

    def test_nfkc_alone_does_not_fold_arabic_digits(self) -> None:
        """Guards the reason ``_DIGIT_MAP`` exists at all.

        Arabic-Indic digits have no Unicode compatibility decomposition, so NFKC leaves
        them untouched. If this ever changes upstream the explicit table is still correct,
        but the comment explaining it would become misleading.
        """
        import unicodedata

        assert unicodedata.normalize("NFKC", "٥") == "٥"
        assert fold_digits("٥") == "5"


class TestPunctuation:
    def test_arabic_question_mark(self) -> None:
        assert fold_punctuation("؟") == "?"

    def test_numeric_separators(self) -> None:
        # U+066C thousands, U+066B decimal: ١٢٬٥٠٠٫٧٥ is 12,500.75
        assert clean_for_model("١٢٬٥٠٠٫٧٥") == "12,500.75"


class TestLetterFolding:
    @pytest.mark.parametrize("variant", ["أ", "إ", "آ", "ٱ"])
    def test_alef_variants_fold(self, variant: str) -> None:
        assert fold_letters(variant) == "ا"

    def test_alef_maksura_folds_to_ya(self) -> None:
        assert fold_letters("ى") == "ي"

    def test_ta_marbuta_folds_to_ha(self) -> None:
        assert fold_letters("ة") == "ه"

    def test_spelling_variants_match(self) -> None:
        # Both mean "late" (muta'akhira); the second omits the hamza on the alef, which is
        # how it is commonly typed.
        assert normalize_for_matching("متأخرة") == normalize_for_matching("متاخره")


class TestDiacriticsAndTatweel:
    def test_diacritics_stripped(self) -> None:
        # مُحَمَّد (Muhammad) with full harakat -> محمد
        assert strip_diacritics("مُحَمَّد") == "محمد"

    def test_tatweel_stripped(self) -> None:
        # Decorative elongation: مــحــمــد -> محمد
        assert strip_tatweel("مــحــمــد") == "محمد"


class TestTwoStrengths:
    """The central design decision: meaning-preserving vs. lossy."""

    def test_clean_for_model_keeps_diacritics(self) -> None:
        text = "مُحَمَّد"
        assert "َ" in clean_for_model(text)  # fatha survives

    def test_normalize_for_matching_drops_diacritics(self) -> None:
        assert normalize_for_matching("مُحَمَّد") == "محمد"

    def test_clean_for_model_still_folds_digits(self) -> None:
        """Digit folding is meaning-preserving, so it belongs in the safe pass too."""
        assert "500" in clean_for_model("فوق ٥٠٠ درهم")  # "over 500 dirhams"

    def test_clean_for_model_removes_tatweel(self) -> None:
        assert clean_for_model("الطلبــات") == "الطلبات"  # "the orders"

    def test_full_pipeline(self) -> None:
        # "How many late orders over 500 dirhams?" with an RTL mark, diacritics, tatweel
        # and Arabic-Indic digits all present.
        raw = "‏كَم عَدَد الطَلَبــات المُتَأخِرة فوق ٥٠٠ درهم؟"
        assert clean_for_model(raw) == "كَم عَدَد الطَلَبات المُتَأخِرة فوق 500 درهم?"
        assert normalize_for_matching(raw) == "كم عدد الطلبات المتاخره فوق 500 درهم?"


class TestScriptDetection:
    def test_arabic(self) -> None:
        assert detect_script("كم عدد الطلبات؟") is Script.ARABIC

    def test_english(self) -> None:
        assert detect_script("how many orders?") is Script.ENGLISH

    def test_code_switched_is_mixed(self) -> None:
        # Arabic grammar with English schema nouns - very common in Gulf usage.
        assert detect_script("كم عدد الـ orders المتأخرة؟") is Script.MIXED

    def test_digits_only_is_not_arabic(self) -> None:
        """Digits and punctuation must not count toward the ratio."""
        assert arabic_ratio("500 ?") == 0.0
        assert detect_script("500 ?") is Script.ENGLISH

    def test_empty_string(self) -> None:
        assert detect_script("") is Script.ENGLISH

    def test_mixed_takes_arabic_prompt_path(self) -> None:
        assert Script.MIXED.prompt_language == "ar"
        assert Script.MIXED.is_rtl is True
        assert Script.ENGLISH.is_rtl is False
