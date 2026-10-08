"""Urdu support, and the scoring changes that came with the trilingual suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.eval import build_holdout, build_suite, get_suite, score_prediction
from mizan.eval.metrics import _percentile
from mizan.eval.suite import INJECTION_CASES, LANGUAGES
from mizan.nl import Script, clean_for_model, detect_script, normalize_for_matching


class TestUrduDetection:
    @pytest.mark.parametrize(
        ("text", "script"),
        [
            ("کتنے آرڈر ہیں؟", Script.URDU),  # "how many orders are there?"
            ("دبئی میں کتنے کسٹمرز ہیں؟", Script.URDU),
            ("کتنے orders pending ہیں؟", Script.URDU),  # code-switched, as Urdu is written
            ("كم عدد الطلبات؟", Script.ARABIC),
            ("ما هي المنتجات الثلاثة الأكثر طلبًا من حيث الكمية؟", Script.ARABIC),
            ("كم عدد الـ orders المتأخرة؟", Script.MIXED),
            ("how many orders?", Script.ENGLISH),
            ("kitne orders pending hain?", Script.ENGLISH),  # Roman Urdu: a known limit
        ],
    )
    def test_detects(self, text: str, script: Script) -> None:
        assert detect_script(text) is script

    def test_every_suite_question_is_detected_as_its_language(self) -> None:
        for case in (*build_suite(), *build_holdout(), *INJECTION_CASES):
            assert detect_script(case.question).value == case.language, case.id

    def test_urdu_is_right_to_left(self) -> None:
        assert Script.URDU.is_rtl and Script.URDU.prompt_language == "ur"


class TestUrduNormalization:
    def test_urdu_digits_and_full_stop_become_ascii(self) -> None:
        assert clean_for_model("۵۰۰ سے زیادہ۔") == "500 سے زیادہ."

    def test_model_text_keeps_urdu_letters(self) -> None:
        """ک, ی and ہ are the correct Urdu letters; folding them is for matching only."""
        assert clean_for_model("کسٹمر ہے") == "کسٹمر ہے"

    def test_matching_folds_urdu_and_arabic_keyboards_together(self) -> None:
        assert normalize_for_matching("کسٹمر") == normalize_for_matching("كسٹمر")
        assert normalize_for_matching("ہے") == normalize_for_matching("هي")


class TestSuites:
    def test_every_question_is_asked_in_every_language(self) -> None:
        for suite in (build_suite(), build_holdout()):
            by_pair: dict[str, set[str]] = {}
            for case in suite:
                by_pair.setdefault(case.pair_id, set()).add(case.language)
                assert case.language == "en" or case.gloss, case.id
            assert all(langs == set(LANGUAGES) for langs in by_pair.values())

    def test_holdout_shares_no_question_with_the_dev_suite(self) -> None:
        dev = {c.pair_id for c in build_suite()}
        assert not dev & {c.pair_id for c in build_holdout()}

    def test_original_case_ids_are_kept_for_the_regression_gate(self) -> None:
        ids = {c.id for c in build_suite()}
        assert {"en_late_by_courier", "ar_late_by_courier", "en_count_orders"} <= ids

    def test_unknown_suite(self) -> None:
        with pytest.raises(ValueError, match="unknown suite"):
            get_suite("bilingual")

    def test_every_gold_query_runs_and_returns_rows(self, db_path: Path) -> None:
        """A gold query that errors, or returns nothing, would score anything as wrong
        (or every empty answer as right). Checked on the test database."""
        from mizan.guardrails import execute

        for case in (*build_suite(), *build_holdout()):
            for gold in case.gold_queries:
                assert execute(gold, db_path, max_rows=500).rows, case.id


class TestScoring:
    def test_strict_and_relaxed(self, db_path: Path) -> None:
        gold = "SELECT name_en FROM couriers WHERE is_active = 0"
        same = score_prediction("SELECT name_en FROM couriers WHERE is_active = 0", gold, db_path)
        assert same.strict and same.relaxed
        extra = score_prediction(
            "SELECT name_en, name_ar FROM couriers WHERE is_active = 0", gold, db_path
        )
        assert not extra.strict and extra.relaxed  # right answer, one extra column
        wrong = score_prediction("SELECT name_en FROM couriers", gold, db_path)
        assert not wrong.strict and not wrong.relaxed

    def test_relaxed_never_accepts_a_missing_column(self, db_path: Path) -> None:
        gold = "SELECT status, COUNT(*) FROM orders GROUP BY status"
        missing = score_prediction("SELECT status FROM orders GROUP BY status", gold, db_path)
        assert not missing.strict and not missing.relaxed

    def test_any_listed_gold_counts(self, db_path: Path) -> None:
        golds = ("SELECT 1", "SELECT 2")
        assert score_prediction("SELECT 2", golds, db_path).strict
        assert not score_prediction("SELECT 3", golds, db_path).strict

    def test_percentile_is_a_measured_value(self) -> None:
        values = sorted(float(v) for v in range(1, 21))
        assert _percentile(values, 50) == 10.0
        assert _percentile(values, 95) == 19.0
        assert _percentile([], 95) == 0.0
