"""Private-value masking tests (privacy.Masker).

The masker swaps known private values in a cleaned question (customer names from the
database, plus emails, phone numbers and national ID numbers by shape) for typed
placeholders such as ``:customer_name_1``. The model only ever sees the placeholder; the
real value is bound as a SQLite parameter when the query runs (DECISIONS.md D44).

Two properties matter more than any single case:

* **No masked value survives in the text** (``TestNoValueSurvives``): the whole point.
* **A question with nothing private is returned unchanged** (``TestNoOp``): every eval
  suite is byte-identical with masking on, so the published numbers stay valid.

Arabic fixtures carry an English gloss in a comment, as in test_normalize.py.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from mizan.db.build_synthetic import build
from mizan.eval.suite import SUITES
from mizan.nl import clean_for_model
from mizan.privacy import Binding, Masker, SensitiveValue, load_privacy_config

REPO_ROOT = Path(__file__).resolve().parent.parent
PRIVACY_CONFIG = REPO_ROOT / "data" / "gulf_logistics.privacy.json"

#: Which columns are private, and the placeholder prefix each one gets.
SENSITIVE = {"customers.name_en": "customer_name", "customers.name_ar": "customer_name_ar"}

PLACEHOLDER_RE = re.compile(r":[a-z_]+_\d+")


@pytest.fixture(scope="module")
def masker(db_path: Path) -> Masker:
    return Masker.from_db(db_path, SENSITIVE)


@pytest.fixture(scope="module")
def full_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The demo database at its real size (180 customers), as the eval runs against."""
    path = tmp_path_factory.mktemp("mizan-full-db") / "full.sqlite"
    build(path)
    return path


def _values(db: Path, column: str) -> list[str]:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return [r[0] for r in conn.execute(f"SELECT DISTINCT {column} FROM customers")]


class TestConfig:
    def test_committed_config_loads(self) -> None:
        assert load_privacy_config(PRIVACY_CONFIG) == SENSITIVE

    def test_unknown_column_is_an_error(self, db_path: Path) -> None:
        # A typo in the config must not silently mask nothing.
        with pytest.raises(ValueError, match=r"customers\.nmae_en"):
            Masker.from_db(db_path, {"customers.nmae_en": "customer_name"})


class TestEnglishNames:
    def test_full_name_is_replaced(self, masker: Masker) -> None:
        masked = masker.mask("how many orders did Tariq Al Thani place?")
        assert masked.text == "how many orders did :customer_name_1 place?"
        assert masked.bindings == {
            "customer_name_1": Binding("Tariq Al Thani", "customers", "name_en")
        }

    def test_case_insensitive_and_bound_to_the_stored_spelling(self, masker: Masker) -> None:
        # The binding carries the database's spelling, so `name_en = :customer_name_1` matches.
        masked = masker.mask("orders of tariq al thani")
        assert masked.text == "orders of :customer_name_1"
        assert masked.bindings["customer_name_1"].value == "Tariq Al Thani"

    def test_punctuation_after_the_name(self, masker: Masker) -> None:
        masked = masker.mask("Did Omar Al Thani, or Khalid Al Mansouri, order more?")
        assert masked.text == "Did :customer_name_1, or :customer_name_2, order more?"
        assert [b.value for b in masked.bindings.values()] == [
            "Omar Al Thani",
            "Khalid Al Mansouri",
        ]

    def test_same_name_twice_reuses_one_placeholder(self, masker: Masker) -> None:
        masked = masker.mask("Tariq Al Thani vs Tariq Al Thani last year")
        assert masked.text == ":customer_name_1 vs :customer_name_1 last year"
        assert len(masked.bindings) == 1

    @pytest.mark.parametrize("apostrophe", ["'", "’"])
    def test_a_possessive_is_masked_and_kept(self, masker: Masker, apostrophe: str) -> None:
        # Found by the privacy eval: "Harbi's" was one word, so the name went unmasked.
        masked = masker.mask(f"how many of Tariq Al Thani{apostrophe}s orders were late?")
        assert masked.text == f"how many of :customer_name_1{apostrophe}s orders were late?"
        assert masked.bindings["customer_name_1"].value == "Tariq Al Thani"

    def test_a_trailing_apostrophe_possessive_is_masked(self, masker: Masker) -> None:
        assert masker.mask("Tariq Al Thani' orders").text == ":customer_name_1' orders"

    def test_first_name_alone_is_not_matched(self, masker: Masker) -> None:
        # Known limit (D44): only full stored values are recognised.
        masked = masker.mask("orders of Tariq")
        assert masked.text == "orders of Tariq"
        assert masked.bindings == {}


class TestArabicScriptNames:
    def test_arabic_name(self, masker: Masker) -> None:
        # "How many orders did Tariq Al Thani place?"
        masked = masker.mask("كم طلب قدم طارق آل ثاني؟")
        assert masked.text == "كم طلب قدم :customer_name_ar_1؟"
        assert masked.bindings == {
            "customer_name_ar_1": Binding("طارق آل ثاني", "customers", "name_ar")
        }

    def test_alef_variant_matches_the_stored_spelling(self, masker: Masker) -> None:
        # آل typed as ال (no madda): same name to a reader, bound to the stored spelling.
        masked = masker.mask("طلبات طارق ال ثاني")  # "Tariq Al Thani's orders"
        assert masked.text == "طلبات :customer_name_ar_1"
        assert masked.bindings["customer_name_ar_1"].value == "طارق آل ثاني"

    def test_urdu_question_with_urdu_letters_in_the_name(self, masker: Masker) -> None:
        # Urdu keyboards type ی (farsi yeh) where the database has ي. "How many orders has
        # Khalid Al Mansouri placed?"
        masked = masker.mask("خالد المنصوری نے کتنے آرڈر دیے؟")
        assert masked.text == ":customer_name_ar_1 نے کتنے آرڈر دیے؟"
        assert masked.bindings["customer_name_ar_1"].value == "خالد المنصوري"

    def test_english_name_inside_an_urdu_question(self, masker: Masker) -> None:
        masked = masker.mask("Tariq Al Thani کے کتنے آرڈر ہیں؟")  # "... how many orders?"
        assert masked.text == ":customer_name_1 کے کتنے آرڈر ہیں؟"

    def test_arabic_name_is_not_matched_inside_a_longer_word(self, masker: Masker) -> None:
        # Word boundaries apply in Arabic script too: a prefix glued to the name
        # (و "and") makes a different token, which is left alone rather than half-masked.
        text = "وطارق آل ثانيين"
        assert masker.mask(text).text == text


class TestLongestMatchAndBoundaries:
    def test_longest_value_wins(self) -> None:
        masker = Masker(
            [
                SensitiveValue("Omar Al Thani", "customers", "name_en", "customer_name"),
                SensitiveValue("Omar Al Thani Jr", "customers", "name_en", "customer_name"),
            ]
        )
        masked = masker.mask("orders of Omar Al Thani Jr")
        assert masked.text == "orders of :customer_name_1"
        assert masked.bindings["customer_name_1"].value == "Omar Al Thani Jr"

    @pytest.mark.parametrize("text", ["Alice ordered", "orders from Bali", "Ali-Baba"])
    def test_whole_words_only(self, text: str) -> None:
        # A hyphenated token ("Ali-Baba") is one word, so "Ali" is not inside it either.
        masker = Masker([SensitiveValue("Ali", "customers", "name_en", "customer_name")])
        assert masker.mask(text).text == text


class TestShapes:
    """Private values recognised by shape, with no column to bind them to."""

    @pytest.mark.parametrize(
        ("text", "masked_text", "value"),
        [
            ("orders for tariq@example.com", "orders for :email_1", "tariq@example.com"),
            ("call +92 300 1234567 today", "call :phone_1 today", "+92 300 1234567"),
            ("customer +971 50 123 4567", "customer :phone_1", "+971 50 123 4567"),
            ("رقم +966 55 123 4567", "رقم :phone_1", "+966 55 123 4567"),  # "number ..."
            ("CNIC 42101-1234567-1 orders", "CNIC :national_id_1 orders", "42101-1234567-1"),
            ("ID 784-1990-1234567-1", "ID :national_id_1", "784-1990-1234567-1"),
        ],
    )
    def test_shape_is_masked(self, masker: Masker, text: str, masked_text: str, value: str) -> None:
        masked = masker.mask(text)
        assert masked.text == masked_text
        (binding,) = masked.bindings.values()
        assert binding.value == value
        assert binding.table is None and binding.column is None

    def test_arabic_digits_are_masked_after_cleanup(self, masker: Masker) -> None:
        # The masker runs on clean_for_model output, which folds Arabic-Indic digits.
        masked = masker.mask(clean_for_model("رقم +٩٧١ ٥٠ ١٢٣ ٤٥٦٧"))
        assert masked.text == "رقم :phone_1"

    @pytest.mark.parametrize(
        "text",
        [
            "status of order 1042",
            "orders placed on 2024-03-15",
            "orders above 1,250 AED",
            "top 3 couriers in 2024",
            "orders between 10:00 and 18:00",
        ],
    )
    def test_ordinary_numbers_are_left_alone(self, masker: Masker, text: str) -> None:
        masked = masker.mask(text)
        assert masked.text == text
        assert masked.bindings == {}


class TestNoOp:
    @pytest.mark.parametrize(
        "text",
        ["orders from Doha", "late deliveries in Kuwait City", "customers in Abu Dhabi"],
    )
    def test_city_names_are_not_private(self, masker: Masker, text: str) -> None:
        assert masker.mask(text).text == text

    @pytest.mark.parametrize("suite", sorted(SUITES))
    def test_every_eval_question_is_unchanged(self, full_db: Path, suite: str) -> None:
        # The guarantee behind "existing evals are byte-identical with masking on". The
        # privacy suites (tag "pii") name customers on purpose and are the exception.
        masker = Masker.from_db(full_db, SENSITIVE)
        for case in SUITES[suite]():
            if "pii" in case.tags:
                continue
            question = clean_for_model(case.question)
            masked = masker.mask(question)
            assert masked.text == question, case.id
            assert masked.bindings == {}, case.id


class TestNoValueSurvives:
    @pytest.mark.parametrize("column", ["name_en", "name_ar"])
    def test_every_stored_name_is_removed(self, full_db: Path, column: str) -> None:
        masker = Masker.from_db(full_db, SENSITIVE)
        for name in _values(full_db, column):
            masked = masker.mask(f"orders of {name} this month")
            assert name not in masked.text, name
            assert [b.value for b in masked.bindings.values()] == [name]


class TestPlaceholderSpoofing:
    def test_typed_placeholder_is_not_a_binding(self, masker: Masker) -> None:
        # A user typing a placeholder gets no value bound to it; the validator later rejects
        # any placeholder the masker didn't issue (unknown_placeholder).
        masked = masker.mask("orders of :customer_name_1")
        assert masked.bindings == {}

    def test_issued_names_never_collide_with_typed_ones(self, masker: Masker) -> None:
        masked = masker.mask(":customer_name_1 and Tariq Al Thani")
        typed = {":customer_name_1"}
        issued = {f":{name}" for name in masked.bindings}
        assert issued and not issued & typed
        assert all(PLACEHOLDER_RE.fullmatch(p) for p in issued)


class TestKnownBindings:
    """Follow-ups: placeholders from earlier in the conversation keep their names."""

    def test_a_known_value_keeps_its_name_and_a_new_one_is_numbered_after_it(
        self, masker: Masker
    ) -> None:
        known = {"customer_name_1": Binding("Tariq Al Thani", "customers", "name_en")}
        masked = masker.mask("Tariq Al Thani and Omar Al Thani", known=known)
        assert masked.text == ":customer_name_1 and :customer_name_2"
        assert masked.bindings["customer_name_2"].value == "Omar Al Thani"
