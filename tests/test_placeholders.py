"""Placeholders through the engine: prompt, validator, executor, grounding, repair, pipeline.

The masker (test_masker.py) turns "orders of Tariq Al Thani" into "orders of
:customer_name_1". These tests follow that placeholder the rest of the way: the model is
told which column it belongs to, the validator only accepts placeholders that were issued,
the executor binds the real value as a SQLite parameter, grounding checks the column without
looking the value up, and the repair loop never quotes a real value back to the model
(DECISIONS.md D44).

Every model here is a scripted ``MockProvider``, and every test that drives the pipeline
also asserts the one property that matters: **no private value appears in anything sent
to the model.**
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mizan.config import Settings
from mizan.errors import ExecutionError
from mizan.generate import TextToSQL
from mizan.generate.prompt import build_user_prompt
from mizan.generate.repair import diagnose
from mizan.guardrails import execute, validate
from mizan.privacy import Binding, Masker
from mizan.providers import MockProvider
from mizan.schema import Catalog
from mizan.validate.grounding import check_values

SENSITIVE = {"customers.name_en": "customer_name", "customers.name_ar": "customer_name_ar"}
TARIQ = Binding("Tariq Al Thani", "customers", "name_en")

ORDERS_OF = (
    "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
    "WHERE c.name_en = {}"
)


@pytest.fixture(scope="module")
def masker(db_path: Path) -> Masker:
    return Masker.from_db(db_path, SENSITIVE)


def _count_orders(db: Path, name: str) -> int:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        (n,) = conn.execute(ORDERS_OF.format("?"), (name,)).fetchone()
    return int(n)


def _engine(
    catalog: Catalog, settings: Settings, masker: Masker | None, *script: str
) -> tuple[TextToSQL, MockProvider]:
    provider = MockProvider(script=list(script))
    return TextToSQL(catalog, provider, settings, masker=masker), provider


def _sent(provider: MockProvider) -> str:
    """Everything the model was shown, across every call."""
    return "\n".join(
        "\n".join([call.system, call.user, *(text for _, text in call.messages)])
        for call in provider.calls
    )


def _assert_no_private_values(provider: MockProvider, *values: str) -> None:
    sent = _sent(provider)
    for value in values:
        assert value not in sent, f"{value!r} reached the model"


class TestPrompt:
    def test_no_bindings_is_byte_identical(self) -> None:
        # What every eval case sends today must not change.
        assert build_user_prompt("how many orders?", bindings={}) == build_user_prompt(
            "how many orders?"
        )

    def test_each_placeholder_is_explained_without_its_value(self) -> None:
        prompt = build_user_prompt(
            "orders of :customer_name_1 or :email_1",
            bindings={"customer_name_1": TARIQ, "email_1": Binding("t@example.com")},
        )
        assert ":customer_name_1 is a value of customers.name_en" in prompt
        assert ":email_1" in prompt
        assert "unquoted" in prompt
        assert "Tariq" not in prompt and "t@example.com" not in prompt
        assert prompt.endswith("SQL:")


class TestValidator:
    def test_an_issued_placeholder_is_allowed(self, catalog: Catalog) -> None:
        report = validate(
            ORDERS_OF.format(":customer_name_1"), catalog, placeholders={"customer_name_1"}
        )
        assert report.ok, report.violations
        assert ":customer_name_1" in report.sql

    @pytest.mark.parametrize("placeholder", [":customer_name_2", ":_user_country", "?"])
    def test_a_placeholder_that_was_not_issued_is_rejected(
        self, catalog: Catalog, placeholder: str
    ) -> None:
        report = validate(ORDERS_OF.format(placeholder), catalog, placeholders={"customer_name_1"})
        assert report.rules_fired == ["unknown_placeholder"]

    def test_no_placeholders_are_allowed_by_default(self, catalog: Catalog) -> None:
        report = validate(ORDERS_OF.format(":customer_name_1"), catalog)
        assert report.rules_fired == ["unknown_placeholder"]

    def test_a_quoted_placeholder_name_without_its_colon_is_rejected(
        self, catalog: Catalog
    ) -> None:
        # Found by the privacy eval: qwen2.5:7b wrote `= 'customer_name_1'`, which ran and
        # silently matched nothing. It is the same mistake as quoting it with the colon.
        report = validate(
            ORDERS_OF.format("'customer_name_1'"), catalog, placeholders={"customer_name_1"}
        )
        assert report.rules_fired == ["quoted_placeholder"]

    def test_a_quoted_placeholder_is_rejected(self, catalog: Catalog) -> None:
        # Quoted, it is the literal text ':customer_name_1' and silently matches nothing.
        report = validate(
            ORDERS_OF.format("':customer_name_1'"), catalog, placeholders={"customer_name_1"}
        )
        assert report.rules_fired == ["quoted_placeholder"]


class TestExecutor:
    def test_the_value_is_bound_not_spliced(self, db_path: Path) -> None:
        result = execute(
            ORDERS_OF.format(":customer_name_1"),
            db_path,
            params={"customer_name_1": "Tariq Al Thani"},
        )
        assert result.rows == ((_count_orders(db_path, "Tariq Al Thani"),),)

    def test_a_bound_value_cannot_change_the_statement(self, db_path: Path) -> None:
        result = execute(
            ORDERS_OF.format(":customer_name_1"),
            db_path,
            params={"customer_name_1": "x' OR '1'='1"},
        )
        assert result.rows == ((0,),)

    def test_a_missing_value_is_an_execution_error(self, db_path: Path) -> None:
        with pytest.raises(ExecutionError):
            execute(ORDERS_OF.format(":customer_name_1"), db_path)


class TestGrounding:
    def test_a_placeholder_on_its_own_column_is_grounded(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        sql = ORDERS_OF.format(":customer_name_1")
        assert check_values(sql, catalog, db_path, bindings={"customer_name_1": TARIQ}) == []

    def test_a_placeholder_on_the_wrong_column_is_reported_by_name(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        sql = (
            "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE c.city = :customer_name_1"
        )
        (issue,) = check_values(sql, catalog, db_path, bindings={"customer_name_1": TARIQ})
        text = str(issue)
        assert ":customer_name_1" in text and "customers.name_en" in text
        assert "Tariq" not in text

    def test_a_sensitive_column_never_offers_stored_values(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        # Without this, "orders of tariq" would come back as "did you mean 'Tariq Al
        # Thani', 'Tariq Al Otaibi'?": names the user never typed, sent to the model.
        sql = ORDERS_OF.format("'tariq al thani'")
        issues = check_values(
            sql, catalog, db_path, sensitive={("customers", "name_en"), ("customers", "name_ar")}
        )
        assert all("Tariq" not in str(i) for i in issues)


class TestRepairHints:
    def test_an_unknown_placeholder_lists_the_issued_ones(self, catalog: Catalog) -> None:
        report = validate(
            ORDERS_OF.format(":customer_name_2"), catalog, placeholders={"customer_name_1"}
        )
        problem = diagnose(report, None, [], catalog)
        assert problem is not None and ":customer_name_1" in problem.feedback

    def test_a_quoted_placeholder_is_told_to_drop_the_quotes(self, catalog: Catalog) -> None:
        report = validate(
            ORDERS_OF.format("':customer_name_1'"), catalog, placeholders={"customer_name_1"}
        )
        problem = diagnose(report, None, [], catalog)
        assert problem is not None and "without quotes" in problem.feedback


class TestPipeline:
    QUESTION = "how many orders did Tariq Al Thani place?"

    def test_the_model_sees_the_placeholder_and_the_answer_uses_the_value(
        self, catalog: Catalog, settings: Settings, masker: Masker, db_path: Path
    ) -> None:
        engine, provider = _engine(catalog, settings, masker, ORDERS_OF.format(":customer_name_1"))
        answer = engine.ask(self.QUESTION)
        assert answer.ok, answer.error
        assert answer.result is not None
        assert answer.result.rows == ((_count_orders(db_path, "Tariq Al Thani"),),)
        assert ":customer_name_1" in provider.calls[0].user
        assert answer.sql is not None and ":customer_name_1" in answer.sql
        assert "Tariq" not in answer.sql
        assert answer.bindings == {"customer_name_1": TARIQ}
        _assert_no_private_values(provider, "Tariq Al Thani")

    def test_to_dict_names_placeholders_but_not_values(
        self, catalog: Catalog, settings: Settings, masker: Masker
    ) -> None:
        engine, _ = _engine(catalog, settings, masker, ORDERS_OF.format(":customer_name_1"))
        record = engine.ask(self.QUESTION).to_dict()
        assert record["placeholders"] == {
            "customer_name_1": {"table": "customers", "column": "name_en"}
        }

    def test_without_a_masker_nothing_changes(self, catalog: Catalog, settings: Settings) -> None:
        engine, provider = _engine(catalog, settings, None, "SELECT COUNT(*) FROM orders")
        engine.ask("how many orders are there?")
        assert provider.calls[0].user == build_user_prompt("how many orders are there?")

    def test_a_question_with_nothing_private_sends_the_same_prompt(
        self, catalog: Catalog, settings: Settings, masker: Masker
    ) -> None:
        masked, with_masker = _engine(catalog, settings, masker, "SELECT COUNT(*) FROM orders")
        plain, without = _engine(catalog, settings, None, "SELECT COUNT(*) FROM orders")
        masked.ask("how many orders are there in Doha?")
        plain.ask("how many orders are there in Doha?")
        assert with_masker.calls[0].user == without.calls[0].user

    def test_a_quoted_placeholder_is_repaired(
        self, catalog: Catalog, settings: Settings, masker: Masker, db_path: Path
    ) -> None:
        engine, provider = _engine(
            catalog,
            settings,
            masker,
            ORDERS_OF.format("':customer_name_1'"),
            ORDERS_OF.format(":customer_name_1"),
        )
        answer = engine.ask(self.QUESTION)
        assert answer.ok and answer.repairs == 1
        assert answer.result is not None
        assert answer.result.rows == ((_count_orders(db_path, "Tariq Al Thani"),),)
        _assert_no_private_values(provider, "Tariq Al Thani")

    def test_a_placeholder_on_the_wrong_column_is_repaired_without_the_value(
        self, catalog: Catalog, settings: Settings, masker: Masker
    ) -> None:
        wrong = (
            "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE c.city = :customer_name_1"
        )
        engine, provider = _engine(
            catalog, settings, masker, wrong, ORDERS_OF.format(":customer_name_1")
        )
        answer = engine.ask(self.QUESTION)
        assert answer.ok and answer.repairs == 1
        assert "customers.name_en" in provider.calls[1].user
        _assert_no_private_values(provider, "Tariq Al Thani")

    def test_an_arabic_name_is_bound_to_the_arabic_column(
        self, catalog: Catalog, settings: Settings, masker: Masker, db_path: Path
    ) -> None:
        sql = (
            "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE c.name_ar = :customer_name_ar_1"
        )
        engine, provider = _engine(catalog, settings, masker, sql)
        answer = engine.ask(
            "كم طلب قدم طارق آل ثاني؟"
        )  # "How many orders did Tariq Al Thani place?"
        assert answer.ok, answer.error
        assert answer.result is not None
        assert answer.result.rows == ((_count_orders(db_path, "Tariq Al Thani"),),)
        _assert_no_private_values(provider, "طارق آل ثاني")

    def test_a_follow_up_keeps_earlier_placeholders_and_numbers_new_ones(
        self, catalog: Catalog, settings: Settings, masker: Masker, db_path: Path
    ) -> None:
        earlier_sql = ORDERS_OF.format(":customer_name_1")
        both = (
            "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
            "WHERE c.name_en IN (:customer_name_1, :customer_name_2)"
        )
        engine, provider = _engine(catalog, settings, masker, both)
        answer = engine.ask(
            "and Omar Al Thani together with them?",
            context=[(self.QUESTION, earlier_sql)],
        )
        assert answer.ok, answer.error
        assert answer.result is not None
        expected = _count_orders(db_path, "Tariq Al Thani") + _count_orders(
            db_path, "Omar Al Thani"
        )
        assert answer.result.rows == ((expected,),)
        history_user = provider.calls[0].messages[0][1]
        assert ":customer_name_1" in history_user
        assert ":customer_name_2" in provider.calls[0].user
        _assert_no_private_values(provider, "Tariq Al Thani", "Omar Al Thani")
