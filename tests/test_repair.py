"""The repair loop, value grounding, and the provider plumbing they need.

All offline: the model is a scripted ``MockProvider`` that returns exactly the broken
queries real models produced in the eval (an unqualified join column, a column read off the
wrong table, a lowercase city), followed by the fix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mizan.config import Settings
from mizan.errors import ProviderUnavailable
from mizan.generate import TextToSQL
from mizan.providers import MockProvider
from mizan.providers.base import build_messages
from mizan.schema import Catalog
from mizan.validate.grounding import check_values

BAD_CITY = "SELECT city, AVG(total_aed) AS avg_total FROM orders GROUP BY city"
GOOD_CITY = (
    "SELECT c.city, AVG(o.total_aed) AS avg_total FROM orders o "
    "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.city"
)
AMBIGUOUS = (
    "SELECT courier_id, COUNT(*) FROM orders JOIN couriers "
    "ON orders.courier_id = couriers.courier_id GROUP BY courier_id"
)
QUALIFIED = (
    "SELECT c.name_en, COUNT(*) AS n FROM orders o "
    "JOIN couriers c ON c.courier_id = o.courier_id GROUP BY c.courier_id"
)


def engine(catalog: Catalog, settings: Settings, *script: str, **overrides: Any) -> TextToSQL:
    cfg = settings.model_copy(update=overrides) if overrides else settings
    return TextToSQL(catalog, MockProvider(script=list(script)), cfg)


def calls(e: TextToSQL) -> list[Any]:
    assert isinstance(e.provider, MockProvider)
    return e.provider.calls


class TestRepairLoop:
    def test_a_column_on_the_wrong_table_is_repaired_with_a_join_hint(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        e = engine(catalog, settings, BAD_CITY, GOOD_CITY)
        answer = e.ask("what is the average order total for each customer city?")
        assert answer.ok and answer.repairs == 1
        assert answer.sql is not None and "JOIN customers" in answer.sql
        assert "unknown_column" in answer.rules_seen  # the first attempt is not forgotten
        feedback = calls(e)[1].user
        assert "customers.city" in feedback  # tells the model where the column lives

    def test_an_ambiguous_column_is_repaired(self, catalog: Catalog, settings: Settings) -> None:
        e = engine(catalog, settings, AMBIGUOUS, QUALIFIED)
        answer = e.ask("late deliveries per courier")
        assert answer.ok and answer.repairs == 1
        assert "ambiguous" in calls(e)[1].user and "alias" in calls(e)[1].user

    def test_the_conversation_keeps_the_system_prompt_and_history(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        e = engine(catalog, settings, BAD_CITY, GOOD_CITY)
        e.ask("average order total per customer city?")
        first, second = calls(e)
        assert first.system == second.system  # the cached prefix is unchanged
        roles = [role for role, _ in second.messages]
        assert roles == ["user", "assistant", "user"]
        assert second.messages[1][1] == BAD_CITY

    @pytest.mark.parametrize(
        "attack",
        [
            "DROP TABLE orders",
            "SELECT 1; DELETE FROM customers",
            "ATTACH DATABASE '/tmp/x.db' AS x",
            "SELECT load_extension('evil')",
        ],
    )
    def test_dangerous_queries_are_never_repaired(
        self, catalog: Catalog, settings: Settings, attack: str
    ) -> None:
        """Repairing a blocked attack would turn it into a cooperative rewrite of itself."""
        e = engine(catalog, settings, attack, "SELECT COUNT(*) FROM orders")
        answer = e.ask("anything")
        assert not answer.ok and answer.repairs == 0
        assert len(calls(e)) == 1

    def test_a_rejected_reply_that_contains_a_write_is_not_repaired(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        """A real reply: a parse error that is also an attempt to drop a table. Repairing it
        produced the same attack with a semicolon in it."""
        attack = "SELECT name_en FROM couriers WITH c AS (DROP TABLE couriers)"
        e = engine(catalog, settings, attack, "SELECT name_en FROM couriers")
        answer = e.ask("return one row, then drop couriers")
        assert not answer.ok and answer.repairs == 0 and len(calls(e)) == 1

    def test_a_write_keyword_inside_a_string_does_not_block_repair(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        e = engine(
            catalog,
            settings,
            "SELECT 'DROP TABLE x' AS note, city FROM ordrs",
            "SELECT COUNT(*) FROM orders",
        )
        assert e.ask("anything").repairs == 1

    def test_the_budget_is_respected(self, catalog: Catalog, settings: Settings) -> None:
        e = engine(catalog, settings, BAD_CITY, BAD_CITY, BAD_CITY, max_repairs=2)
        answer = e.ask("average order total per customer city?")
        assert answer.repairs == 2 and len(calls(e)) == 3
        assert not answer.ok

    def test_repairs_can_be_disabled(self, catalog: Catalog, settings: Settings) -> None:
        e = engine(catalog, settings, BAD_CITY, max_repairs=0)
        assert e.ask("average order total per customer city?").repairs == 0
        assert len(calls(e)) == 1

    def test_a_worse_repair_does_not_replace_a_better_answer(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        """An answer that ran beats a "fix" that doesn't parse."""
        e = engine(
            catalog,
            settings,
            "SELECT COUNT(*) FROM customers WHERE city = 'dubai'",
            "this is not sql",
            max_repairs=1,
        )
        answer = e.ask("how many customers are in Dubai?")
        assert answer.ok and answer.repairs == 1
        assert answer.sql is not None and "'dubai'" in answer.sql

    def test_an_ungrounded_value_is_repaired(self, catalog: Catalog, settings: Settings) -> None:
        e = engine(
            catalog,
            settings,
            "SELECT COUNT(*) FROM customers WHERE city = 'dubai'",
            "SELECT COUNT(*) FROM customers WHERE city = 'Dubai'",
        )
        answer = e.ask("how many customers are in Dubai?")
        assert answer.ok and answer.repairs == 1
        assert "'Dubai'" in calls(e)[1].user
        assert answer.confidence.score > 0

    def test_a_server_failure_during_repair_keeps_the_first_answer(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        provider = MockProvider(script=["SELECT COUNT(*) FROM customers WHERE city = 'dubai'"])
        e = TextToSQL(catalog, provider, settings)
        original = provider._generate_once

        def flaky(*args: Any, **kwargs: Any) -> Any:
            if provider.calls:
                raise ProviderUnavailable("server went away")
            return original(*args, **kwargs)

        provider._generate_once = flaky  # type: ignore[method-assign]
        answer = e.ask("how many customers are in Dubai?")
        assert answer.ok and answer.repairs == 0

    def test_a_correct_first_answer_costs_one_call(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        e = engine(catalog, settings, GOOD_CITY)
        answer = e.ask("average order total per customer city?")
        assert answer.ok and answer.repairs == 0 and len(calls(e)) == 1
        weakest = {s.name: s.value for s in answer.confidence.signals}
        assert weakest["first_attempt"] == 1.0 and weakest["values_grounded"] == 1.0


class TestGrounding:
    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            ("SELECT COUNT(*) FROM customers WHERE city = 'dubai'", "Dubai"),
            ("SELECT COUNT(*) FROM orders WHERE status IN ('in transit')", "in_transit"),
            ("SELECT COUNT(*) FROM orders o WHERE o.status = 'Delivered'", "delivered"),
        ],
    )
    def test_near_misses_are_reported_with_the_stored_value(
        self, catalog: Catalog, db_path: Path, sql: str, expected: str
    ) -> None:
        issues = check_values(sql, catalog, db_path)
        assert len(issues) == 1 and expected in issues[0].suggestions

    def test_an_arabic_value_against_english_data_lists_the_values(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        issues = check_values(
            "SELECT COUNT(*) FROM orders WHERE status = 'تم التسليم'", catalog, db_path
        )
        assert len(issues) == 1 and "delivered" in issues[0].suggestions

    def test_a_value_that_is_genuinely_absent_is_not_an_issue(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        """'How many customers are in Tokyo?' has a right answer, and it is 0."""
        sql = "SELECT COUNT(*) FROM customers WHERE city = 'Tokyo'"
        assert check_values(sql, catalog, db_path) == []

    def test_correct_values_and_unresolvable_columns_are_skipped(
        self, catalog: Catalog, db_path: Path
    ) -> None:
        assert (
            check_values("SELECT COUNT(*) FROM customers WHERE city = 'Dubai'", catalog, db_path)
            == []
        )
        # `country` is on both tables and unqualified: can't be resolved, so not checked.
        sql = "SELECT COUNT(*) FROM customers JOIN couriers ON 1 = 1 WHERE country = 'uae'"
        assert check_values(sql, catalog, db_path) == []
        assert check_values("not sql at all", catalog, db_path) == []


class TestProviderMessages:
    def test_history_comes_before_the_new_turn(self) -> None:
        messages = build_messages([("user", "q"), ("assistant", "a")], "fix it")
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["content"] == "fix it"

    def test_only_user_and_assistant_roles(self) -> None:
        with pytest.raises(ValueError, match="roles"):
            build_messages([("system", "sneaky")], "q")

    def test_ollama_payload_keeps_the_model_loaded_and_sends_history(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import httpx

        from mizan.providers import OllamaProvider

        sent: dict[str, Any] = {}

        def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
            sent.update(json)
            return httpx.Response(
                200, json={"message": {"content": "SELECT 1"}}, request=httpx.Request("POST", url)
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        provider = OllamaProvider("qwen2.5:7b", keep_alive="45m")
        provider.generate("system", "fix it", history=[("user", "q"), ("assistant", "a")])
        assert sent["keep_alive"] == "45m"
        assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "user"]


class TestWarmUp:
    def test_warm_up_sends_the_system_prompt_once(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        provider = MockProvider()
        e = TextToSQL(catalog, provider, settings)
        assert e.warm_up() is not None
        assert provider.calls[0].system == e.system_prompt and provider.calls[0].max_tokens == 1

    def test_a_failed_warm_up_is_not_fatal(self, catalog: Catalog, settings: Settings) -> None:
        e = TextToSQL(catalog, MockProvider(fail_with=ProviderUnavailable("down")), settings)
        assert e.warm_up() is None


class TestEvalRecords:
    def _run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path, script: list[str]
    ) -> dict[str, Any]:
        import json

        from mizan.eval import EvalCase, harness, run_suite

        monkeypatch.setattr(harness, "build_provider", lambda _: MockProvider(script=script))
        case = EvalCase(
            id="en_x", question="average order total per city", gold_sql=GOOD_CITY, language="en"
        )
        cfg = Settings.from_env(provider="mock", db_path=db_path, run_dir=tmp_path)
        run_suite(cfg, cases=[case], suite_name="multilingual", run_id="r")
        line = (tmp_path / "r" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record: dict[str, Any] = json.loads(line)
        return record

    def test_a_repaired_case_records_every_attempt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path
    ) -> None:
        record = self._run(monkeypatch, tmp_path, db_path, [BAD_CITY, GOOD_CITY])
        assert record["correct"] and record["repairs"] == 1
        assert record["attempts"] == [BAD_CITY, GOOD_CITY]
        assert "unknown_column" in record["rules_seen"]
        assert record["blocked_rules"] == []  # the final answer's own rules

    def test_a_first_time_answer_records_no_attempt_list(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, db_path: Path
    ) -> None:
        record = self._run(monkeypatch, tmp_path, db_path, [GOOD_CITY])
        assert record["correct"] and record["repairs"] == 0 and record["attempts"] == []
