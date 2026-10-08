"""Follow-up questions (context in the conversation) and clarifying questions (asking back)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mizan.cache import AnswerCache, CachedTextToSQL, CacheSettings
from mizan.clarify import Clarifier
from mizan.config import Settings
from mizan.eval import build_followups, build_holdout, build_suite
from mizan.eval.suite import INJECTION_CASES
from mizan.generate import TextToSQL
from mizan.providers import MockProvider
from mizan.schema import Catalog

LATE = "SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at"
LATE_RIYADH = (
    "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
    "WHERE c.city = 'Riyadh' AND o.delivered_at IS NOT NULL AND o.delivered_at > o.promised_at"
)
CLARIFICATIONS = Path(__file__).resolve().parent.parent / "data/gulf_logistics.clarifications.json"


class TestFollowUpPipeline:
    def test_a_standalone_question_is_sent_exactly_as_before(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        """No context, no history: the request the eval measured, byte for byte."""
        provider = MockProvider(script=[LATE])
        TextToSQL(catalog, provider, settings).ask("how many orders were delivered late?")
        assert [role for role, _ in provider.calls[0].messages] == ["user"]

    def test_the_earlier_exchange_comes_before_the_follow_up(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        provider = MockProvider(script=[LATE_RIYADH])
        engine = TextToSQL(catalog, provider, settings)
        answer = engine.ask(
            "and how many of those went to customers in Riyadh?",
            context=[("how many orders were delivered late?", LATE)],
        )
        assert answer.ok
        messages = provider.calls[0].messages
        assert [role for role, _ in messages] == ["user", "assistant", "user"]
        assert "delivered late" in messages[0][1] and messages[1][1] == LATE
        assert provider.calls[0].system == engine.system_prompt  # same cached prefix

    def test_a_repair_keeps_the_earlier_exchange(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        bad = "SELECT city, COUNT(*) FROM orders GROUP BY city"
        provider = MockProvider(script=[bad, LATE_RIYADH])
        TextToSQL(catalog, provider, settings).ask(
            "and in Riyadh?", context=[("how many orders were delivered late?", LATE)]
        )
        roles = [role for role, _ in provider.calls[1].messages]
        assert roles == ["user", "assistant", "user", "assistant", "user"]

    def test_follow_up_answers_are_never_cached(
        self, catalog: Catalog, settings: Settings, tmp_path: Path
    ) -> None:
        """ "what about Riyadh?" means nothing on its own, so it can't be reused."""
        provider = MockProvider(script=[LATE_RIYADH, LATE_RIYADH])
        front = CachedTextToSQL(
            TextToSQL(catalog, provider, settings),
            AnswerCache(tmp_path / "c.sqlite"),
            CacheSettings(),
        )
        context = [("how many orders were delivered late?", LATE)]
        front.ask("and in Riyadh?", context)
        _, hit = front.ask("and in Riyadh?", context)
        assert hit is None and len(provider.calls) == 2


class TestFollowUpSuite:
    def test_every_conversation_in_three_languages(self) -> None:
        cases = build_followups()
        assert len(cases) == 24
        assert all(len(c.context) == 1 and c.context[0][1] for c in cases)
        assert {c.language for c in cases} == {"en", "ar", "ur"}
        assert any("topic_switch" in c.tags for c in cases)  # the control is there

    def test_gold_queries_and_earlier_answers_run(self, db_path: Path) -> None:
        from mizan.guardrails import execute

        for case in build_followups():
            assert execute(case.gold_sql, db_path).rows, case.id
            assert execute(case.context[0][1], db_path).rows, case.id


class TestClarifier:
    clarifier = Clarifier.from_file(CLARIFICATIONS)

    def test_asks_only_where_the_eval_found_real_ambiguity(self) -> None:
        """Across every eval question, it fires on exactly the two whose verbs ("fulfil",
        "shipped") split the model and the gold answer, in all three languages."""
        questions = [
            (c.id, c.question)
            for c in (*build_suite(), *build_holdout(), *build_followups(), *INJECTION_CASES)
        ]
        asked = {i for i, q in questions if self.clarifier.check(q)}
        assert asked == {
            f"{lang}_{pair}"
            for lang in ("en", "ar", "ur")
            for pair in ("warehouse_orders_riyadh", "cross_border_orders")
        }

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("which product has the most sales?", "sales"),
            ("ما المنتجات الأكثر مبيعاً؟", "sales"),
            ("سب سے زیادہ بکنے والی پروڈکٹ کون سی ہے؟", "sales"),
            ("what are the best-selling products by units?", None),  # "units" settles it
            ("top selling product by revenue", None),
            ("how many orders were sent to Dubai?", "fulfilled"),
            ("how many delivered orders were shipped from Dubai Hub?", None),
            ("please write a sentence", None),  # "sent" is a whole word, not a prefix
        ],
    )
    def test_terms_and_the_words_that_settle_them(
        self, question: str, expected: str | None
    ) -> None:
        found = self.clarifier.check(question)
        assert (found.id if found else None) == expected

    def test_the_choice_becomes_a_hint_and_its_label_is_in_the_question_s_language(self) -> None:
        question, label = self.clarifier.apply("کتنے آرڈر بھیجے گئے؟", "fulfilled", 1)
        assert question.endswith("(count only orders that were delivered)")
        assert label == "صرف ڈیلیور شدہ آرڈر"
        with pytest.raises(ValueError):
            self.clarifier.apply("x", "fulfilled", 7)
        with pytest.raises(ValueError):
            self.clarifier.apply("x", "nonsense", 0)

    def test_a_missing_file_means_nothing_is_ambiguous(self, tmp_path: Path) -> None:
        assert Clarifier.from_file(tmp_path / "none.json").check("best-selling?") is None


@pytest.fixture()
def api(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, MockProvider]]:
    import shutil

    import mizan.api as api_module
    from mizan.api import create_app

    # The test database gets the real clarifications file beside it.
    db = tmp_path / "db.sqlite"
    shutil.copy(db_path, db)
    shutil.copy(CLARIFICATIONS, db.with_suffix(".clarifications.json"))
    provider = MockProvider(
        fixtures={
            "how many orders were delivered late?": LATE,
            "in riyadh": LATE_RIYADH,
            "count all orders": "SELECT w.name_en, COUNT(*) FROM orders o JOIN warehouses w "
            "ON w.warehouse_id = o.warehouse_id GROUP BY w.warehouse_id",
        }
    )
    monkeypatch.setattr(api_module, "build_provider", lambda _: provider)
    settings = Settings.from_env(provider="mock", db_path=db)
    with TestClient(create_app(settings, CacheSettings(path=tmp_path / "c.sqlite"))) as client:
        yield client, provider


class TestApi:
    def test_a_follow_up_carries_the_earlier_exchange(
        self, api: tuple[TestClient, MockProvider]
    ) -> None:
        client, provider = api
        first = client.post("/api/ask", json={"question": "how many orders were delivered late?"})
        answer_id = first.json()["answer_id"]
        follow = client.post(
            "/api/ask", json={"question": "and in Riyadh?", "follow_up_of": answer_id}
        ).json()
        assert follow["ok"] and follow["follow_up_of"] == "how many orders were delivered late?"
        last: Any = provider.calls[-1]
        assert [role for role, _ in last.messages] == ["user", "assistant", "user"]
        assert last.messages[1][1] == LATE  # the server's own record, not the browser's

    def test_an_expired_follow_up_is_answered_on_its_own(
        self, api: tuple[TestClient, MockProvider]
    ) -> None:
        client, provider = api
        data = client.post(
            "/api/ask", json={"question": "and in Riyadh?", "follow_up_of": "f" * 32}
        ).json()
        assert data["follow_up_of"] is None
        assert [role for role, _ in provider.calls[-1].messages] == ["user"]

    def test_an_ambiguous_question_is_asked_back_without_calling_the_model(
        self, api: tuple[TestClient, MockProvider]
    ) -> None:
        client, provider = api
        question = "how many orders did each warehouse fulfil?"
        data = client.post("/api/ask", json={"question": question}).json()
        assert data["clarify"]["id"] == "fulfilled" and len(data["clarify"]["options"]) == 2
        assert data["answer_id"] is None and provider.calls == []
        urdu = client.post("/api/ask", json={"question": "ہر گودام نے کتنے آرڈر پورے کیے؟"})
        assert urdu.json()["clarify"]["ask"] == "کون سے آرڈر گنے جائیں؟"

    def test_the_choice_is_answered_and_reported_back(
        self, api: tuple[TestClient, MockProvider]
    ) -> None:
        client, _ = api
        question = "how many orders did each warehouse fulfil?"
        data = client.post(
            "/api/ask",
            json={"question": question, "clarification": {"id": "fulfilled", "option": 0}},
        ).json()
        assert data["ok"] and data["clarified_as"] == "All orders, whatever their status"
        assert data["question"].endswith("(count all orders, whatever their status)")
        bad = client.post(
            "/api/ask",
            json={"question": question, "clarification": {"id": "fulfilled", "option": 5}},
        )
        assert bad.status_code == 422
