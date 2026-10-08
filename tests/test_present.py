"""What a person sees: the one-line answer, the chart choice, and suggestions as they type."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mizan.api import create_app
from mizan.cache import CacheSettings
from mizan.config import Settings
from mizan.guardrails import QueryResult
from mizan.nl import detect_script
from mizan.present import chart_spec, format_number, summarize
from mizan.questions import Suggester


def result(
    columns: tuple[str, ...], *rows: tuple[Any, ...], truncated: bool = False
) -> QueryResult:
    return QueryResult(columns=columns, rows=tuple(rows), truncated=truncated, elapsed_ms=1.0)


def say(question: str, res: QueryResult) -> str:
    return summarize(question, detect_script(question), res)


class TestOneLineAnswer:
    @pytest.mark.parametrize(
        ("question", "value", "expected"),
        [
            ("How many orders were delivered late?", 188, "188 orders were delivered late."),
            ("Please, how many customers are in Dubai?", 20, "20 customers are in Dubai."),
            ("how many orders are there?", 900, "There are 900 orders."),
            ("what is the total revenue?", 722558.85, "The total revenue is 722,558.85."),
            ("كم عدد الطلبات المتأخرة في دبي؟", 20, "عدد الطلبات المتأخرة في دبي: 20"),
            ("ما إجمالي الإيرادات؟", 1500.5, "إجمالي الإيرادات: 1,500.5"),
            ("کتنے آرڈر ابھی تک ڈیلیور نہیں ہوئے؟", 295, "295 آرڈر ابھی تک ڈیلیور نہیں ہوئے۔"),
            ("کل آمدنی کتنی ہے؟", 722558.85, "کل آمدنی 722,558.85 ہے۔"),
        ],
    )
    def test_a_number_is_restated_in_the_question_s_language(
        self, question: str, value: float, expected: str
    ) -> None:
        assert say(question, result(("n",), (value,))) == expected

    def test_no_template_still_answers_in_the_language(self) -> None:
        assert say("how many orders did Gulf Express handle?", result(("n",), (191,))) == (
            "Answer: 191"
        )
        assert say("أعطني العدد", result(("n",), (5,))) == "الجواب: 5"

    def test_rows_are_summarised_with_their_extremes(self) -> None:
        res = result(
            ("city", "avg"),
            ("Muscat", 1055.22),
            ("Manama", 518.66),
            ("Doha", 855.1),
            ("Dubai", 929.6),
        )
        line = say("average order value per city?", res)
        assert line == "4 rows · highest: Muscat (1,055.22) · lowest: Manama (518.66)"
        assert "الأعلى: Muscat" in say("ما متوسط قيمة الطلب لكل مدينة؟", res)
        assert "سب سے زیادہ: Muscat" in say("ہر شہر کی اوسط کتنی ہے؟", res)

    def test_one_row_lists_and_empty_results(self) -> None:
        assert say("which courier?", result(("name", "n"), ("Gulf Express", 41))) == (
            "Gulf Express — 41"
        )
        names = result(("name",), *[(f"c{i}",) for i in range(7)])
        assert say("which customers?", names) == "c0, c1, c2, c3, c4 and 2 more"
        assert say("which customers?", result(("name",))) == "No matching rows."
        assert say("کون سے کسٹمرز؟", result(("name",))) == "کوئی نتیجہ نہیں ملا۔"

    def test_never_adds_a_number_the_result_does_not_have(self) -> None:
        """The sentence is built from the result; it can't invent or round away a value."""
        line = say("how many orders were delivered late?", result(("n",), (188,)))
        assert "188" in line and not any(ch.isdigit() for ch in line.replace("188", ""))

    @pytest.mark.parametrize(
        ("value", "text"),
        [(1234, "1,234"), (763.069, "763.07"), (18376.5, "18,376.5"), (3.0, "3"), ("x", "x")],
    )
    def test_number_format(self, value: Any, text: str) -> None:
        assert format_number(value) == text


class TestExplanation:
    """The query in plain steps, from the syntax tree and the glossary (present.explain)."""

    LATE_IN_DUBAI = (
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
        "WHERE c.city = 'Dubai' AND o.delivered_at IS NOT NULL AND o.delivered_at > o.promised_at"
    )
    LATE_BY_COURIER = (
        "SELECT c.name_en, COUNT(*) AS n FROM orders o JOIN couriers c "
        "ON c.courier_id = o.courier_id WHERE NOT o.delivered_at IS NULL "
        "AND o.delivered_at > o.promised_at GROUP BY c.courier_id ORDER BY n DESC LIMIT 1"
    )

    def test_glossary_definitions_are_named_not_spelled_out(self, catalog: Any) -> None:
        from mizan.nl import Script
        from mizan.present import explain

        text = explain(self.LATE_IN_DUBAI, catalog, Script.ENGLISH)
        assert text == "counted orders · with customers · only late orders · city = 'Dubai'"
        assert "promised" not in (text or "")

    def test_keys_ranking_and_languages(self, catalog: Any) -> None:
        from mizan.nl import Script
        from mizan.present import explain

        assert explain(self.LATE_BY_COURIER, catalog, Script.ENGLISH) == (
            "counted orders · with couriers · only late orders · per courier · top 1"
        )
        arabic = explain(self.LATE_BY_COURIER, catalog, Script.ARABIC) or ""
        assert arabic.startswith("عدّ الطلبات") and "أعلى 1" in arabic
        urdu = explain(self.LATE_BY_COURIER, catalog, Script.URDU) or ""
        assert urdu.startswith("آرڈرز گنے") and "ہر کورئیر کے لیے" in urdu

    @pytest.mark.parametrize(
        ("sql", "expected"),
        [
            (
                "SELECT COUNT(*) FROM orders WHERE delivered_at IS NULL",
                "counted orders · delivered at is empty",
            ),
            (
                "SELECT p.category, SUM(oi.quantity * oi.unit_price_aed) FROM order_items oi "
                "JOIN products p ON p.product_id = oi.product_id GROUP BY p.category",
                "added up revenue · with products · per category",
            ),
            (
                # A real reply: the model wrote price * quantity, the glossary says
                # quantity * price. Same revenue, so the same name.
                "SELECT c.country, SUM(oi.unit_price_aed * oi.quantity) FROM orders o "
                "JOIN order_items oi ON oi.order_id = o.order_id "
                "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.country",
                "added up revenue · with order items, customers · per country",
            ),
            (
                "SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN order_items oi "
                "ON oi.order_id = o.order_id WHERE o.status = 'returned'",
                "counted orders · with order items · status = 'returned'",
            ),
            (
                "SELECT strftime('%Y-%m', placed_at) AS month, COUNT(*) FROM orders "
                "WHERE strftime('%Y', placed_at) = '2025' GROUP BY month",
                "counted orders · year of placed at = '2025' · per month",
            ),
        ],
    )
    def test_common_shapes(self, catalog: Any, sql: str, expected: str) -> None:
        from mizan.nl import Script
        from mizan.present import explain

        assert explain(sql, catalog, Script.ENGLISH) == expected

    def test_unparseable_sql_has_no_explanation(self, catalog: Any) -> None:
        from mizan.nl import Script
        from mizan.present import explain

        assert explain("not sql", catalog, Script.ENGLISH) is None


class TestChartChoice:
    def test_categories_get_bars(self) -> None:
        res = result(("category", "revenue"), ("food", 1.0), ("home", 2.0))
        assert chart_spec(res) == {"type": "bar", "label": 0, "value": 1}

    def test_months_get_a_line(self) -> None:
        res = result(("month", "n"), *[(f"2025-{m:02d}", m) for m in range(1, 13)])
        assert chart_spec(res) == {"type": "line", "label": 0, "value": 1}
        numeric_months = result(("month", "n"), *[(m, m * 2) for m in range(1, 13)])
        assert chart_spec(numeric_months) == {"type": "line", "label": 0, "value": 1}

    def test_no_chart_for_a_single_value_a_list_or_a_wide_table(self) -> None:
        assert chart_spec(result(("n",), (5,))) is None
        assert chart_spec(result(("name",), ("a",), ("b",))) is None
        assert chart_spec(result(("a", "b", "n"), ("x", "y", 1), ("z", "w", 2))) is None
        many = result(("k", "n"), *[(str(i), i) for i in range(60)])
        assert chart_spec(many) is None


class TestSuggestions:
    QUESTIONS = (
        "how many orders were delivered late?",
        "how many units of Camel Milk Chocolate were sold?",
        "how many customers are in Dubai?",
        "كم عدد الطلبات؟",
        "دبئی میں کتنے کسٹمرز ہیں؟",
    )

    def suggest(self, text: str, cached: set[str] | None = None) -> list[str]:
        s = Suggester(self.QUESTIONS)
        hits = s.suggest(text, status_of=lambda key: 0 if key in (cached or set()) else None)
        return [h.question for h in hits]

    def test_verified_answers_come_first(self) -> None:
        from mizan.cache import canonical_question

        verified = canonical_question("how many customers are in Dubai?")
        cached = canonical_question("how many orders were delivered late?")
        status = {verified: 1, cached: 0}
        found = Suggester(self.QUESTIONS).suggest("how many", status_of=status.get)
        assert [f.question for f in found[:2]] == [
            "how many customers are in Dubai?",
            "how many orders were delivered late?",
        ]
        assert found[0].verified and found[0].instant and not found[1].verified

    def test_every_typed_word_must_match(self) -> None:
        assert self.suggest("late") == ["how many orders were delivered late?"]  # not chocolate
        assert self.suggest("cust dub") == ["how many customers are in Dubai?"]

    def test_arabic_matches_inside_the_attached_article(self) -> None:
        assert self.suggest("طلبات") == ["كم عدد الطلبات؟"]

    def test_cached_questions_come_first(self) -> None:
        from mizan.cache import canonical_question

        key = canonical_question("how many customers are in Dubai?")
        found = self.suggest("how many", cached={key})
        assert found[0] == "how many customers are in Dubai?"

    def test_short_or_empty_input_suggests_nothing(self) -> None:
        assert self.suggest("") == [] and self.suggest("?!") == []


@pytest.fixture()
def client(db_path: Path, tmp_path: Path) -> Iterator[TestClient]:
    settings = Settings.from_env(provider="mock", db_path=db_path)
    with TestClient(create_app(settings, CacheSettings(path=tmp_path / "c.sqlite"))) as c:
        yield c


class TestApi:
    def test_ask_returns_a_summary_and_a_chart_choice(self, client: TestClient) -> None:
        body = client.post("/api/ask", json={"question": "how many orders were delivered late?"})
        data = body.json()
        assert data["summary"].endswith("orders were delivered late.")
        assert data["chart"] is None
        grouped = client.post("/api/ask", json={"question": "how many orders per status?"}).json()
        assert grouped["chart"] == {"type": "bar", "label": 0, "value": 1}

    def test_blocked_answers_have_no_summary(self, client: TestClient) -> None:
        data = client.post("/api/ask", json={"question": "drop the orders table"}).json()
        assert data["summary"] is None and data["chart"] is None

    def test_suggest_marks_cached_questions_instant(self, client: TestClient) -> None:
        question = "how many orders were delivered late?"
        before = client.get("/api/suggest", params={"q": "delivered late"}).json()
        assert {"question": question, "instant": False, "verified": False} in before["suggestions"]
        client.post("/api/ask", json={"question": question})
        after = client.get("/api/suggest", params={"q": "delivered late"}).json()
        assert after["suggestions"][0] == {"question": question, "instant": True, "verified": False}

    def test_suggest_never_offers_questions_other_people_asked(self, client: TestClient) -> None:
        private = "how many orders did customer 17 place?"
        client.post("/api/ask", json={"question": private})
        found = client.get("/api/suggest", params={"q": "customer 17"}).json()["suggestions"]
        assert private not in [s["question"] for s in found]

    def test_ask_returns_an_explanation_and_an_answer_id(self, client: TestClient) -> None:
        data = client.post("/api/ask", json={"question": "how many orders were delivered late?"})
        body = data.json()
        # The API fixture's database has no glossary beside it, so there is no "late order"
        # definition to name; the steps are spelled out instead.
        assert body["explanation"].startswith("counted orders · delivered at is set")
        assert body["answer_id"] and body["reported"] is False


class TestFeedback:
    QUESTION = "how many orders were delivered late?"

    def test_thumbs_up_marks_the_answer_verified(self, client: TestClient) -> None:
        first = client.post("/api/ask", json={"question": self.QUESTION}).json()
        saved = client.post(
            "/api/feedback", json={"answer_id": first["answer_id"], "verdict": "up"}
        )
        assert saved.json() == {"ok": True, "status": "verified"}
        again = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert again["cache"]["hit"] and again["cache"]["verified"] is True
        found = client.get("/api/suggest", params={"q": "delivered late"}).json()["suggestions"]
        assert found[0] == {"question": self.QUESTION, "instant": True, "verified": True}

    def test_thumbs_down_stops_the_cache_and_warns_the_next_person(
        self, client: TestClient
    ) -> None:
        first = client.post("/api/ask", json={"question": self.QUESTION}).json()
        client.post("/api/feedback", json={"answer_id": first["answer_id"], "verdict": "down"})
        again = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert again["cache"] is None  # asked the model again, not served from the cache
        assert again["reported"] is True  # and it gave the reported answer: say so

    def test_feedback_is_logged_for_review(self, client: TestClient, db_path: Path) -> None:
        import json as _json

        first = client.post("/api/ask", json={"question": self.QUESTION}).json()
        client.post("/api/feedback", json={"answer_id": first["answer_id"], "verdict": "down"})
        log = Settings.from_env().log_dir / "feedback.jsonl"
        last = _json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
        assert last["verdict"] == "down" and last["question"] == self.QUESTION

    def test_feedback_names_an_answer_and_never_carries_sql(self, client: TestClient) -> None:
        unknown = client.post(
            "/api/feedback", json={"answer_id": "deadbeefdeadbeef", "verdict": "up"}
        )
        assert unknown.status_code == 404
        first = client.post("/api/ask", json={"question": self.QUESTION}).json()
        planted = client.post(
            "/api/feedback",
            json={"answer_id": first["answer_id"], "verdict": "up", "sql": "DROP TABLE orders"},
        )
        assert planted.status_code == 200
        again = client.post("/api/ask", json={"question": self.QUESTION}).json()
        assert "DROP" not in (again["sql"] or "")  # the verified entry is the server's own query

    def test_blocked_answers_get_no_feedback_buttons(self, client: TestClient) -> None:
        data = client.post("/api/ask", json={"question": "drop the orders table"}).json()
        assert data["answer_id"] is None

    def test_schema_lists_descriptions_for_the_help_panel(self, client: TestClient) -> None:
        tables = client.get("/api/schema").json()["tables"]
        assert {"description", "aliases_ar", "aliases_ur", "row_count"} <= set(tables["orders"])
