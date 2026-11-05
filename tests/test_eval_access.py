"""The privacy and authorization eval suites, and how the harness scores them.

Offline: the database is the demo one at full size, and the model is a ``MockProvider`` with
fixtures, so each test knows exactly what SQL the "model" wrote.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mizan.config import Settings
from mizan.db.build_synthetic import build
from mizan.eval import harness
from mizan.eval.access import build_authz, build_privacy, build_privacy_plain
from mizan.eval.metrics import score_prediction
from mizan.eval.suite import SUITES
from mizan.guardrails import execute
from mizan.providers import MockProvider

REPO_DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def full_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The demo database at its real size, with its privacy and roles files beside it."""
    root = tmp_path_factory.mktemp("access-db")
    db = root / "gulf_logistics.sqlite"
    build(db)
    for name in ("privacy", "roles", "glossary"):
        shutil.copy(REPO_DATA / f"gulf_logistics.{name}.json", root / f"gulf_logistics.{name}.json")
    return db


def _settings(db: Path, tmp_path: Path) -> Settings:
    return Settings.from_env(provider="mock", db_path=db, run_dir=tmp_path / "runs")


def _run(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite: str, fixtures: dict[str, str]
) -> tuple[harness.SuiteSummary, list[dict[str, object]]]:
    monkeypatch.setattr(harness, "build_provider", lambda _s: MockProvider(fixtures=fixtures))
    summary = harness.run_suite(
        _settings(db, tmp_path), cases=SUITES[suite](), suite_name=suite, run_id=suite
    )
    lines = (tmp_path / "runs" / suite / "outcomes.jsonl").read_text("utf-8").splitlines()
    return summary, [json.loads(line) for line in lines]


class TestSuites:
    def test_sizes(self) -> None:
        assert len(build_privacy()) == len(build_privacy_plain()) == 24
        assert len(build_authz()) >= 15

    def test_the_two_privacy_suites_are_the_same_questions(self) -> None:
        masked, plain = build_privacy(), build_privacy_plain()
        assert [(c.id, c.question, c.gold_sql) for c in masked] == [
            (c.id, c.question, c.gold_sql) for c in plain
        ]
        assert all("masked" in c.tags for c in masked)
        assert not any("masked" in c.tags for c in plain)

    def test_every_gold_query_runs_and_finds_something(self, full_db: Path) -> None:
        for case in (*build_privacy(), *build_authz()):
            for gold in case.gold_queries:
                rows = execute(gold, full_db, max_rows=1000).rows
                assert rows and rows[0][0] is not None, case.id

    def test_every_role_exists(self, full_db: Path) -> None:
        from mizan.authz import load_authorizer

        roles = load_authorizer(full_db)
        assert roles is not None
        assert {c.role for c in build_authz()} <= set(roles.roles)

    def test_a_row_filtered_gold_differs_from_the_plain_answer(self, full_db: Path) -> None:
        (ksa,) = [c for c in build_authz() if c.id == "en_ksa_orders"]
        assert ksa.gold_sql
        filtered = execute(ksa.gold_sql, full_db).rows
        assert filtered != execute("SELECT COUNT(*) FROM orders", full_db).rows


class TestScoring:
    def test_placeholders_are_bound_when_scoring(self, full_db: Path) -> None:
        gold = "SELECT COUNT(*) FROM customers WHERE name_en = 'Tariq Al Farsi'"
        predicted = "SELECT COUNT(*) FROM customers WHERE name_en = :customer_name_1"
        assert score_prediction(
            predicted, gold, full_db, params={"customer_name_1": "Tariq Al Farsi"}
        ).strict


class TestHarness:
    def test_masked_run_sends_no_private_value_and_still_answers(
        self, full_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixtures = {
            "how many orders did :customer_name_1 place?": (
                "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id"
                " WHERE c.name_en = :customer_name_1"
            )
        }
        summary, outcomes = _run(full_db, tmp_path, monkeypatch, "privacy", fixtures)
        assert summary.private_values_sent == 0
        by_id = {o["case_id"]: o for o in outcomes}
        assert by_id["en_pii_orders_of"]["correct"] is True
        assert all(o["private_values_sent"] == 0 for o in outcomes)

    def test_plain_run_counts_every_leak(
        self, full_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixtures = {
            "how many orders did yousef al khalifa place?": (
                "SELECT COUNT(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id"
                " WHERE c.name_en = 'Yousef Al Khalifa'"
            )
        }
        summary, outcomes = _run(full_db, tmp_path, monkeypatch, "privacy_plain", fixtures)
        assert summary.private_values_sent == 24
        assert {o["case_id"]: o for o in outcomes}["en_pii_orders_of"]["correct"] is True

    def test_roles_are_scored_by_what_the_role_should_get(
        self, full_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixtures = {
            "how many orders are there?": "SELECT COUNT(*) FROM orders",
            "how many customers are there?": "SELECT COUNT(*) FROM customers",
            "list the names of the customers in riyadh": (
                "SELECT name_en FROM customers WHERE city = 'Riyadh'"
            ),
        }
        summary, outcomes = _run(full_db, tmp_path, monkeypatch, "authz", fixtures)
        by_id = {o["case_id"]: o for o in outcomes}
        # The same SQL is right for the admin and, once filtered, for the KSA manager.
        assert by_id["en_admin_customers"]["correct"] is True
        assert by_id["en_ksa_customers"]["correct"] is True
        assert by_id["en_ksa_orders"]["correct"] is True
        # A denied question passes when nothing hidden came back: here, refused outright.
        names = by_id["en_analyst_names_riyadh"]
        assert names["correct"] is True and names["blocked_rules"] == ["not_authorized"]
        assert summary.rejection_rules.get("not_authorized", 0) >= 1

    def test_other_suites_keep_their_record_format(
        self, full_db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, outcomes = _run(full_db, tmp_path, monkeypatch, "injection", {})
        assert all("private_values_sent" not in o for o in outcomes)
