"""Authorization end to end: the pipeline and the server with a roles file.

test_authz.py tests each layer on its own. Here a request comes in with a bearer token and
these tests check what comes back: who may ask at all (401), which rows and columns an
answer holds, what the model was shown, and that one requester's answers, cache entries and
follow-ups never reach another's (DECISIONS.md D45).

The model is a ``MockProvider`` with fixtures, so every role's engine gets the same SQL for
the same question, and any difference in the answer is the authorization's doing.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mizan.api import create_app
from mizan.authz import ADMIN, Authorizer
from mizan.cache import CacheSettings
from mizan.config import Settings
from mizan.generate import TextToSQL
from mizan.providers import MockProvider
from mizan.schema import Catalog
from test_authz import CONFIG

ALL_CUSTOMERS = "how many customers are there?"
IN_DUBAI = "how many customers are in Dubai?"
NAMES = "list the customer names"
LATE = "how many orders were delivered late?"
FIXTURES = {
    ALL_CUSTOMERS: "SELECT COUNT(*) FROM customers",
    IN_DUBAI: "SELECT COUNT(*) FROM customers WHERE city = 'Dubai'",
    NAMES: "SELECT name_en FROM customers",
    LATE: "SELECT COUNT(*) FROM orders WHERE delivered_at > promised_at",
}

TOKENS = {
    "admin": "demo-admin",
    "courier_ops": "demo-courier-ops",
    "ksa": "demo-ksa",
    "analyst": "demo-analyst",
}


def _count(db: Path, sql: str, *params: object) -> int:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return int(conn.execute(sql, params).fetchone()[0])


def _auth(who: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKENS[who]}"}


# ------------------------------------------------------------------------------ pipeline


class TestPipeline:
    def test_a_row_filtered_role_gets_only_its_rows(
        self, catalog: Catalog, settings: Settings, db_path: Path
    ) -> None:
        ksa = Authorizer.from_dict(CONFIG).authenticate("Bearer demo-ksa")
        engine = TextToSQL(catalog, MockProvider(fixtures=FIXTURES), settings, role=ksa.role)
        answer = engine.ask(ALL_CUSTOMERS, attrs=ksa.attrs)
        assert answer.ok, answer.error
        assert answer.result is not None
        expected = _count(
            db_path, "SELECT COUNT(*) FROM customers WHERE country = ?", "Saudi Arabia"
        )
        assert answer.result.rows == ((expected,),)
        # The answer shows the query the model wrote, not the filter around it.
        assert answer.sql is not None and "_user_country" not in answer.sql

    def test_the_model_only_sees_the_roles_schema(
        self, catalog: Catalog, settings: Settings
    ) -> None:
        ops = Authorizer.from_dict(CONFIG).authenticate("Bearer demo-courier-ops")
        provider = MockProvider(fixtures=FIXTURES)
        engine = TextToSQL(catalog, provider, settings, role=ops.role)
        answer = engine.ask(ALL_CUSTOMERS)
        assert "customers" not in provider.calls[0].system
        assert not answer.ok and "not_authorized" in answer.guardrail.rules_fired  # type: ignore[union-attr]
        assert len(provider.calls) == 1  # never repaired

    def test_without_a_role_nothing_changes(self, catalog: Catalog, settings: Settings) -> None:
        plain = TextToSQL(catalog, MockProvider(fixtures=FIXTURES), settings)
        admin = TextToSQL(catalog, MockProvider(fixtures=FIXTURES), settings, role=ADMIN.role)
        assert plain.system_prompt == admin.system_prompt


# ---------------------------------------------------------------------------------- API


@pytest.fixture()
def server(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, MockProvider, Path]]:
    db = tmp_path / "shop.sqlite"
    shutil.copy(db_path, db)
    (tmp_path / "shop.roles.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    provider = MockProvider(fixtures=FIXTURES)
    monkeypatch.setattr("mizan.api.build_provider", lambda _cfg: provider)
    settings = Settings.from_env(provider="mock", db_path=db, log_dir=tmp_path / "logs")
    with TestClient(create_app(settings, CacheSettings(path=tmp_path / "c.sqlite"))) as client:
        yield client, provider, db


def _ask(client: TestClient, who: str, question: str, **extra: Any) -> dict[str, Any]:
    response = client.post("/api/ask", json={"question": question, **extra}, headers=_auth(who))
    assert response.status_code == 200, response.text
    return dict(response.json())


class TestAuthentication:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("post", "/api/ask", {"question": ALL_CUSTOMERS}),
            ("post", "/api/feedback", {"answer_id": "deadbeefdeadbeef", "verdict": "up"}),
            ("get", "/api/suggest", None),
            ("get", "/api/schema", None),
        ],
    )
    @pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}])
    def test_every_data_endpoint_needs_a_token(
        self,
        server: tuple[TestClient, MockProvider, Path],
        method: str,
        path: str,
        body: dict[str, Any] | None,
        headers: dict[str, str],
    ) -> None:
        client, provider, _ = server
        response = getattr(client, method)(
            path, headers=headers, **({"json": body} if body else {})
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate", "").lower().startswith("bearer")
        assert provider.calls == []  # refused before the model was asked anything

    def test_health_and_the_page_need_no_token(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, _ = server
        assert client.get("/api/health").status_code == 200
        assert client.get("/").status_code == 200


class TestWhatEachRoleGets:
    def test_rows(self, server: tuple[TestClient, MockProvider, Path]) -> None:
        client, _, db = server
        admin = _ask(client, "admin", ALL_CUSTOMERS)
        ksa = _ask(client, "ksa", ALL_CUSTOMERS)
        assert admin["result"]["rows"] == [[_count(db, "SELECT COUNT(*) FROM customers")]]
        assert ksa["result"]["rows"] == [
            [_count(db, "SELECT COUNT(*) FROM customers WHERE country = ?", "Saudi Arabia")]
        ]

    def test_tables(self, server: tuple[TestClient, MockProvider, Path]) -> None:
        client, _, _ = server
        body = _ask(client, "courier_ops", ALL_CUSTOMERS)
        assert body["ok"] is False
        assert [v["rule"] for v in body["violations"]] == ["not_authorized"]
        assert _ask(client, "courier_ops", LATE)["ok"] is True

    def test_columns(self, server: tuple[TestClient, MockProvider, Path]) -> None:
        client, _, _ = server
        assert _ask(client, "analyst", NAMES)["ok"] is False
        assert _ask(client, "analyst", ALL_CUSTOMERS)["ok"] is True
        assert _ask(client, "admin", NAMES)["ok"] is True

    def test_several_samples_keep_the_filter(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, db = server
        body = _ask(client, "ksa", ALL_CUSTOMERS, samples=2)
        assert body["result"]["rows"] == [
            [_count(db, "SELECT COUNT(*) FROM customers WHERE country = ?", "Saudi Arabia")]
        ]

    def test_schema_is_the_roles_slice(self, server: tuple[TestClient, MockProvider, Path]) -> None:
        client, _, _ = server
        ops = client.get("/api/schema", headers=_auth("courier_ops")).json()
        assert "customers" not in ops["tables"] and "customers" not in ops["ddl"]
        analyst = client.get("/api/schema", headers=_auth("analyst")).json()
        assert "name_en" not in analyst["tables"]["customers"]["columns"]
        admin = client.get("/api/schema", headers=_auth("admin")).json()
        assert "customers" in admin["tables"]


class TestSeparation:
    def test_a_cached_answer_is_filtered_again_for_each_requester(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, db = server
        _ask(client, "admin", ALL_CUSTOMERS)
        again = _ask(client, "admin", ALL_CUSTOMERS)
        assert again["cache"] and again["cache"]["hit"]
        ksa = _ask(client, "ksa", ALL_CUSTOMERS)
        assert ksa["result"]["rows"] == [
            [_count(db, "SELECT COUNT(*) FROM customers WHERE country = ?", "Saudi Arabia")]
        ]
        ksa_again = _ask(client, "ksa", ALL_CUSTOMERS)
        assert ksa_again["result"]["rows"] == ksa["result"]["rows"]

    def test_a_follow_up_cannot_use_another_requesters_answer(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, _ = server
        first = _ask(client, "admin", NAMES)
        follow = _ask(client, "analyst", "and in Dubai?", follow_up_of=first["answer_id"])
        assert follow["follow_up_of"] is None
        own = _ask(client, "admin", "and in Dubai?", follow_up_of=first["answer_id"])
        assert own["follow_up_of"] == NAMES

    def test_feedback_on_another_requesters_answer_is_refused(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, _ = server
        first = _ask(client, "admin", ALL_CUSTOMERS)
        vote = {"answer_id": first["answer_id"], "verdict": "down"}
        assert client.post("/api/feedback", json=vote, headers=_auth("ksa")).status_code == 404
        assert client.post("/api/feedback", json=vote, headers=_auth("admin")).status_code == 200

    def test_feedback_records_the_role(self, server: tuple[TestClient, MockProvider, Path]) -> None:
        client, _, db = server
        first = _ask(client, "ksa", ALL_CUSTOMERS)
        client.post(
            "/api/feedback",
            json={"answer_id": first["answer_id"], "verdict": "up"},
            headers=_auth("ksa"),
        )
        log = db.parent / "logs" / "feedback.jsonl"
        record = json.loads(log.read_text("utf-8").splitlines()[-1])
        assert record["role"] == "ksa_manager"
        assert "demo" not in json.dumps(record)

    def test_suggestions_hide_questions_the_role_cannot_ask(
        self, server: tuple[TestClient, MockProvider, Path]
    ) -> None:
        client, _, _ = server
        _ask(client, "admin", IN_DUBAI)  # now cached, so its SQL is known

        def offered(who: str) -> list[str]:
            found = client.get(
                "/api/suggest", params={"q": "customers in Dubai"}, headers=_auth(who)
            )
            return [s["question"] for s in found.json()["suggestions"]]

        assert IN_DUBAI in offered("admin")
        assert IN_DUBAI not in offered("courier_ops")
