"""Authorization: who is asking decides which tables, columns and rows they can read.

A role (``<db>.roles.json``) limits what a requester sees, enforced in layers that don't
depend on each other (DECISIONS.md D45):

1. **Prompt slice**: the model is only shown the tables and columns the role can read.
2. **Validator**: a query that touches anything else is rejected as ``not_authorized``, and
   that rejection is final: the repair loop must not help anyone probe the schema.
3. **Row filters**: after validation, every reference to a filtered table is rewritten to a
   filtered subquery, with the requester's own attributes bound as ``:_user_*`` parameters
   that the model can't write or override.
4. **Runtime**: the SQLite connection refuses denied reads (task 8, test_security.py).

Identity comes only from a bearer token, looked up by its SHA-256. Nothing in a request body
says who is asking.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from mizan.authz import (
    ADMIN,
    Authorizer,
    Principal,
    Unauthorized,
    apply_row_filters,
    load_authorizer,
)
from mizan.errors import ExecutionError
from mizan.generate.prompt import build_system_prompt
from mizan.generate.repair import diagnose
from mizan.guardrails import execute, validate
from mizan.schema import Catalog

REPO_ROOT = Path(__file__).resolve().parent.parent
ROLES_FILE = REPO_ROOT / "data" / "gulf_logistics.roles.json"

KSA_ORDERS = "warehouse_id IN (SELECT warehouse_id FROM warehouses WHERE country = :_user_country)"
CONFIG: dict[str, Any] = {
    "roles": {
        "admin": {"tables": "*"},
        "courier_ops": {"tables": {"orders": "*", "couriers": "*", "warehouses": "*"}},
        "ksa_manager": {
            "tables": "*",
            "row_filters": {"orders": KSA_ORDERS, "customers": "country = :_user_country"},
        },
        "analyst": {"tables": "*", "deny_columns": ["customers.name_en", "customers.name_ar"]},
    },
    "tokens": {
        hashlib.sha256(token.encode()).hexdigest(): spec
        for token, spec in {
            "demo-admin": {"role": "admin"},
            "demo-courier-ops": {"role": "courier_ops"},
            "demo-ksa": {"role": "ksa_manager", "attrs": {"country": "Saudi Arabia"}},
            "demo-analyst": {"role": "analyst"},
        }.items()
    },
}


@pytest.fixture(scope="module")
def authorizer() -> Authorizer:
    return Authorizer.from_dict(CONFIG)


def _who(authorizer: Authorizer, token: str) -> Principal:
    return authorizer.authenticate(f"Bearer {token}")


def _scalar(db: Path, sql: str, *params: object) -> Any:
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        return conn.execute(sql, params).fetchone()[0]


def _run(sql: str, principal: Principal, db: Path) -> tuple[tuple[Any, ...], ...]:
    """Row-filter ``sql`` for ``principal`` and run it the way the pipeline will."""
    rewritten, params = apply_row_filters(sql, principal)
    return execute(rewritten, db, params=params).rows


# ------------------------------------------------------------------------------ identity


class TestAuthentication:
    def test_a_token_names_its_role_and_attributes(self, authorizer: Authorizer) -> None:
        who = _who(authorizer, "demo-ksa")
        assert who.role.name == "ksa_manager"
        assert who.attrs == {"country": "Saudi Arabia"}

    def test_tokens_are_stored_hashed(self) -> None:
        # The config file must never hold a usable token.
        assert "demo-ksa" not in json.dumps(CONFIG)

    def test_the_subject_identifies_a_token_without_revealing_it(
        self, authorizer: Authorizer
    ) -> None:
        a, b = _who(authorizer, "demo-ksa"), _who(authorizer, "demo-analyst")
        assert a.subject != b.subject
        assert a.subject == _who(authorizer, "demo-ksa").subject
        assert "demo" not in a.subject

    @pytest.mark.parametrize(
        "header",
        [None, "", "Bearer", "Bearer ", "Bearer wrong-token", "Basic demo-ksa", "demo-ksa"],
    )
    def test_anything_but_a_known_bearer_token_is_refused(
        self, authorizer: Authorizer, header: str | None
    ) -> None:
        with pytest.raises(Unauthorized):
            authorizer.authenticate(header)

    def test_admin_reads_everything(self) -> None:
        assert ADMIN.role.can_read("customers", "name_en")
        assert ADMIN.role.row_filters == {}


class TestConfig:
    def test_the_committed_demo_roles_load(self) -> None:
        loaded = Authorizer.from_file(ROLES_FILE)
        assert {"admin", "courier_ops", "ksa_manager", "analyst"} <= set(loaded.roles)

    def test_no_roles_file_means_no_authorizer(self, tmp_path: Path) -> None:
        # A single-user local tool without a roles file keeps today's behaviour.
        assert load_authorizer(tmp_path / "shop.sqlite") is None

    def test_a_roles_file_beside_the_database_is_loaded(self, tmp_path: Path) -> None:
        (tmp_path / "shop.roles.json").write_text(json.dumps(CONFIG), encoding="utf-8")
        loaded = load_authorizer(tmp_path / "shop.sqlite")
        assert loaded is not None and "analyst" in loaded.roles

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"tokens": {"ab" * 32: {"role": "nobody"}}}, "nobody"),
            # A filter parameter must be a requester attribute, written :_user_<name>.
            ({"roles": {"r": {"tables": "*", "row_filters": {"orders": "status = :s"}}}}, ":s"),
            # And every token of that role must carry the attribute.
            (
                {
                    "roles": {
                        "r": {
                            "tables": "*",
                            "row_filters": {"customers": "country = :_user_country"},
                        }
                    },
                    "tokens": {"cd" * 32: {"role": "r"}},
                },
                "country",
            ),
            (
                {"roles": {"r": {"tables": "*", "row_filters": {"orders": "1; DROP TABLE x"}}}},
                "orders",
            ),
            ({"roles": {"r": {"tables": {"orders": "*"}, "deny_columns": ["orders"]}}}, "orders"),
        ],
    )
    def test_a_broken_config_fails_loudly(self, change: dict[str, Any], message: str) -> None:
        config = {"roles": {"admin": {"tables": "*"}}, "tokens": {}, **change}
        with pytest.raises(ValueError, match=message):
            Authorizer.from_dict(config)


# ---------------------------------------------------------------------------- 1. prompt


class TestPromptSlice:
    def test_a_role_without_a_table_never_sees_it(
        self, catalog: Catalog, authorizer: Authorizer
    ) -> None:
        sliced = catalog.restricted(_who(authorizer, "demo-courier-ops").role)
        assert not sliced.has_table("customers")
        assert sliced.has_table("orders") and sliced.has_column("orders", "courier_id")
        prompt = build_system_prompt(sliced)
        assert "customers" not in prompt
        # Join conditions to a table the role can't read go too.
        assert all("customers" not in j for j in sliced.join_conditions())

    def test_a_denied_column_is_left_out(self, catalog: Catalog, authorizer: Authorizer) -> None:
        sliced = catalog.restricted(_who(authorizer, "demo-analyst").role)
        assert sliced.has_column("customers", "city")
        assert not sliced.has_column("customers", "name_en")
        assert not sliced.has_column("customers", "name_ar")
        assert sliced.has_column("couriers", "name_en")  # only the customers' names are private

    def test_admin_sees_the_catalog_unchanged(self, catalog: Catalog) -> None:
        assert build_system_prompt(catalog.restricted(ADMIN.role)) == build_system_prompt(catalog)


# ------------------------------------------------------------------------- 2. validator


class TestValidator:
    @pytest.mark.parametrize(
        ("token", "sql"),
        [
            ("demo-courier-ops", "SELECT COUNT(*) FROM customers"),
            (
                "demo-courier-ops",
                "SELECT o.order_id FROM orders o WHERE o.customer_id IN "
                "(SELECT customer_id FROM customers WHERE city = 'Doha')",
            ),
            ("demo-analyst", "SELECT name_en FROM customers"),
            ("demo-analyst", "SELECT c.name_ar FROM orders o JOIN customers c USING (customer_id)"),
            ("demo-analyst", "SELECT * FROM customers"),
            ("demo-analyst", "SELECT c.* FROM customers c"),
            ("demo-analyst", "SELECT city FROM customers WHERE name_en LIKE 'T%'"),
        ],
    )
    def test_reading_outside_the_role_is_not_authorized(
        self, catalog: Catalog, authorizer: Authorizer, token: str, sql: str
    ) -> None:
        report = validate(sql, catalog, role=_who(authorizer, token).role)
        assert "not_authorized" in report.rules_fired, report.violations

    @pytest.mark.parametrize(
        ("token", "sql"),
        [
            (
                "demo-courier-ops",
                "SELECT c.name_en, COUNT(*) FROM orders o JOIN couriers c "
                "ON c.courier_id = o.courier_id GROUP BY c.name_en",
            ),
            ("demo-analyst", "SELECT city, COUNT(*) FROM customers GROUP BY city"),
            ("demo-analyst", "SELECT COUNT(*) FROM customers"),
            ("demo-analyst", "SELECT name_en FROM couriers"),
        ],
    )
    def test_reading_inside_the_role_is_allowed(
        self, catalog: Catalog, authorizer: Authorizer, token: str, sql: str
    ) -> None:
        report = validate(sql, catalog, role=_who(authorizer, token).role)
        assert report.ok, report.violations

    def test_the_violation_names_what_the_role_cannot_read(
        self, catalog: Catalog, authorizer: Authorizer
    ) -> None:
        report = validate(
            "SELECT name_en FROM customers", catalog, role=_who(authorizer, "demo-analyst").role
        )
        (violation,) = [v for v in report.violations if v.rule == "not_authorized"]
        assert "customers.name_en" in violation.detail and "analyst" in violation.detail

    def test_not_authorized_is_never_repaired(
        self, catalog: Catalog, authorizer: Authorizer
    ) -> None:
        report = validate(
            "SELECT COUNT(*) FROM customers",
            catalog,
            role=_who(authorizer, "demo-courier-ops").role,
        )
        assert diagnose(report, None, [], catalog) is None

    def test_no_role_means_admin(self, catalog: Catalog) -> None:
        assert validate("SELECT name_en FROM customers", catalog).ok

    def test_the_model_cannot_write_the_requesters_attributes(self, catalog: Catalog) -> None:
        report = validate(
            "SELECT COUNT(*) FROM customers WHERE country = :_user_country",
            catalog,
            placeholders={"customer_name_1"},
        )
        assert report.rules_fired == ["unknown_placeholder"]


# ----------------------------------------------------------------------- 3. row filters


class TestRowFilters:
    def test_a_plain_table(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        expected = _scalar(
            db_path, "SELECT COUNT(*) FROM customers WHERE country = ?", "Saudi Arabia"
        )
        assert _run("SELECT COUNT(*) FROM customers", ksa, db_path) == ((expected,),)

    def test_an_alias_and_qualified_columns(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        rows = _run("SELECT DISTINCT c.country FROM customers c", ksa, db_path)
        assert rows == (("Saudi Arabia",),)

    def test_unqualified_and_table_qualified_columns(
        self, authorizer: Authorizer, db_path: Path
    ) -> None:
        ksa = _who(authorizer, "demo-ksa")
        rows = _run("SELECT DISTINCT customers.country FROM customers", ksa, db_path)
        assert rows == (("Saudi Arabia",),)
        assert _run("SELECT DISTINCT country FROM customers", ksa, db_path) == (("Saudi Arabia",),)

    def test_both_sides_of_a_join(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        sql = (
            "SELECT DISTINCT c.country, w.country FROM orders o "
            "JOIN customers c ON c.customer_id = o.customer_id "
            "JOIN warehouses w ON w.warehouse_id = o.warehouse_id"
        )
        rows = _run(sql, ksa, db_path)
        assert rows and all(r == ("Saudi Arabia", "Saudi Arabia") for r in rows)

    def test_a_subquery(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        sql = "SELECT COUNT(*) FROM couriers WHERE courier_id IN (SELECT courier_id FROM orders)"
        expected = _scalar(
            db_path,
            "SELECT COUNT(*) FROM couriers WHERE courier_id IN (SELECT courier_id FROM orders "
            "WHERE warehouse_id IN (SELECT warehouse_id FROM warehouses WHERE country = ?))",
            "Saudi Arabia",
        )
        assert _run(sql, ksa, db_path) == ((expected,),)

    def test_a_cte_cannot_launder_the_table(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        sql = "WITH everyone AS (SELECT * FROM customers) SELECT DISTINCT country FROM everyone"
        assert _run(sql, ksa, db_path) == (("Saudi Arabia",),)

    def test_a_cte_named_after_the_table_cannot_launder_it(
        self, authorizer: Authorizer, db_path: Path
    ) -> None:
        # SQLite refuses this query outright ("circular reference"), rewritten or not. Either
        # outcome is safe; returning another country's rows is the only failure.
        ksa = _who(authorizer, "demo-ksa")
        sql = "WITH customers AS (SELECT * FROM customers) SELECT DISTINCT country FROM customers"
        try:
            rows = _run(sql, ksa, db_path)
        except ExecutionError:
            return
        assert rows == (("Saudi Arabia",),)

    def test_a_union_branch_cannot_escape(self, authorizer: Authorizer, db_path: Path) -> None:
        ksa = _who(authorizer, "demo-ksa")
        sql = "SELECT country FROM customers WHERE 1 = 0 UNION SELECT country FROM customers"
        assert _run(sql, ksa, db_path) == (("Saudi Arabia",),)

    def test_the_filter_reads_an_unfiltered_table_once(
        self, authorizer: Authorizer, db_path: Path
    ) -> None:
        # The orders filter reads warehouses; warehouses has no filter of its own, and the
        # filter's own subquery must not be rewritten again.
        ksa = _who(authorizer, "demo-ksa")
        expected = _scalar(
            db_path,
            "SELECT COUNT(*) FROM orders WHERE warehouse_id IN "
            "(SELECT warehouse_id FROM warehouses WHERE country = ?)",
            "Saudi Arabia",
        )
        assert _run("SELECT COUNT(*) FROM orders", ksa, db_path) == ((expected,),)

    def test_parameters_come_from_the_principal_only(self, authorizer: Authorizer) -> None:
        _, params = apply_row_filters(
            "SELECT COUNT(*) FROM customers", _who(authorizer, "demo-ksa")
        )
        assert params == {"_user_country": "Saudi Arabia"}

    def test_no_filter_leaves_the_query_unchanged(self, authorizer: Authorizer) -> None:
        sql = "SELECT COUNT(*) FROM customers"
        assert apply_row_filters(sql, _who(authorizer, "demo-analyst")) == (sql, {})
        assert apply_row_filters(sql, ADMIN) == (sql, {})
