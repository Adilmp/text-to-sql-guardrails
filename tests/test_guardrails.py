"""Guardrail tests.

Two properties are asserted, and both matter:

* **No attack passes.** Parameterised over every bypass class the design anticipates.
* **No legitimate query is rejected.** A guardrail with a high false-positive rate silently
  destroys accuracy and looks like a weak model. The false-positive suite is as important
  as the attack suite — three of these cases were real bugs found during the build.
"""

from __future__ import annotations

import pytest

from mizan.guardrails import GuardrailPolicy, extract_sql, has_multiple_statements, validate
from mizan.guardrails.validator import _collect_functions, parse_sql
from mizan.schema import Catalog

ATTACKS: list[tuple[str, str, str]] = [
    ("stacked_statements", "SELECT 1; DROP TABLE orders", "stacked query"),
    ("parse_error", "DR/**/OP TABLE orders", "comment-split keyword"),
    (
        "write_operation",
        "WITH x AS (DELETE FROM orders RETURNING 1) SELECT * FROM x",
        "CTE-hidden DML",
    ),
    ("write_operation", "UpDaTe orders SET status=1", "mixed-case write"),
    ("write_operation", "INSERT INTO orders VALUES (1)", "insert"),
    ("write_operation", "DROP TABLE orders", "drop"),
    ("dangerous_statement", "ATTACH DATABASE '/tmp/evil.db' AS e", "attach another database"),
    ("dangerous_statement", "PRAGMA writable_schema=ON", "pragma escalation"),
    ("unsupported_statement", "VACUUM", "unmodelled statement"),
    ("denied_function", "SELECT load_extension('/tmp/x.so')", "arbitrary code execution"),
    ("denied_function", "SELECT readfile('/etc/passwd')", "filesystem read"),
    ("denied_function", "SELECT zeroblob(2000000000)", "memory exhaustion"),
    ("unknown_table", "SELECT * FROM invoices", "hallucinated table"),
    ("unknown_column", "SELECT bogus_col FROM orders", "hallucinated column"),
]

LEGITIMATE: list[tuple[str, str]] = [
    (
        "boolean operators",
        "SELECT COUNT(*) FROM orders "
        "WHERE delivered_at IS NOT NULL AND delivered_at > promised_at",
    ),
    (
        "select alias in ORDER BY",
        "SELECT courier_id, COUNT(*) AS n FROM orders GROUP BY courier_id ORDER BY n DESC",
    ),
    (
        "select alias in HAVING",
        "SELECT city, AVG(capacity_m3) AS a FROM warehouses GROUP BY city HAVING a > 1",
    ),
    (
        "join with aliases",
        "SELECT c.name_en, AVG(o.total_aed) FROM orders o "
        "JOIN customers c ON c.customer_id = o.customer_id GROUP BY c.name_en",
    ),
    (
        "cte",
        "WITH late AS (SELECT courier_id FROM orders WHERE delivered_at > promised_at) "
        "SELECT COUNT(*) FROM late",
    ),
    ("or and like", "SELECT * FROM customers WHERE city = 'Dubai' OR name_en LIKE 'A%'"),
    ("strftime", "SELECT strftime('%Y', placed_at) AS yr, COUNT(*) FROM orders GROUP BY yr"),
    ("date function", "SELECT COUNT(*) FROM orders WHERE date(placed_at) > '2025-06-01'"),
    (
        "subquery in IN",
        "SELECT * FROM orders WHERE customer_id IN "
        "(SELECT customer_id FROM customers WHERE country = 'UAE')",
    ),
    (
        "case expression",
        "SELECT CASE WHEN total_aed > 500 THEN 'big' ELSE 'small' END AS bucket, "
        "COUNT(*) FROM orders GROUP BY bucket",
    ),
    ("string concat", "SELECT name_en || ' - ' || city AS label FROM customers"),
    (
        "window function",
        "SELECT order_id, ROW_NUMBER() OVER (ORDER BY total_aed DESC) AS rk FROM orders",
    ),
]


class TestAttacks:
    @pytest.mark.parametrize(
        ("expected_rule", "sql", "label"),
        ATTACKS,
        ids=[label for _, _, label in ATTACKS],
    )
    def test_attack_blocked(
        self, expected_rule: str, sql: str, label: str, catalog: Catalog
    ) -> None:
        report = validate(sql, catalog)
        assert not report.ok, f"{label} was NOT blocked"
        assert expected_rule in report.rules_fired, (
            f"{label} blocked by {report.rules_fired}, expected {expected_rule}"
        )

    def test_blocked_query_returns_no_executable_sql(self, catalog: Catalog) -> None:
        """A rejected report must not hand back anything runnable."""
        assert validate("DROP TABLE orders", catalog).sql == ""

    def test_oversized_sql_rejected(self, catalog: Catalog) -> None:
        policy = GuardrailPolicy(max_sql_chars=50)
        report = validate("SELECT " + "a," * 100 + "1 FROM orders", catalog, policy)
        assert "sql_too_long" in report.rules_fired


class TestNoFalsePositives:
    @pytest.mark.parametrize(
        ("label", "sql"), LEGITIMATE, ids=[label for label, _ in LEGITIMATE]
    )
    def test_legitimate_query_passes(self, label: str, sql: str, catalog: Catalog) -> None:
        report = validate(sql, catalog)
        assert report.ok, f"false positive on {label}: {report.rules_fired}"


class TestFunctionNameResolution:
    """Regression tests for the two traps in :func:`_function_name`."""

    def test_operators_are_not_functions(self) -> None:
        tree = parse_sql("SELECT a FROM x WHERE p AND q OR r")
        assert _collect_functions(tree) == set()

    def test_dialect_name_not_sqlglot_internal_name(self) -> None:
        """sqlglot parses strftime to exp.TimeToStr; the allowlist must see 'strftime'."""
        tree = parse_sql("SELECT strftime('%Y', placed_at) FROM orders")
        assert "strftime" in _collect_functions(tree)
        assert "time_to_str" not in _collect_functions(tree)

    def test_unmodelled_function_keeps_literal_name(self) -> None:
        tree = parse_sql("SELECT load_extension('x')")
        assert "load_extension" in _collect_functions(tree)


class TestLimitHandling:
    def test_limit_injected_when_absent(self, catalog: Catalog) -> None:
        report = validate("SELECT * FROM orders", catalog, GuardrailPolicy(max_rows=25))
        assert "LIMIT 25" in report.sql
        assert any("injected" in w for w in report.warnings)

    def test_looser_limit_lowered(self, catalog: Catalog) -> None:
        report = validate("SELECT * FROM orders LIMIT 5000", catalog, GuardrailPolicy(max_rows=25))
        assert "LIMIT 25" in report.sql
        assert any("lowered" in w for w in report.warnings)

    def test_tighter_limit_preserved(self, catalog: Catalog) -> None:
        report = validate("SELECT * FROM orders LIMIT 3", catalog, GuardrailPolicy(max_rows=25))
        assert "LIMIT 3" in report.sql
        assert report.warnings == []


class TestComplexity:
    def test_too_many_joins(self, catalog: Catalog) -> None:
        sql = (
            "SELECT 1 FROM orders o "
            + " ".join(
                f"JOIN customers c{i} ON c{i}.customer_id = o.customer_id" for i in range(8)
            )
        )
        assert "too_many_joins" in validate(sql, catalog, GuardrailPolicy(max_joins=3)).rules_fired


class TestAmbiguityWarning:
    """Ambiguity is reported as a warning and only where the scope check is exact."""

    def _ambiguity(self, sql: str, catalog: Catalog) -> list[str]:
        report = validate(sql, catalog)
        assert report.ok, f"ambiguity must never reject: {report.rules_fired}"
        return [w for w in report.warnings if "ambiguous" in w]

    def test_flat_scope_ambiguity_is_warned(self, catalog: Catalog) -> None:
        sql = (
            "SELECT courier_id, COUNT(*) FROM orders "
            "JOIN couriers ON couriers.courier_id = orders.courier_id GROUP BY courier_id"
        )
        assert self._ambiguity(sql, catalog)

    def test_qualified_column_is_not_ambiguous(self, catalog: Catalog) -> None:
        sql = "SELECT o.courier_id FROM orders o JOIN couriers c ON c.courier_id = o.courier_id"
        assert not self._ambiguity(sql, catalog)

    def test_subquery_scope_is_not_flagged(self, catalog: Catalog) -> None:
        """The union of referenced tables is not the visible scope once a subquery exists.

        `courier_id` here is unambiguous inside the subquery. Flagging it would be a false
        positive, which is why the check is restricted to a single flat SELECT.
        """
        sql = "SELECT (SELECT COUNT(*) FROM couriers WHERE courier_id = 5) AS n FROM orders"
        assert not self._ambiguity(sql, catalog)

    def test_single_table_is_never_ambiguous(self, catalog: Catalog) -> None:
        assert not self._ambiguity("SELECT courier_id FROM orders", catalog)


class TestRobustness:
    """The validator must never raise on model output, however malformed.

    It sits between an LLM and a database, so its input is adversarial by construction. A
    crash here is an availability bug at best and a bypass at worst — an exception escaping
    `validate()` means the caller's `except` block decides what happens next, and that is
    not where the security decision belongs.
    """

    @pytest.mark.parametrize(
        ("label", "sql"),
        [
            ("empty", ""),
            ("whitespace only", "   \n\t "),
            ("prose", "I cannot answer that question."),
            ("deep nesting", "SELECT * FROM (" * 200 + "SELECT 1" + ")" * 200),
            ("unbalanced parens", "SELECT * FROM ((((orders"),
            ("lone keyword", "SELECT"),
            ("null byte in literal", "SELECT 1 FROM orders WHERE status = char(0)"),
            ("escaped quote", "SELECT * FROM customers WHERE city = 'Dub''ai'"),
            ("arabic literal", "SELECT * FROM customers WHERE name_ar = 'أحمد المنصوري'"),
            ("very long identifier", "SELECT " + "a" * 5000 + " FROM orders"),
        ],
    )
    def test_never_raises(self, label: str, sql: str, catalog: Catalog) -> None:
        report = validate(sql, catalog)
        assert isinstance(report.ok, bool)
        if not report.ok:
            assert report.sql == ""

    def test_recursive_cte_passes_validation_but_executor_bounds_it(
        self, catalog: Catalog
    ) -> None:
        """Defence in depth in one test.

        An unbounded recursive CTE is structurally valid SQL — no static check can decide
        whether it terminates, and pretending otherwise would mean rejecting every legitimate
        recursive query. The runtime deadline is what contains it, which is precisely why the
        executor's protections do not depend on the validator.
        """
        sql = (
            "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) "
            "SELECT * FROM r"
        )
        assert validate(sql, catalog).ok


class TestExtraction:
    def test_markdown_fence(self) -> None:
        assert extract_sql("```sql\nSELECT 1\n```") == "SELECT 1"

    def test_unterminated_fence(self) -> None:
        """A response truncated at the token limit still yields its body."""
        assert extract_sql("```sql\nSELECT 1") == "SELECT 1"

    def test_conversational_preamble(self) -> None:
        assert extract_sql("Sure! Here is the query:\nSELECT 1") == "SELECT 1"

    def test_trailing_explanation_removed(self) -> None:
        assert extract_sql("SELECT 1; -- this counts rows") == "SELECT 1"

    def test_semicolon_inside_string_literal_is_not_a_terminator(self) -> None:
        sql = "SELECT * FROM t WHERE x = 'a;b'"
        assert extract_sql(sql) == sql

    def test_no_statement_returns_empty(self) -> None:
        assert extract_sql("I cannot answer that.") == ""

    def test_empty_input(self) -> None:
        assert extract_sql("") == ""

    def test_multiple_statements_detected(self) -> None:
        assert has_multiple_statements("SELECT 1; DROP TABLE t")
        assert not has_multiple_statements("SELECT 1;")
        assert not has_multiple_statements("SELECT * FROM t WHERE x = 'a;b'")
