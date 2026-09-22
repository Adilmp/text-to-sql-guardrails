"""Executor tests: the runtime half of defence in depth.

These assert that the safety properties hold *independently of the validator*. If every
static check were removed, a write would still fail here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mizan.errors import ExecutionError, QueryTimeout
from mizan.guardrails import execute


class TestReadOnly:
    def test_write_rejected_at_the_driver(self, db_path: Path) -> None:
        """No validator involved: the connection itself refuses writes."""
        with pytest.raises(ExecutionError) as exc:
            execute("DELETE FROM orders", db_path)
        assert "readonly" in str(exc.value).lower() or "read-only" in str(exc.value).lower()

    def test_create_table_rejected(self, db_path: Path) -> None:
        with pytest.raises(ExecutionError):
            execute("CREATE TABLE evil (x INT)", db_path)

    def test_missing_database(self, tmp_path: Path) -> None:
        with pytest.raises(ExecutionError, match="database not found"):
            execute("SELECT 1", tmp_path / "nope.sqlite")


class TestRowCapping:
    def test_rows_capped(self, db_path: Path) -> None:
        result = execute("SELECT * FROM orders", db_path, max_rows=5)
        assert result.row_count == 5
        assert result.truncated is True

    def test_truncated_false_when_under_cap(self, db_path: Path) -> None:
        result = execute("SELECT * FROM orders LIMIT 3", db_path, max_rows=100)
        assert result.row_count == 3
        assert result.truncated is False

    def test_exactly_at_cap_is_not_truncated(self, db_path: Path) -> None:
        """Boundary: n rows with a cap of n must not be reported as truncated."""
        result = execute("SELECT * FROM orders LIMIT 4", db_path, max_rows=4)
        assert result.row_count == 4
        assert result.truncated is False


class TestTimeout:
    def test_runaway_query_interrupted(self, db_path: Path) -> None:
        """A recursive CTE with no bound would run forever without the progress handler."""
        runaway = (
            "WITH RECURSIVE forever(n) AS ("
            "  SELECT 1 UNION ALL SELECT n + 1 FROM forever"
            ") SELECT COUNT(*) FROM forever"
        )
        with pytest.raises(QueryTimeout):
            execute(runaway, db_path, timeout_s=0.5)


class TestResults:
    def test_columns_and_fingerprint(self, db_path: Path) -> None:
        result = execute("SELECT status, COUNT(*) AS n FROM orders GROUP BY status", db_path)
        assert result.columns == ("status", "n")
        assert result.row_count > 0

    def test_fingerprint_is_order_independent(self, db_path: Path) -> None:
        """Execution accuracy compares results, so row order must not matter.

        The query must return fewer rows than the cap: truncating two differently-ordered
        result sets yields genuinely different subsets, and comparing those would be
        testing the row cap rather than the fingerprint.
        """
        a = execute("SELECT DISTINCT status FROM orders ORDER BY status ASC", db_path, max_rows=50)
        b = execute("SELECT DISTINCT status FROM orders ORDER BY status DESC", db_path, max_rows=50)
        assert not a.truncated and not b.truncated
        assert a.rows != b.rows  # different order...
        assert a.fingerprint() == b.fingerprint()  # ...same identity

    def test_blob_is_described_not_embedded(self, db_path: Path) -> None:
        result = execute("SELECT CAST('abc' AS BLOB) AS b", db_path)
        assert str(result.rows[0][0]).startswith("<blob")

    def test_result_is_json_serialisable(self, db_path: Path) -> None:
        import json

        result = execute("SELECT * FROM orders LIMIT 2", db_path)
        json.dumps(result.to_dict())  # must not raise
