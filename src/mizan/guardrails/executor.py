"""Safe execution of validated SQL.

Defence in depth
----------------
The AST validator is the first line, not the only one. A validator bug, a ``sqlglot``
version change, or a SQLite feature nobody anticipated should still not be able to write to
the database. Four independent mechanisms have to fail simultaneously for that to happen:

1. **The connection is opened read-only at the driver level** via the URI form
   ``file:...?mode=ro``. SQLite itself refuses writes; this is not something application
   code can accidentally undo.
2. **``PRAGMA query_only = ON``** is set on top, which blocks writes even on a connection
   that somehow opened read-write.
3. **Extension loading is explicitly disabled**, closing ``load_extension`` even if the
   function allowlist were bypassed.
4. **A progress handler enforces a wall-clock deadline**, so a query that passes every
   static check but happens to be a cartesian product is interrupted rather than pinning a
   core until the process is killed.

Row capping is done with ``fetchmany(n + 1)``: fetching one row more than the limit is what
distinguishes "there were exactly n rows" from "there were more and we truncated", without
ever materialising the full result set.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ExecutionError, QueryTimeout
from ..logging import get_logger

logger = get_logger("executor")

#: How often SQLite calls the progress handler, in virtual-machine instructions. Small
#: enough that the deadline is honoured promptly, large enough that the callback overhead
#: stays in the noise.
_PROGRESS_INSTRUCTIONS = 1_000

#: Maximum characters kept from any single cell.
#:
#: **This exists because of a real vulnerability found by security testing.** Every other
#: limit in this system bounds *row count* or *wall-clock time*; none bounded *bytes*. So
#: ``SELECT printf('%.*c', 200000000, 'x')`` returned one row, in 1.1 seconds, containing a
#: 200 MB string — passing the row cap (1 row), the timeout (1.1s of 5s), the function
#: allowlist (``printf`` is a legitimate formatting function) and every AST check.
#:
#: The fix deliberately bounds the *outcome* rather than blacklisting the primitive. A
#: denylist of amplification functions is the same losing game as a denylist of dangerous
#: ones: ``printf``, ``char``, ``replace``, ``hex``, ``zeroblob`` and ``group_concat`` can
#: all inflate a result, and the next SQLite release may add another. A byte budget cannot
#: be routed around, because it constrains what any query is allowed to *produce*.
_MAX_CELL_CHARS = 4_096

#: Maximum characters across the whole result set. Bounds the aggregate case that a
#: per-cell cap alone would miss: 200 rows x 4 KB is fine, 200 rows of maximum-width cells
#: across 30 columns is not.
_MAX_TOTAL_CHARS = 1_000_000

#: Ceiling on the length of any string SQLite will construct, applied at the engine level
#: when the interpreter supports it (``Connection.setlimit`` is Python 3.11+). On 3.10 this
#: is unavailable, and the residual exposure is documented in ``execute``.
_SQLITE_LENGTH_LIMIT = 8 * 1024 * 1024


@dataclass(frozen=True)
class QueryResult:
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    truncated: bool
    elapsed_ms: float
    #: Cells shortened because they exceeded the per-cell character cap. Surfaced rather
    #: than silent: a truncated value is not the value the database holds, and a caller
    #: comparing results needs to know that.
    cells_truncated: int = 0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
            "row_count": self.row_count,
            "truncated": self.truncated,
            "cells_truncated": self.cells_truncated,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }

    def fingerprint(self) -> frozenset[tuple[Any, ...]]:
        """Order-independent identity of the result set.

        Execution accuracy compares what two queries *return*, not how they are written.
        Using a set of row tuples makes ``ORDER BY`` differences and column ordering
        irrelevant — which is correct, because a question like "which cities have the most
        late orders" has many equally right SQL spellings.
        """
        return frozenset(self.rows)


def _to_jsonable(value: Any, max_chars: int = _MAX_CELL_CHARS) -> tuple[Any, bool]:
    """Coerce a SQLite value into something JSON can hold, bounded in size.

    Returns ``(value, was_truncated)``.
    """
    if isinstance(value, bytes):
        # Blobs are never useful in an answer and can be megabytes. Describe, don't embed.
        return f"<blob {len(value)} bytes>", False
    if isinstance(value, (int, float, type(None))):
        return value, False
    text = value if isinstance(value, str) else str(value)
    if len(text) > max_chars:
        return f"{text[:max_chars]}… <truncated, {len(text):,} chars total>", True
    return text, False


def execute(
    sql: str,
    db_path: Path,
    *,
    max_rows: int = 200,
    timeout_s: float = 5.0,
) -> QueryResult:
    """Run ``sql`` read-only against ``db_path``.

    Raises :class:`QueryTimeout` if the deadline elapses and :class:`ExecutionError` for
    any other database failure.
    """
    if not db_path.exists():
        raise ExecutionError(f"database not found: {db_path}", path=str(db_path))

    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    timed_out = False

    def _watchdog() -> int:
        # Returning non-zero tells SQLite to abort the current statement. The flag is
        # needed because the resulting OperationalError says only "interrupted" and is
        # indistinguishable from a caller-issued interrupt.
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1
        return 0

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
        # Not available in every CPython build; absence is fine because extension loading
        # is off by default, but disable it explicitly wherever we can.
        if hasattr(conn, "enable_load_extension"):
            with suppress(AttributeError, sqlite3.NotSupportedError):
                conn.enable_load_extension(False)
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(_watchdog, _PROGRESS_INSTRUCTIONS)

        # Engine-level ceiling on constructed string length. Python 3.11+ only; on 3.10
        # this is a no-op and the residual exposure is the transient allocation inside
        # SQLite before the cell cap below discards it (see the note in the docstring).
        # Resolved dynamically: both `Connection.setlimit` and the constant are Python
        # 3.11+, so a static reference does not type-check on 3.10.
        length_limit_id = getattr(sqlite3, "SQLITE_LIMIT_LENGTH", None)
        if length_limit_id is not None and hasattr(conn, "setlimit"):
            with suppress(AttributeError, sqlite3.Error, ValueError):
                conn.setlimit(length_limit_id, _SQLITE_LENGTH_LIMIT)

        cursor = conn.execute(sql)
        columns = tuple(d[0] for d in (cursor.description or ()))
        fetched = cursor.fetchmany(max_rows + 1)
        truncated = len(fetched) > max_rows

        rows_out: list[tuple[Any, ...]] = []
        cells_truncated = 0
        total_chars = 0
        for raw_row in fetched[:max_rows]:
            converted: list[Any] = []
            for value in raw_row:
                cell, was_cut = _to_jsonable(value)
                cells_truncated += int(was_cut)
                total_chars += len(cell) if isinstance(cell, str) else 8
                converted.append(cell)
            rows_out.append(tuple(converted))
            if total_chars > _MAX_TOTAL_CHARS:
                # Aggregate budget blown: stop here and say so, rather than returning a
                # response large enough to be its own denial of service downstream.
                truncated = True
                logger.warning(
                    "result truncated on total size budget",
                    extra={"rows_kept": len(rows_out), "total_chars": total_chars},
                )
                break
        rows = tuple(rows_out)

        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.debug(
            "query executed",
            extra={
                "rows": len(rows),
                "truncated": truncated,
                "cells_truncated": cells_truncated,
                "elapsed_ms": elapsed_ms,
            },
        )
        return QueryResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=elapsed_ms,
            cells_truncated=cells_truncated,
        )

    except sqlite3.OperationalError as exc:
        if timed_out or "interrupted" in str(exc).lower():
            raise QueryTimeout(
                f"query exceeded {timeout_s}s", timeout_s=timeout_s, sql=sql[:200]
            ) from exc
        raise ExecutionError(f"SQLite error: {exc}", sql=sql[:200]) from exc
    except sqlite3.Error as exc:
        raise ExecutionError(f"SQLite error: {exc}", sql=sql[:200]) from exc
    finally:
        if conn is not None:
            # Detach the handler before closing: a handler holding a reference to a closed
            # connection is a segfault risk in some CPython/SQLite combinations.
            with suppress(sqlite3.Error):
                conn.set_progress_handler(None, 0)
            conn.close()
