"""The repair loop's half of the conversation: what to tell the model about its broken query.

Why repair at all
-----------------
Most wrong answers on hard questions were not wrong *ideas*. The model joined the right
tables but left ``courier_id`` unqualified (SQLite: "ambiguous column name"), or read
``city`` off ``orders`` when it lives on ``customers``. The validator and SQLite already
know exactly what is wrong; showing the model that message and asking again fixes most of
them for the price of one more generation, and only on the questions that need it.

DDIA ch. 12 makes the general argument ("trust, but verify"): check at the end of the
pipeline, where the answer is known, rather than hope each stage got it right. The
validator is that end-to-end check; this module turns its verdict into a second chance.

What is never repaired
----------------------
A query that tried to write, escape the sandbox, call a denied function or stack a second
statement is **final**. Asking the model to "fix" ``DROP TABLE orders`` would turn a
blocked attack into a cooperative rewrite of it; the request was the problem, not the
syntax. Those answers stay blocked, and the rule that fired is reported.

The same goes for a reply that *contains* a write keyword but was rejected for something
else. ``SELECT name_en FROM couriers WITH c AS (DROP TABLE couriers)`` is a parse error, and
the first version of this loop asked the model to fix it; the "fix" was the same attack
with a semicolon in it. A reply that mentions ``DROP`` as SQL (not inside a string) is an
attempt, whatever rule it happened to break first.

Every repaired query goes through the same validator and the same sandbox as the first one,
so the loop can't make anything less safe; it can only spend another generation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..guardrails import GuardrailReport, suggest_identifier, write_keywords
from ..guardrails.policy import DANGEROUS_RULES
from ..schema.catalog import Catalog
from ..validate.grounding import GroundingIssue

#: Rules whose rejection is final: the problem is what the query tries to do.
FINAL_RULES: frozenset[str] = DANGEROUS_RULES | {"sql_too_long"}

_AMBIGUOUS_RE = re.compile(r"ambiguous column name: (\S+)", re.IGNORECASE)
_NO_COLUMN_RE = re.compile(r"no such column: (\S+)", re.IGNORECASE)
_QUOTED_RE = re.compile(r"'([^']+)'")

_DATE_FUNCTIONS = "date, datetime, julianday, strftime"


@dataclass(frozen=True)
class Problem:
    """Why a candidate needs another attempt, and what to tell the model."""

    #: ``rejected`` (validator), ``execution_error`` (SQLite) or ``ungrounded_value``.
    kind: str
    feedback: str


def diagnose(
    report: GuardrailReport,
    error: str | None,
    issues: list[GroundingIssue],
    catalog: Catalog,
    raw_output: str = "",
) -> Problem | None:
    """The problem to send back to the model, or ``None`` if the answer stands as it is.

    ``raw_output`` is the model's whole reply; a rejected reply that contains a write
    keyword is final (see the module docstring).
    """
    if not report.ok:
        if set(report.rules_fired) & FINAL_RULES:
            return None
        if write_keywords(raw_output or report.original_sql):
            return None
        hints = [_violation_hint(v.rule, v.detail, catalog) for v in report.violations]
        return Problem("rejected", _compose(hints))
    if error:
        return Problem("execution_error", _compose([_execution_hint(error, catalog)]))
    if issues:
        lines = [f"{issue}." for issue in issues]
        lines.append("Use the stored values exactly as written.")
        return Problem("ungrounded_value", _compose(lines))
    return None


def _compose(lines: list[str]) -> str:
    body = "\n".join(f"- {line}" for line in dict.fromkeys(lines) if line)
    return (
        "That query has a problem:\n"
        f"{body}\n"
        "Write the corrected SQLite query for the same question. Output only the SQL."
    )


def _violation_hint(rule: str, detail: str, catalog: Catalog) -> str:
    quoted = _QUOTED_RE.findall(detail)
    name = quoted[0] if quoted else ""
    if rule == "unknown_column" and name:
        return _column_hint(name, detail, catalog)
    if rule == "unknown_table" and name:
        guess = suggest_identifier(name, catalog.all_table_names)
        tables = ", ".join(sorted(catalog.all_table_names))
        return (
            f"There is no table {name!r}"
            + (f" (did you mean {guess!r}?)" if guess else "")
            + f". The tables are: {tables}."
        )
    if rule == "unknown_alias":
        return f"{detail}. Define every alias in FROM or JOIN before using it."
    if rule in ("function_not_allowed", "denied_function"):
        fn = detail.split("(")[0]
        return (
            f"{fn}() is not available. Use SQLite functions only; for dates use "
            f"{_DATE_FUNCTIONS} (julianday(a) - julianday(b) gives days)."
        )
    if rule == "parse_error":
        return f"It is not a single valid SQLite statement ({detail})."
    return f"{rule}: {detail}"


def _column_hint(name: str, detail: str, catalog: Catalog) -> str:
    owners = sorted(t.name for t in catalog.tables.values() if catalog.has_column(t.name, name))
    if owners:
        where = ", ".join(f"{t}.{name}" for t in owners)
        return (
            f"Column {name!r} is not on the table you used it with ({detail}). It exists as "
            f"{where}: JOIN that table using the join conditions and qualify the column."
        )
    guess = suggest_identifier(name, catalog.all_column_names)
    return f"There is no column {name!r} in any table" + (
        f" (did you mean {guess!r}?)." if guess else "."
    )


def _execution_hint(error: str, catalog: Catalog) -> str:
    if match := _AMBIGUOUS_RE.search(error):
        column = match.group(1)
        return (
            f"SQLite: ambiguous column name {column}. More than one joined table has a "
            f"column called {column}: prefix EVERY column with its table alias."
        )
    if match := _NO_COLUMN_RE.search(error):
        return _column_hint(match.group(1).split(".")[-1], "SQLite: no such column", catalog)
    if error.startswith("query exceeded"):
        return "The query took too long. Avoid joins without an ON condition."
    message = error.split(" (sql=")[0]
    return f"SQLite rejected it: {message}."
