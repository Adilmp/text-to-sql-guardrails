"""Value grounding: does each text value the query filters on actually occur in the data?

The failure this catches
------------------------
``WHERE c.city = 'dubai'`` is valid SQL with real identifiers. It passes every guardrail,
runs without error, and returns zero rows (or a ``COUNT`` of 0), because the stored value
is ``'Dubai'``. Nothing anywhere says it went wrong. Arabic and Urdu questions make it more
likely: ``WHERE city = 'دبي'`` against English data is the same bug.

What it does
------------
For every ``column = 'text'`` and ``column IN ('text', ...)`` in the query, where the column
resolves to one real text column, it looks the value up. A value that doesn't occur is
reported **only when the data has a close match** (same letters in another case, a small
spelling difference) **or the value is in Arabic script** while the stored values are not.
"How many customers are in Tokyo?" filters on a value that genuinely isn't there, and the
right answer is 0; a check that complained about every missing value would talk the model
out of a correct answer.

The issues are not a guardrail and never block anything: the pipeline hands them to the
model as a repair hint ("customers.city has no 'dubai'; did you mean 'Dubai'?").

The lookups go through :func:`mizan.guardrails.execute`, so they run with the same four
runtime defences as the query itself, and they are built as syntax trees, so a value taken
from the model's query is quoted by the SQL generator, never pasted into a string.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path

from sqlglot import exp

from ..errors import MizanError
from ..guardrails.executor import execute
from ..guardrails.validator import DIALECT, parse_sql
from ..logging import get_logger
from ..nl.normalize import arabic_ratio, normalize_for_matching
from ..schema.catalog import Catalog, is_prompt_safe

logger = get_logger("grounding")

#: Distinct values read to look for a near match. Enough for a categorical column or a
#: name column on a demo-sized table; on a large table the check degrades to "no
#: suggestion", which is the safe direction.
_MAX_DISTINCT = 500
#: Lookups per query. A model that writes ``IN`` with 200 values is not asking for help.
_MAX_LOOKUPS = 12
_LOOKUP_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class GroundingIssue:
    table: str
    column: str
    value: str
    #: Stored values the model probably meant, closest first. Empty only for an
    #: Arabic-script value whose column has no short value list to offer.
    suggestions: tuple[str, ...]

    def __str__(self) -> str:
        where = f"{self.table}.{self.column}"
        if self.suggestions:
            options = ", ".join(repr(s) for s in self.suggestions)
            return f"no row has {where} = {self.value!r}; the stored values include {options}"
        return f"no row has {where} = {self.value!r}; stored values are in English"


def check_values(sql: str, catalog: Catalog, db_path: Path) -> list[GroundingIssue]:
    """Text values in ``sql`` that don't occur in their column but probably meant one that does.

    ``sql`` is expected to have passed validation already. Anything that can't be checked
    (unparseable, a column that can't be resolved to one table, a lookup that fails) is
    skipped: this is a hint for the repair loop, and a missing hint is harmless.
    """
    try:
        tree = parse_sql(sql)
    except MizanError:
        return []

    aliases = _aliases(tree, catalog)
    issues: list[GroundingIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for column, value in _comparisons(tree):
        if len(seen) >= _MAX_LOOKUPS:
            break
        resolved = _resolve(column, aliases, catalog)
        if resolved is None:
            continue
        table, col = resolved
        key = (table, col, value)
        if key in seen:
            continue
        seen.add(key)
        issue = _check_one(table, col, value, catalog, db_path)
        if issue is not None:
            issues.append(issue)
    return issues


# ----------------------------------------------------------------------------- internals


def _aliases(tree: exp.Expression, catalog: Catalog) -> dict[str, str]:
    """Alias (or bare name) -> real table name, for tables that exist in the catalog."""
    out: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if catalog.has_table(name):
            out[(table.alias or table.name).lower()] = name
    return out


def _comparisons(tree: exp.Expression) -> list[tuple[exp.Column, str]]:
    """``(column, text value)`` for every equality and IN-list against a string literal."""
    pairs: list[tuple[exp.Column, str]] = []
    for node in tree.find_all(exp.EQ):
        left, right = node.this, node.expression
        if isinstance(left, exp.Column) and _is_text(right):
            pairs.append((left, right.name))
        elif isinstance(right, exp.Column) and _is_text(left):
            pairs.append((right, left.name))
    for in_node in tree.find_all(exp.In):
        if isinstance(in_node.this, exp.Column):
            pairs += [(in_node.this, e.name) for e in in_node.expressions if _is_text(e)]
    return pairs


def _is_text(node: exp.Expression | None) -> bool:
    return isinstance(node, exp.Literal) and node.is_string


def _resolve(
    column: exp.Column, aliases: dict[str, str], catalog: Catalog
) -> tuple[str, str] | None:
    name = column.name
    if column.table:
        table = aliases.get(column.table.lower())
        owners = [table] if table and catalog.has_column(table, name) else []
    else:
        owners = sorted({t for t in aliases.values() if catalog.has_column(t, name)})
    if len(owners) != 1:
        return None
    col = catalog.table(owners[0]).column(name)
    if col is None or not any(k in col.type for k in ("CHAR", "TEXT", "CLOB")):
        return None
    return catalog.table(owners[0]).name, col.name


def _check_one(
    table: str, column: str, value: str, catalog: Catalog, db_path: Path
) -> GroundingIssue | None:
    target = exp.Table(this=exp.to_identifier(table, quoted=True))
    col = exp.Column(this=exp.to_identifier(column, quoted=True))
    exists = (
        exp.select(exp.Literal.number(1))
        .from_(target.copy())
        .where(exp.EQ(this=col.copy(), expression=exp.Literal.string(value)))
        .limit(1)
    )
    try:
        if execute(
            exists.sql(dialect=DIALECT), db_path, max_rows=1, timeout_s=_LOOKUP_TIMEOUT_S
        ).rows:
            return None
        stored = catalog.table(table).column(column)
        values = list(stored.sample_values) if stored and stored.sample_values else None
        if values is None:
            distinct = (
                exp.select(col.copy())
                .distinct()
                .from_(target.copy())
                .where(exp.Not(this=exp.Is(this=col.copy(), expression=exp.Null())))
                .limit(_MAX_DISTINCT + 1)
            )
            rows = execute(
                distinct.sql(dialect=DIALECT),
                db_path,
                max_rows=_MAX_DISTINCT + 1,
                timeout_s=_LOOKUP_TIMEOUT_S,
            ).rows
            # These values reach the model's prompt in a repair hint, so they get the same
            # filter as the sample values in the schema card (D20): a stored value is
            # untrusted input.
            values = [v for r in rows[:_MAX_DISTINCT] if is_prompt_safe(v := str(r[0]))]
    except MizanError as exc:
        logger.debug("grounding lookup skipped", extra={"error": str(exc)})
        return None

    close = _closest(value, values)
    if close:
        return GroundingIssue(table, column, value, tuple(close))
    if arabic_ratio(value) > 0.5 and not any(arabic_ratio(v) > 0.5 for v in values):
        # An Arabic-script value against Latin data has no "close" spelling to find; offer
        # the whole value list when it is short enough to be a category.
        listed = tuple(values) if stored and stored.sample_values else ()
        return GroundingIssue(table, column, value, listed)
    return None


def _closest(value: str, values: list[str]) -> list[str]:
    """Stored values that differ from ``value`` only in case, spacing or a small typo."""
    key = normalize_for_matching(value).replace("_", " ").strip()
    exact = [v for v in values if normalize_for_matching(v).replace("_", " ").strip() == key]
    if exact:
        return exact[:3]
    pool = {normalize_for_matching(v): v for v in values}
    return [pool[m] for m in difflib.get_close_matches(key, list(pool), n=3, cutoff=0.8)]
