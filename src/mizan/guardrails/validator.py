"""AST-level validation of generated SQL.

Why an AST and never a regex
----------------------------
Regex-based SQL filtering is the most common security hole in Text-to-SQL systems, and
every bypass is a one-liner:

===============================================  ==========================================
Filter                                           Bypass
===============================================  ==========================================
block ``r"\\bDROP\\b"``                            ``DR/**/OP TABLE t``
block writes by prefix-matching ``SELECT``       ``SELECT 1; DROP TABLE t``
block ``DELETE``                                 ``WITH x AS (DELETE FROM t RETURNING 1)``
block ``;``                                      ``ATTACH DATABASE '/tmp/e.db' AS e``
case-sensitive matching                          ``dRoP TaBlE t``
===============================================  ==========================================

A parser is immune to all of these because it works on what the statement *is*, not on how
it is spelled. Every check below walks a ``sqlglot`` syntax tree.

The ``exp.Command`` trap
------------------------
``sqlglot`` parses syntax it does not model into a generic ``exp.Command`` node rather than
failing. ``PRAGMA``, ``ATTACH``, ``VACUUM`` and friends all land there. A validator that
only enumerates known-bad node types will wave every one of them through. This module
rejects ``exp.Command`` outright: an unmodelled statement is, by definition, one whose
effects we cannot reason about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import cast

import sqlglot
from sqlglot import exp

from ..errors import SQLParseError
from ..logging import get_logger
from ..nl.normalize import normalize_for_matching
from ..schema.catalog import Catalog
from .extract import has_multiple_statements
from .policy import GuardrailPolicy

logger = get_logger("guardrails")

DIALECT = "sqlite"

#: Any of these appearing anywhere in the tree means the statement mutates state.
_WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable,
)


def _optional_nodes(*names: str) -> tuple[type[exp.Expression], ...]:
    """Resolve sqlglot node classes that only exist in some versions.

    Newer sqlglot models ``ATTACH`` and ``PRAGMA`` as dedicated classes; older versions
    parse them into the generic ``exp.Command``. Resolving by name keeps this validator
    correct across both without pinning an exact sqlglot version — and, critically, without
    silently losing a check when the dependency is upgraded.
    """
    found = []
    for name in names:
        node = getattr(exp, name, None)
        if isinstance(node, type) and issubclass(node, exp.Expression):
            found.append(node)
    return tuple(found)


#: Statements that are neither reads nor ordinary writes: they reconfigure the engine or
#: reach outside the current database file. ``ATTACH`` is the important one — it opens a
#: second database file, which is a full sandbox escape on SQLite.
_DANGEROUS_NODES: tuple[type[exp.Expression], ...] = _optional_nodes(
    "Attach", "Detach", "Pragma", "Vacuum", "Analyze", "Transaction", "Commit",
    "Rollback", "Set", "Use", "Grant", "Revoke",
)


@dataclass(frozen=True)
class Violation:
    """A single reason the query was rejected.

    ``rule`` is a stable identifier used to bucket rejections in eval reports; ``detail`` is
    human text and may change freely.
    """

    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.detail}"


@dataclass
class GuardrailReport:
    """Outcome of validating one candidate statement."""

    ok: bool
    sql: str
    """The statement to execute — rewritten (e.g. with an injected LIMIT) when ``ok``."""
    original_sql: str
    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: frozenset[str] = frozenset()
    columns: frozenset[str] = frozenset()
    functions: frozenset[str] = frozenset()

    @property
    def rules_fired(self) -> list[str]:
        return [v.rule for v in self.violations]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "sql": self.sql,
            "original_sql": self.original_sql,
            "violations": [{"rule": v.rule, "detail": v.detail} for v in self.violations],
            "warnings": list(self.warnings),
            "tables": sorted(self.tables),
            "columns": sorted(self.columns),
            "functions": sorted(self.functions),
        }


def parse_sql(sql: str) -> exp.Expression:
    """Parse one statement, raising :class:`SQLParseError` on anything unusable."""
    if not sql or not sql.strip():
        raise SQLParseError("empty statement")
    try:
        statements = sqlglot.parse(sql, dialect=DIALECT)
    except Exception as exc:  # sqlglot raises several unrelated types
        raise SQLParseError(f"could not parse SQL: {exc}", sql=sql[:200]) from exc

    real = [s for s in statements if s is not None]
    if not real:
        raise SQLParseError("no statement found", sql=sql[:200])
    if len(real) > 1:
        raise SQLParseError(f"expected 1 statement, got {len(real)}", count=len(real))
    # sqlglot's own annotations are looser than its runtime behaviour here; everything it
    # yields from parse() is an Expression.
    return cast(exp.Expression, real[0])


def validate(
    sql: str,
    catalog: Catalog | None = None,
    policy: GuardrailPolicy | None = None,
) -> GuardrailReport:
    """Validate ``sql`` against ``policy`` and (optionally) a real schema ``catalog``.

    Returns a report rather than raising, so the eval harness can count *which* rule fired
    on every rejected query instead of seeing a single opaque failure.
    """
    policy = policy or GuardrailPolicy()
    violations: list[Violation] = []
    warnings: list[str] = []

    if len(sql) > policy.max_sql_chars:
        return GuardrailReport(
            ok=False,
            sql="",
            original_sql=sql,
            violations=[
                Violation("sql_too_long", f"{len(sql)} chars exceeds {policy.max_sql_chars}")
            ],
        )

    # Checked on the raw text, before extraction trims anything: a stacked query must be
    # reported as an attack, not silently reduced to its harmless first statement.
    if has_multiple_statements(sql):
        violations.append(
            Violation("stacked_statements", "multiple statements in one request")
        )

    try:
        tree = parse_sql(sql)
    except SQLParseError as exc:
        violations.append(Violation("parse_error", exc.message))
        return GuardrailReport(ok=False, sql="", original_sql=sql, violations=violations)

    violations.extend(_check_statement_kind(tree, policy))
    violations.extend(_check_functions(tree, policy))
    violations.extend(_check_complexity(tree, policy))

    tables = _collect_tables(tree)
    columns = _collect_columns(tree)
    functions = _collect_functions(tree)

    if catalog is not None and policy.enforce_schema:
        schema_violations, schema_warnings = _check_schema(tree, catalog)
        violations.extend(schema_violations)
        warnings.extend(schema_warnings)

    ok = not violations
    final_sql = sql
    if ok:
        if policy.inject_limit:
            final_sql, action = _ensure_limit(tree, policy.max_rows)
            if action == "injected":
                warnings.append(f"LIMIT {policy.max_rows} injected (query had none)")
            elif action == "lowered":
                warnings.append(f"LIMIT lowered to {policy.max_rows}")
        else:
            # Re-render from the validated tree even when nothing is being rewritten, so
            # that the executed statement is always the round-trip of the one that was
            # checked. Executing the raw string instead would leave a sliver of daylight
            # between "what we validated" and "what we run" — small here, because parsing
            # has already settled what the statement is, but the principle is worth keeping
            # absolute rather than true-by-default.
            final_sql = tree.sql(dialect=DIALECT)

    if not ok:
        logger.info(
            "guardrail rejection",
            extra={"rules": [v.rule for v in violations], "sql_preview": sql[:160]},
        )

    return GuardrailReport(
        ok=ok,
        sql=final_sql if ok else "",
        original_sql=sql,
        violations=violations,
        warnings=warnings,
        tables=frozenset(tables),
        columns=frozenset(columns),
        functions=frozenset(functions),
    )


# ----------------------------------------------------------------------- statement kind


def _check_statement_kind(tree: exp.Expression, policy: GuardrailPolicy) -> list[Violation]:
    if not policy.allow_only_select:
        return []
    out: list[Violation] = []

    # `exp.Command` is sqlglot's catch-all for syntax it does not model (VACUUM, REINDEX
    # and anything newer than the installed version). Rejecting it is essential: a
    # validator that only enumerates known-bad node types waves through everything it has
    # never heard of, which is precisely the wrong failure direction for a security check.
    for node in tree.find_all(exp.Command):
        out.append(
            Violation(
                "unsupported_statement",
                f"unmodelled statement type: {str(node.this)[:40]!r}",
            )
        )

    # Versions of sqlglot that *do* model these give them dedicated classes, so they never
    # reach the Command branch above and need their own check.
    for node_type in _DANGEROUS_NODES:
        for _ in tree.find_all(node_type):
            out.append(
                Violation(
                    "dangerous_statement",
                    f"{node_type.__name__.upper()} is not permitted",
                )
            )
            break

    # A write node *anywhere* in the tree, including inside a CTE, is fatal. Checking only
    # the root would miss `WITH x AS (DELETE ... RETURNING *) SELECT * FROM x`.
    for node_type in _WRITE_NODES:
        for _ in tree.find_all(node_type):
            out.append(
                Violation("write_operation", f"{node_type.__name__.upper()} is not permitted")
            )
            break

    root = tree
    # `not out` avoids piling a second, vaguer message onto a statement that a more
    # specific rule has already reported.
    if not out and not isinstance(root, (exp.Select, exp.Union, exp.Subquery, exp.With)):
        out.append(
            Violation(
                "not_a_select",
                f"top-level statement is {type(root).__name__}, expected SELECT",
            )
            )
    return out


# ---------------------------------------------------------------------------- functions


#: Matches a rendered function call: an identifier immediately followed by an open paren.
_FUNC_CALL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _function_name(node: exp.Func) -> str | None:
    """Name of the function **as written in the target dialect**, or ``None``.

    Two traps live here, both found by the allowlist producing false rejections on ordinary
    queries.

    **Trap 1 — operators are ``exp.Func`` subclasses.** ``exp.And`` inherits
    ``Connector -> Binary -> Func``, so ``find_all(exp.Func)`` yields every ``AND``, ``OR``,
    ``LIKE`` and comparison. A naive implementation reports a function called ``and`` and
    rejects almost every non-trivial ``WHERE`` clause. (Excluding those base classes by
    ``issubclass(..., exp.Expression)`` does not work either: ``Connector`` and ``Binary``
    are mixins that do not themselves derive from ``Expression``.)

    **Trap 2 — sqlglot canonicalises across dialects.** SQLite's ``strftime`` parses to
    ``exp.TimeToStr``, and its argument gets wrapped in ``exp.TsOrDsToTimestamp``. Checking
    an allowlist against sqlglot's internal class names compares against a vocabulary the
    user never typed and the database never sees.

    Both are solved by the same move: render the node back to SQLite and read the name off
    the generated text. If the rendering is not a call — because the node is an operator, or
    an implicit coercion sqlglot inserted — there is no function to authorise and this
    returns ``None``.
    """
    if isinstance(node, exp.Anonymous):
        # Anonymous is sqlglot's node for functions it does not model, which is exactly
        # where the dangerous ones (load_extension, readfile) land. Its name is verbatim.
        name = node.name or str(node.this or "")
        return name.lower() or None
    try:
        rendered = node.sql(dialect=DIALECT)
    except Exception:  # pragma: no cover - defensive; generation should not raise
        return None
    match = _FUNC_CALL_RE.match(rendered)
    return match.group(1).lower() if match else None


def _collect_functions(tree: exp.Expression) -> set[str]:
    names: set[str] = set()
    for node in tree.find_all(exp.Func):
        if name := _function_name(node):
            names.add(name)
    return names


def _check_functions(tree: exp.Expression, policy: GuardrailPolicy) -> list[Violation]:
    out: list[Violation] = []
    for name in sorted(_collect_functions(tree)):
        if name in policy.denied_functions:
            out.append(Violation("denied_function", f"{name}() is explicitly forbidden"))
        elif policy.enforce_function_allowlist and name not in policy.allowed_functions:
            out.append(Violation("function_not_allowed", f"{name}() is not on the allowlist"))
    return out


# --------------------------------------------------------------------------- complexity


def _check_complexity(tree: exp.Expression, policy: GuardrailPolicy) -> list[Violation]:
    out: list[Violation] = []

    joins = len(list(tree.find_all(exp.Join)))
    if joins > policy.max_joins:
        out.append(Violation("too_many_joins", f"{joins} joins exceeds {policy.max_joins}"))

    unions = len(list(tree.find_all(exp.Union)))
    if unions >= policy.max_union_branches:
        out.append(
            Violation(
                "too_many_unions",
                f"{unions + 1} branches exceeds {policy.max_union_branches}",
            )
        )

    depth = _max_select_depth(tree)
    if depth > policy.max_subquery_depth:
        out.append(
            Violation("subquery_too_deep", f"depth {depth} exceeds {policy.max_subquery_depth}")
        )
    return out


def _max_select_depth(node: exp.Expression, depth: int = 0) -> int:
    """Deepest nesting of SELECT nodes. Iterative walk to avoid recursion limits on a
    pathological input — a 10k-deep parenthesised query would blow the Python stack."""
    best = depth
    stack: list[tuple[exp.Expression, int]] = [(node, depth)]
    while stack:
        current, d = stack.pop()
        for child in current.args.values():
            children = child if isinstance(child, list) else [child]
            for c in children:
                if not isinstance(c, exp.Expression):
                    continue
                nd = d + 1 if isinstance(c, exp.Select) else d
                best = max(best, nd)
                stack.append((c, nd))
    return best


# ------------------------------------------------------------------- schema validation


def _collect_tables(tree: exp.Expression) -> set[str]:
    return {t.name.lower() for t in tree.find_all(exp.Table) if t.name}


def _collect_columns(tree: exp.Expression) -> set[str]:
    return {c.name.lower() for c in tree.find_all(exp.Column) if c.name and c.name != "*"}


def _defined_aliases(tree: exp.Expression) -> set[str]:
    """Names the query itself introduces, which are therefore not hallucinations.

    ``SELECT COUNT(*) AS late_deliveries ... ORDER BY late_deliveries`` parses the ORDER BY
    reference as an ``exp.Column``. Checking it against the catalog reports a column that
    does not exist — which is true, and completely wrong, because the query defined it two
    lines earlier. Exempting declared aliases is safe: an alias cannot be invented, since
    its definition is right there in the same statement.
    """
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    # Column aliases attached to a table alias: `FROM t AS x(a, b)`.
    for table_alias in tree.find_all(exp.TableAlias):
        for column in table_alias.args.get("columns") or []:
            if name := getattr(column, "name", None):
                aliases.add(str(name).lower())
    return aliases


def _alias_map(tree: exp.Expression) -> dict[str, str]:
    """Map table alias -> real table name, plus each real table mapped to itself.

    CTE names map to themselves so that selecting from a CTE is not reported as an unknown
    table; their internal columns are validated where the CTE body is validated.
    """
    mapping: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        if not table.name:
            continue
        real = table.name.lower()
        mapping[real] = real
        if alias := table.alias:
            mapping[alias.lower()] = real
    for cte in tree.find_all(exp.CTE):
        if cte.alias:
            mapping[cte.alias.lower()] = cte.alias.lower()
    return mapping


def _check_schema(
    tree: exp.Expression, catalog: Catalog
) -> tuple[list[Violation], list[str]]:
    """Verify every identifier exists. This *is* the hallucination detector.

    Deterministic, free, and strictly more reliable than asking a second model whether the
    first one made up a column name.

    Known limitation, stated honestly: column checking is membership-based, scoped by alias
    where an alias is present and otherwise checked against the union of referenced tables.
    It therefore does not catch a column that exists in table A being used in a scope where
    only table B is visible. Full scope resolution would need sqlglot's qualifier and a
    complete schema; the trade was deliberate — this catches every *invented* identifier,
    which is the failure mode that matters, with far less machinery.
    """
    violations: list[Violation] = []
    warnings: list[str] = []

    aliases = _alias_map(tree)
    cte_names = {c.alias.lower() for c in tree.find_all(exp.CTE) if c.alias}
    declared = _defined_aliases(tree)

    #: A single SELECT with no CTEs has exactly one scope, so "the tables referenced
    #: anywhere in the statement" and "the tables visible here" are the same set. Only then
    #: can ambiguity be decided without real scope resolution.
    is_flat_scope = not cte_names and len(list(tree.find_all(exp.Select))) == 1

    referenced_real_tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if name in cte_names:
            continue
        if not catalog.has_table(name):
            violations.append(
                Violation("unknown_table", f"table {table.name!r} does not exist")
            )
        else:
            referenced_real_tables.add(name)

    if not referenced_real_tables:
        return violations, warnings

    for column in tree.find_all(exp.Column):
        name = column.name
        if not name or name == "*":
            continue
        if name.lower() in declared:
            continue  # introduced by this query, not read from a table
        qualifier = (column.table or "").lower()

        if qualifier:
            target = aliases.get(qualifier)
            if target is None:
                violations.append(
                    Violation("unknown_alias", f"alias {column.table!r} is not defined")
                )
                continue
            if target in cte_names:
                continue  # columns of a CTE are validated inside the CTE body
            if not catalog.has_column(target, name):
                violations.append(
                    Violation(
                        "unknown_column",
                        f"column {name!r} does not exist on table {target!r}",
                    )
                )
        else:
            owners = [t for t in referenced_real_tables if catalog.has_column(t, name)]
            if not owners:
                if cte_names:
                    warnings.append(
                        f"column {name!r} not found in base tables; assumed to come from a CTE"
                    )
                else:
                    violations.append(
                        Violation(
                            "unknown_column",
                            f"column {name!r} does not exist on any referenced table",
                        )
                    )
            elif is_flat_scope and len(owners) > 1:
                # Ambiguity is a *warning*, never a rejection, and only in a flat scope.
                #
                # SQLite raises "ambiguous column name" at execution, so this is already
                # caught — the warning just turns a runtime error into a diagnosable one.
                # It is deliberately not a violation: outside a flat scope, the set of
                # referenced tables is the union across the whole statement rather than the
                # visible scope, so `SELECT (SELECT COUNT(*) FROM couriers WHERE courier_id
                # = 5) FROM orders` would be flagged despite being perfectly unambiguous.
                # Rejecting valid queries to catch one the executor already catches is a bad
                # trade; the restriction to a single flat SELECT is what makes the check
                # exact wherever it fires at all.
                warnings.append(
                    f"column {name!r} is ambiguous - it exists on "
                    f"{', '.join(sorted(owners))}; qualify it with a table or alias"
                )
    return violations, warnings


def suggest_identifier(name: str, candidates: frozenset[str] | set[str]) -> str | None:
    """Closest catalog identifier to ``name``, for "did you mean" messages.

    Matching happens on the Arabic-aware normalized form so that a bilingual glossary entry
    still matches when the model echoes a differently-spelled variant.
    """
    import difflib

    target = normalize_for_matching(name)
    pool = {normalize_for_matching(c): c for c in candidates}
    matches = difflib.get_close_matches(target, list(pool), n=1, cutoff=0.75)
    return pool[matches[0]] if matches else None


# -------------------------------------------------------------------------- rewriting


def _ensure_limit(tree: exp.Expression, max_rows: int) -> tuple[str, str]:
    """Return SQL with a LIMIT guaranteed, plus what had to be done to get there.

    The action is one of ``"none"``, ``"injected"`` or ``"lowered"``. An existing tighter
    LIMIT is preserved; a looser one is lowered. The executor caps rows independently, so
    this is convenience and transparency rather than the only thing standing between us and
    a full table scan.

    A non-literal LIMIT (``LIMIT ?`` or ``LIMIT (SELECT ...)``) is left untouched and
    reported as ``"none"``: rewriting an expression we cannot evaluate risks changing the
    query's meaning, and the executor's independent row cap already bounds the damage.
    """
    target = tree
    if isinstance(tree, exp.Subquery):
        target = tree.this
    if not isinstance(target, (exp.Select, exp.Union)):
        return tree.sql(dialect=DIALECT), "none"

    if (existing := target.args.get("limit")) is not None:
        try:
            current = int(existing.expression.name)
        except (AttributeError, TypeError, ValueError):
            return tree.sql(dialect=DIALECT), "none"
        if current <= max_rows:
            return tree.sql(dialect=DIALECT), "none"
        target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
        return tree.sql(dialect=DIALECT), "lowered"

    target.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return tree.sql(dialect=DIALECT), "injected"
