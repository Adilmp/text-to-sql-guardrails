"""Authorization: who is asking decides which tables, columns and rows they can read.

The guardrails decide *what statements run*. This module decides *whose data they return*.
A role in ``<db>.roles.json`` lists the tables (and optionally columns) it can read, columns
it must never read, and row filters: a condition every row of a table must meet, written
with the requester's own attributes (``country = :_user_country``).

Four layers, each enough on its own for what it covers (DECISIONS.md D45):

1. **Prompt slice** (``Catalog.restricted``): the model is only shown what the role can read.
2. **Validator** (``validate(..., role=)``): a query touching anything else is rejected as
   ``not_authorized``. The rejection is final; the repair loop must not help anyone probe.
3. **Row filters** (:func:`apply_row_filters`): after validation, every reference to a
   filtered table becomes ``(SELECT * FROM t WHERE <filter>) AS <alias>``, through joins,
   subqueries, CTEs and UNION branches. The ``:_user_*`` values come from the principal;
   the validator rejects any ``:_user_*`` the model writes itself.
4. **Runtime** (executor): the SQLite connection refuses reads the role doesn't allow.

Identity
--------
Only a bearer token says who is asking, and the roles file holds its SHA-256, never the
token. Nothing in a request body is trusted for identity. With no roles file there is no
authorization at all: the single-user local tool behaves exactly as before, as ``ADMIN``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlglot import exp

from .errors import MizanError, SQLParseError
from .guardrails.validator import DIALECT, parse_sql

#: Row-filter parameters are the requester's attributes: ``:_user_<attribute>``.
_USER_PARAM_RE = re.compile(r"_user_(\w+)")


class Unauthorized(MizanError):
    """No valid bearer token: the request is refused before anything runs."""

    code = "unauthorized"


@dataclass(frozen=True)
class RolePolicy:
    """What one role can read."""

    name: str
    #: Readable tables, each with its readable columns (``None`` = every column). ``None``
    #: for the whole mapping means every table.
    tables: Mapping[str, frozenset[str] | None] | None = None
    #: ``(table, column)`` pairs never readable, whatever ``tables`` says.
    deny_columns: frozenset[tuple[str, str]] = frozenset()
    #: Table -> SQL condition every visible row must meet.
    row_filters: Mapping[str, str] = field(default_factory=dict)

    def can_read(self, table: str, column: str | None = None) -> bool:
        t = table.lower()
        if self.tables is not None and t not in self.tables:
            return False
        if column is None:
            return True
        c = column.lower()
        if (t, c) in self.deny_columns:
            return False
        allowed = self.tables[t] if self.tables is not None else None
        return allowed is None or c in allowed

    def hides_columns(self, table: str) -> bool:
        """Whether some column of a readable ``table`` is not readable (``SELECT *`` check)."""
        t = table.lower()
        if any(dt == t for dt, _ in self.deny_columns):
            return True
        return self.tables is not None and self.tables.get(t) is not None

    @property
    def filter_attributes(self) -> frozenset[str]:
        """The requester attributes this role's row filters need."""
        return frozenset(
            match.group(1)
            for condition in self.row_filters.values()
            for name in _placeholders(condition)
            if (match := _USER_PARAM_RE.fullmatch(name))
        )


@dataclass(frozen=True)
class Principal:
    """Who is asking: their role, their attributes, and a stable id for their token."""

    role: RolePolicy
    attrs: Mapping[str, str] = field(default_factory=dict)
    #: Identifies the token (a prefix of its hash) without revealing it. Used to keep one
    #: requester's answers away from another (follow-ups, feedback).
    subject: str = ""


#: Everything, no filters. The principal of the local tool when there is no roles file.
ADMIN = Principal(RolePolicy("admin"), {}, "local")


class Authorizer:
    """Turns a bearer token into a :class:`Principal`, using a roles file."""

    def __init__(self, roles: Mapping[str, RolePolicy], tokens: Mapping[str, Principal]):
        self.roles = dict(roles)
        self._tokens = dict(tokens)

    @classmethod
    def from_file(cls, path: Path) -> Authorizer:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Authorizer:
        """Build from the roles file's contents. Anything malformed raises ``ValueError``."""
        roles = {name: _parse_role(name, spec) for name, spec in data.get("roles", {}).items()}
        tokens: dict[str, Principal] = {}
        for digest, spec in data.get("tokens", {}).items():
            role = roles.get(spec.get("role"))
            if role is None:
                raise ValueError(
                    f"token {digest[:8]}... names an unknown role {spec.get('role')!r}"
                )
            attrs = {str(k): str(v) for k, v in (spec.get("attrs") or {}).items()}
            if missing := sorted(role.filter_attributes - attrs.keys()):
                raise ValueError(
                    f"token {digest[:8]}... (role {role.name!r}) lacks the attribute(s) its "
                    f"row filters need: {', '.join(missing)}"
                )
            tokens[digest.lower()] = Principal(role, attrs, digest.lower()[:12])
        return cls(roles, tokens)

    def authenticate(self, header: str | None) -> Principal:
        """The principal for an ``Authorization: Bearer <token>`` header, or raise."""
        scheme, _, token = (header or "").strip().partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            raise Unauthorized("a bearer token is required")
        principal = self._tokens.get(hashlib.sha256(token.encode("utf-8")).hexdigest())
        if principal is None:
            raise Unauthorized("unknown token")
        return principal


def load_authorizer(db_path: Path) -> Authorizer | None:
    """The authorizer from ``<db>.roles.json`` beside the database, or ``None`` if absent."""
    path = db_path.with_suffix(".roles.json")
    return Authorizer.from_file(path) if path.exists() else None


def apply_row_filters(sql: str, principal: Principal) -> tuple[str, dict[str, str]]:
    """Rewrite every reference to a filtered table, and the parameters to bind.

    Runs on SQL that has already passed validation. Every ``exp.Table`` node naming a
    filtered table is replaced, wherever it is (a join, a subquery, a CTE body, a UNION
    branch); the alias is kept, or the table's own name used as one, so qualified and
    unqualified column references keep working. The filter's own subqueries are added
    after the nodes were collected, so they are never rewritten again.
    """
    filters = {t.lower(): c for t, c in principal.role.row_filters.items()}
    if not filters:
        return sql, {}
    tree = parse_sql(sql)
    targets = [t for t in tree.find_all(exp.Table) if t.name.lower() in filters]
    if not targets:
        return sql, {}
    used: set[str] = set()
    for table in targets:
        name = table.name.lower()
        condition = filters[name]
        used |= set(_placeholders(condition))
        inner = (
            exp.select("*").from_(exp.to_identifier(table.name)).where(condition, dialect=DIALECT)
        )
        table.replace(
            exp.Subquery(
                this=inner, alias=exp.TableAlias(this=exp.to_identifier(table.alias or table.name))
            )
        )
    params = {
        name: principal.attrs[match.group(1)]
        for name in sorted(used)
        if (match := _USER_PARAM_RE.fullmatch(name))
    }
    return tree.sql(dialect=DIALECT), params


# ----------------------------------------------------------------------------- internals


def _placeholders(condition: str) -> list[str]:
    tree = parse_sql(f"SELECT 1 WHERE {condition}")
    return [p.name for p in tree.find_all(exp.Placeholder)]


def _parse_role(name: str, spec: Mapping[str, Any]) -> RolePolicy:
    raw_tables = spec.get("tables", "*")
    tables: dict[str, frozenset[str] | None] | None
    if raw_tables == "*":
        tables = None
    elif isinstance(raw_tables, dict):
        tables = {
            t.lower(): None if cols == "*" else frozenset(c.lower() for c in cols)
            for t, cols in raw_tables.items()
        }
    else:
        raise ValueError(f'role {name!r}: tables must be "*" or an object')

    deny: set[tuple[str, str]] = set()
    for item in spec.get("deny_columns", []):
        table, dot, column = str(item).partition(".")
        if not dot or not table or not column:
            raise ValueError(f"role {name!r}: deny_columns entry {item!r} is not table.column")
        deny.add((table.lower(), column.lower()))

    row_filters: dict[str, str] = {}
    for table, condition in (spec.get("row_filters") or {}).items():
        try:
            names = _placeholders(condition)
        except SQLParseError as exc:
            raise ValueError(
                f"role {name!r}: row filter on {table!r} is not one condition"
            ) from exc
        for param in names:
            if not _USER_PARAM_RE.fullmatch(param):
                raise ValueError(
                    f"role {name!r}: row filter on {table!r} uses :{param}; filter parameters "
                    "are requester attributes, written :_user_<attribute>"
                )
        row_filters[table.lower()] = condition
    return RolePolicy(name, tables, frozenset(deny), row_filters)
