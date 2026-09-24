"""Database catalog: introspection, the bilingual glossary, and prompt rendering.

Three responsibilities, deliberately kept in one place because they must never disagree:

1. **Introspection** reads the real structure out of SQLite (``PRAGMA table_info``,
   ``PRAGMA foreign_key_list``). This is ground truth.
2. **The glossary** layers human and Arabic-language meaning on top — what a column means,
   and which Arabic words refer to it. It lives in a separate JSON file because it is
   curated content, not derived data, and must survive a database rebuild.
3. **Prompt rendering** turns both into the schema card the model sees.

The reason these share a class is the hallucination detector: it validates identifiers
against exactly the same object the prompt was rendered from. If the prompt and the
validator could ever be built from different sources, the model could be *told* about a
column that the validator then rejects — a bug that would look like a model failure and be
extremely hard to diagnose.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SchemaError
from ..logging import get_logger
from ..nl.normalize import normalize_for_matching

logger = get_logger("schema")

#: Characters permitted in a value rendered into the prompt. Latin letters and digits,
#: Arabic letters (categorical values here are legitimately bilingual), spaces and a small
#: set of separators. Quotes, semicolons, backslashes and newlines are excluded.
_SAFE_SAMPLE_RE = re.compile(r"[\w \-./؀-ۿ]{1,40}", re.UNICODE)

#: Substrings that are never part of a category label but are meaningful to a SQL parser.
#: ``-`` alone is legitimate (``in-transit``); ``--`` opens a comment.
_UNSAFE_SEQUENCES = ("--", "/*", "*/", "||")


def quote_identifier(name: str) -> str:
    """Quote a SQLite identifier, escaping any embedded double quote.

    Identifiers cannot be bound as query parameters, so introspection has to interpolate
    table and column names into SQL. Wrapping them in double quotes is necessary but *not
    sufficient*: SQLite permits a double quote inside an identifier, written doubled —
    ``CREATE TABLE "a""b" (x)`` is legal and is stored in ``sqlite_master`` as ``a"b``.

    Naive interpolation of that name produces ``SELECT COUNT(*) FROM "a"b"``, which is an
    injection point. It is narrow — the attacker must control a table *name*, which means
    supplying the database file — but ``MIZAN_DB_PATH`` accepts any SQLite file, and
    introspection runs *before* any guardrail, on a connection the validator never sees.
    A crafted file would therefore execute attacker-chosen SQL during catalog loading.

    Doubling embedded quotes is the documented SQLite escaping rule and closes it.
    Found by static analysis (ruff's flake8-bandit ``S608``), which is precisely the class
    of bug that is invisible on review and obvious to a scanner.
    """
    return '"' + name.replace('"', '""') + '"'


def _is_prompt_safe(value: str) -> bool:
    """Whether a database value may be rendered into the prompt.

    **Scope, stated precisely — this control stops *syntactic* injection only.**

    It blocks values that carry SQL-meaningful characters or sequences: quotes, semicolons,
    comment markers, newlines, concatenation operators. Those are the payloads that try to
    break out of the quoted context they are rendered in.

    It does **not** and cannot block *semantic* injection. A value of ``ignore all rules``
    is pure letters and spaces — indistinguishable by character class from a legitimate
    category label like ``in transit``. No charset filter can separate those, because the
    difference is meaning, not form.

    What covers the residual risk instead:

    * The AST guardrails still reject every destructive statement, whatever talked the
      model into producing it. A semantic injection cannot cause data loss.
    * What it *could* cause is a legal-but-wrong ``SELECT`` — steering the model toward the
      wrong rows. Nothing syntactic catches that, and this is recorded as a known
      limitation rather than papered over.
    * The practical mitigation is upstream: do not sample columns that accept unvalidated
      user input. That is a deployment decision, not something this function can enforce.
    """
    return bool(_SAFE_SAMPLE_RE.fullmatch(value)) and not any(
        seq in value for seq in _UNSAFE_SEQUENCES
    )


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    nullable: bool = True
    primary_key: bool = False
    description: str = ""
    #: Arabic words or phrases a user might use for this column, e.g. ``الكمية`` for
    #: ``quantity``. Matched after :func:`normalize_for_matching`.
    aliases_ar: tuple[str, ...] = ()
    #: Sample values, rendered into the prompt for low-cardinality categorical columns so
    #: the model emits ``'delivered'`` rather than guessing ``'DELIVERED'``.
    sample_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    description: str = ""
    aliases_ar: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()
    row_count: int = 0

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(c.name.lower() for c in self.columns)

    def column(self, name: str) -> Column | None:
        lowered = name.lower()
        return next((c for c in self.columns if c.name.lower() == lowered), None)


@dataclass
class Catalog:
    """Everything the generator and the validator know about the database."""

    tables: dict[str, Table] = field(default_factory=dict)
    source_path: Path | None = None

    # ------------------------------------------------------------------ lookups

    def has_table(self, name: str) -> bool:
        return name.lower() in self.tables

    def has_column(self, table: str, column: str) -> bool:
        tbl = self.tables.get(table.lower())
        return bool(tbl and column.lower() in tbl.column_names)

    def table(self, name: str) -> Table:
        try:
            return self.tables[name.lower()]
        except KeyError as exc:
            raise SchemaError(f"unknown table {name!r}", known=sorted(self.tables)) from exc

    @property
    def all_table_names(self) -> frozenset[str]:
        return frozenset(self.tables)

    @property
    def all_column_names(self) -> frozenset[str]:
        return frozenset(c for t in self.tables.values() for c in t.column_names)

    # -------------------------------------------------------------- introspection

    @classmethod
    def from_sqlite(cls, path: Path, *, sample_limit: int = 8) -> Catalog:
        """Read structure from a SQLite file. Opens strictly read-only."""
        if not path.exists():
            raise SchemaError(f"database not found: {path}", path=str(path))

        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            tables: dict[str, Table] = {}

            names = [
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
                )
            ]
            for name in names:
                columns = cls._introspect_columns(conn, name, sample_limit)
                fks = tuple(
                    ForeignKey(
                        column=row["from"],
                        references_table=row["table"],
                        references_column=row["to"],
                    )
                    for row in conn.execute(f"PRAGMA foreign_key_list({quote_identifier(name)})")
                )
                count_row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM {quote_identifier(name)}"  # noqa: S608
                ).fetchone()
                tables[name.lower()] = Table(
                    name=name,
                    columns=columns,
                    foreign_keys=fks,
                    row_count=int(count_row["n"]) if count_row else 0,
                )
            logger.debug("catalog introspected", extra={"tables": len(tables)})
            return cls(tables=tables, source_path=path)
        finally:
            conn.close()

    @staticmethod
    def _introspect_columns(
        conn: sqlite3.Connection, table: str, sample_limit: int
    ) -> tuple[Column, ...]:
        # Identifiers cannot be bound as parameters in SQLite, so the table name is
        # interpolated. It is safe here specifically because it came from sqlite_master
        # rather than from user input — but it is quoted anyway so that a table named with
        # a reserved word or a space still works.
        info = list(conn.execute(f"PRAGMA table_info({quote_identifier(table)})"))
        columns: list[Column] = []
        for row in info:
            col_name = row["name"]
            samples = Catalog._sample_values(conn, table, col_name, row["type"], sample_limit)
            columns.append(
                Column(
                    name=col_name,
                    type=(row["type"] or "").upper() or "TEXT",
                    nullable=not row["notnull"],
                    primary_key=bool(row["pk"]),
                    sample_values=samples,
                )
            )
        return tuple(columns)

    @staticmethod
    def _sample_values(
        conn: sqlite3.Connection, table: str, column: str, decl_type: str, limit: int
    ) -> tuple[str, ...]:
        """Distinct values for low-cardinality text columns only.

        Rationale: showing the model that ``status`` is one of
        ``{pending, in_transit, delivered, returned}`` removes an entire class of error
        where it invents a plausible-but-absent literal such as ``'shipped'``. Doing the
        same for a free-text or numeric column would leak data into the prompt and bloat it
        for no benefit, so both are skipped.
        """
        if "CHAR" not in decl_type.upper() and "TEXT" not in decl_type.upper():
            return ()
        try:
            col, tbl = quote_identifier(column), quote_identifier(table)
            rows = conn.execute(
                f"SELECT DISTINCT {col} AS v FROM {tbl} "  # noqa: S608
                f"WHERE {col} IS NOT NULL LIMIT ?",
                (limit + 1,),
            ).fetchall()
        except sqlite3.Error:
            return ()
        if len(rows) > limit:
            return ()  # high cardinality: a sample would mislead more than it helps
        values = [str(r["v"]) for r in rows]
        # Long values are free text, not a category.
        if any(len(v) > 40 for v in values):
            return ()

        # Content allowlist — this is a security control, not a formatting nicety.
        #
        # Sample values are read from the database and rendered verbatim into the system
        # prompt, directly above an instruction telling the model to use them "exactly as
        # written". That makes any column sampled here a **stored prompt-injection
        # channel**: anyone who can write a row (a signup form, an imported CSV, a
        # partner feed) can plant text that reaches every later user's prompt.
        #
        # Security testing confirmed this was exploitable. A value of
        # ``'; DROP TABLE t--`` was rendered into the prompt intact. The AST guardrails
        # still block destructive SQL, so the blast radius is not data loss — but they
        # cannot stop *semantic* manipulation, where the model is steered into a perfectly
        # legal SELECT that returns the wrong rows. No syntactic check catches that.
        #
        # A genuine categorical label is alphanumeric: `delivered`, `in_transit`,
        # `enterprise`. Quotes, semicolons, comment markers and newlines are not category
        # names, so the allowlist costs nothing legitimate. Consistent with the function
        # policy (see guardrails/policy.py), this fails closed on anything unrecognised.
        if not all(_is_prompt_safe(v) for v in values):
            # Drop the whole set, not just the offending value: showing a partial value
            # domain would mislead the model about what the column can contain. An
            # attacker can therefore suppress a helpful hint, which is a far smaller harm
            # than injecting one — and the warning makes it visible rather than silent.
            logger.warning(
                "sample values withheld: unsafe characters for prompt inclusion",
                extra={"table": table, "column": column},
            )
            return ()
        return tuple(sorted(values))

    # ------------------------------------------------------------------ glossary

    def apply_glossary(self, glossary_path: Path) -> None:
        """Merge curated bilingual metadata onto the introspected structure.

        Unknown tables or columns in the glossary are a hard error rather than a warning:
        a glossary that has drifted out of sync with the schema will silently degrade
        Arabic matching, and a loud failure at startup is far cheaper than debugging why
        ``الشحنات`` stopped resolving three weeks later.
        """
        if not glossary_path.exists():
            logger.warning("no glossary found", extra={"path": str(glossary_path)})
            return

        data = json.loads(glossary_path.read_text(encoding="utf-8"))
        for table_name, entry in data.get("tables", {}).items():
            key = table_name.lower()
            if key not in self.tables:
                raise SchemaError(
                    f"glossary references unknown table {table_name!r}",
                    known=sorted(self.tables),
                )
            existing = self.tables[key]
            new_columns: list[Column] = []
            col_entries = entry.get("columns", {})
            for col in existing.columns:
                meta = col_entries.get(col.name, {})
                new_columns.append(
                    Column(
                        name=col.name,
                        type=col.type,
                        nullable=col.nullable,
                        primary_key=col.primary_key,
                        description=meta.get("description", col.description),
                        aliases_ar=tuple(meta.get("ar", ())),
                        sample_values=col.sample_values,
                    )
                )
            unknown = set(col_entries) - {c.name for c in existing.columns}
            if unknown:
                raise SchemaError(
                    f"glossary references unknown columns on {table_name!r}",
                    columns=sorted(unknown),
                )
            self.tables[key] = Table(
                name=existing.name,
                columns=tuple(new_columns),
                description=entry.get("description", existing.description),
                aliases_ar=tuple(entry.get("ar", ())),
                foreign_keys=existing.foreign_keys,
                row_count=existing.row_count,
            )
        logger.debug("glossary applied", extra={"path": str(glossary_path)})

    def arabic_index(self) -> dict[str, tuple[str, str | None]]:
        """Normalized Arabic alias -> (table, column or None).

        Built once and reused; the key is the *matching* normalization, so ``الكميّة`` with
        a shadda and ``الكمية`` without one resolve to the same entry.
        """
        index: dict[str, tuple[str, str | None]] = {}
        for table in self.tables.values():
            for alias in table.aliases_ar:
                index[normalize_for_matching(alias)] = (table.name, None)
            for column in table.columns:
                for alias in column.aliases_ar:
                    index[normalize_for_matching(alias)] = (table.name, column.name)
        return index

    # ------------------------------------------------------------ prompt rendering

    def to_prompt(self, *, include_arabic: bool, include_samples: bool = True) -> str:
        """Render the schema card the model is shown.

        Format is CREATE-TABLE-like rather than JSON or prose. Models have seen orders of
        magnitude more DDL than any bespoke schema format during pretraining, so DDL is the
        representation they follow most reliably — and it costs fewer tokens than JSON.
        """
        blocks: list[str] = []
        for table in sorted(self.tables.values(), key=lambda t: t.name):
            header = f"CREATE TABLE {table.name} ("
            lines: list[str] = []
            for col in table.columns:
                bits = [f"  {col.name} {col.type}"]
                if col.primary_key:
                    bits.append("PRIMARY KEY")
                if not col.nullable:
                    bits.append("NOT NULL")
                line = " ".join(bits)

                notes: list[str] = []
                if col.description:
                    notes.append(col.description)
                if include_arabic and col.aliases_ar:
                    notes.append("ar: " + " / ".join(col.aliases_ar))
                if include_samples and col.sample_values:
                    notes.append("one of: " + ", ".join(repr(v) for v in col.sample_values))
                if notes:
                    line += f"  -- {'; '.join(notes)}"
                lines.append(line)

            for fk in table.foreign_keys:
                lines.append(
                    f"  FOREIGN KEY ({fk.column}) "
                    f"REFERENCES {fk.references_table}({fk.references_column})"
                )

            block = header + "\n" + ",\n".join(lines) + "\n);"
            comment_bits: list[str] = []
            if table.description:
                comment_bits.append(table.description)
            if include_arabic and table.aliases_ar:
                comment_bits.append("ar: " + " / ".join(table.aliases_ar))
            comment_bits.append(f"{table.row_count} rows")
            block = f"-- {'; '.join(comment_bits)}\n{block}"
            blocks.append(block)
        return "\n\n".join(blocks)
