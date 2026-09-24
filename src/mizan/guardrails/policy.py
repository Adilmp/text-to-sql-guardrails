"""The guardrail policy: what a generated query is permitted to do.

Allowlist, not denylist
-----------------------
Function filtering here is an **allowlist**. A denylist is a losing position: SQLite ships
new functions between point releases, builds enable different extensions, and one missed
name (``load_extension``, ``readfile``, ``writefile``, ``edit``) is full compromise. An
allowlist fails closed — an unrecognised function is rejected, and the cost of being wrong
is a false rejection that shows up in the eval numbers rather than a breach that does not.

``DENIED_FUNCTIONS`` still exists, but only as documentation and as a second line of
defence if someone flips ``enforce_function_allowlist`` off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Functions a read-only analytical query legitimately needs. Lower-case, compared after
#: case-folding. Grouped by purpose so the list stays auditable.
ALLOWED_FUNCTIONS: frozenset[str] = frozenset(
    # aggregates
    {"avg", "count", "group_concat", "max", "min", "sum", "total"}
    # conditional / null handling
    | {"coalesce", "ifnull", "iif", "nullif", "case"}
    # string
    | {
        "char",
        "concat",
        "concat_ws",
        "format",
        "hex",
        "instr",
        "length",
        "lower",
        "ltrim",
        "printf",
        "quote",
        "replace",
        "rtrim",
        "substr",
        "substring",
        "trim",
        "unicode",
        "upper",
    }
    # numeric
    | {
        "abs",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "ceiling",
        "cos",
        "degrees",
        "exp",
        "floor",
        "ln",
        "log",
        "log10",
        "log2",
        "mod",
        "pi",
        "pow",
        "power",
        "radians",
        "round",
        "sign",
        "sin",
        "sqrt",
        "tan",
        "trunc",
    }
    # date / time
    | {
        "date",
        "datetime",
        "julianday",
        "strftime",
        "time",
        "timediff",
        "unixepoch",
    }
    # window
    | {
        "cume_dist",
        "dense_rank",
        "first_value",
        "lag",
        "last_value",
        "lead",
        "nth_value",
        "ntile",
        "percent_rank",
        "rank",
        "row_number",
    }
    # json (read-only accessors only)
    | {
        "json",
        "json_array",
        "json_array_length",
        "json_extract",
        "json_group_array",
        "json_group_object",
        "json_object",
        "json_quote",
        "json_type",
        "json_valid",
    }
    # type inspection / casting
    | {"cast", "typeof"}
)

#: Explicitly hostile in a SQLite context. Kept for documentation and defence in depth.
#:
#: * ``load_extension`` loads a shared library — arbitrary code execution.
#: * ``readfile``/``writefile``/``edit`` are filesystem access (shell builtins, but present
#:   in some embeddings).
#: * ``zeroblob``/``randomblob`` allocate attacker-controlled memory: ``zeroblob(2e9)`` is a
#:   one-line denial of service that never touches a table.
#: * ``sqlite_compileoption_*`` and ``sqlite_source_id`` leak build details useful for
#:   fingerprinting the target before a real attack.
DENIED_FUNCTIONS: frozenset[str] = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "zeroblob",
        "randomblob",
        "sqlite_compileoption_get",
        "sqlite_compileoption_used",
        "sqlite_source_id",
        "last_insert_rowid",
        "changes",
        "total_changes",
    }
)


@dataclass(frozen=True)
class GuardrailPolicy:
    """Immutable description of what this deployment permits.

    Frozen because a policy must not be mutated after a run starts — the serialized policy
    written into each run directory has to describe the rules that actually applied.
    """

    #: Reject anything that is not a single SELECT (or a WITH ... SELECT).
    allow_only_select: bool = True
    #: Reject unknown table/column identifiers. Doubles as hallucination detection.
    enforce_schema: bool = True
    #: Reject any function not in ``ALLOWED_FUNCTIONS``.
    enforce_function_allowlist: bool = True

    #: Structural complexity ceilings. These exist to stop an accidental cartesian product
    #: from pinning the CPU, not to stop an attacker — the row cap and timeout do that.
    max_joins: int = 6
    max_subquery_depth: int = 4
    max_union_branches: int = 4

    max_sql_chars: int = 8_000
    max_rows: int = 200

    #: Injected as a LIMIT when the query has none. Defence in depth: the executor caps
    #: rows independently, so a bug here cannot alone cause an unbounded read.
    inject_limit: bool = True

    allowed_functions: frozenset[str] = field(default=ALLOWED_FUNCTIONS)
    denied_functions: frozenset[str] = field(default=DENIED_FUNCTIONS)

    @classmethod
    def permissive(cls) -> GuardrailPolicy:
        """Looser policy used to *measure* the guardrails in evals.

        Running the eval set twice — once strict, once permissive — is what lets the
        README state how many queries the guardrails rejected and how many of those
        rejections were correct. Without this, "0 breaches" is an unfalsifiable claim.
        """
        return cls(
            enforce_function_allowlist=False,
            max_joins=12,
            max_subquery_depth=8,
        )
