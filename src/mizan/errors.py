"""Exception hierarchy.

Design note
-----------
Every exception carries a stable, machine-readable ``code``. The API layer maps codes to
HTTP responses and the eval harness buckets failures by code, so neither has to parse
human-readable messages. Changing a message is safe; changing a code is a breaking change.
"""

from __future__ import annotations

from typing import Any


class MizanError(Exception):
    """Base for every error this package raises deliberately.

    Catching ``MizanError`` is guaranteed to catch our failures without also swallowing
    genuine bugs such as ``TypeError`` or ``AttributeError``.
    """

    code: str = "mizan_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({details})"


# --------------------------------------------------------------------------- config


class ConfigError(MizanError):
    code = "config_error"


# ------------------------------------------------------------------------- providers


class ProviderError(MizanError):
    code = "provider_error"


class ProviderUnavailable(ProviderError):
    """The backend could not be reached at all (connection refused, DNS, model missing)."""

    code = "provider_unavailable"


class ProviderTimeout(ProviderError):
    code = "provider_timeout"


class ProviderResponseError(ProviderError):
    """Backend responded, but with something we cannot use (bad status, unparseable body)."""

    code = "provider_response_error"


# ---------------------------------------------------------------------------- schema


class SchemaError(MizanError):
    code = "schema_error"


class UnknownTable(SchemaError):
    code = "unknown_table"


class UnknownColumn(SchemaError):
    code = "unknown_column"


# ------------------------------------------------------------------------ guardrails


class GuardrailViolation(MizanError):
    """Generated SQL was rejected before execution.

    ``rule`` identifies which specific guardrail fired, so the eval harness can report a
    breakdown (how many rejections were writes vs. stacked queries vs. unknown columns)
    rather than a single opaque "blocked" count.
    """

    code = "guardrail_violation"

    def __init__(self, message: str, rule: str, **context: Any) -> None:
        super().__init__(message, rule=rule, **context)
        self.rule = rule


class SQLParseError(MizanError):
    """The model emitted something that is not parseable SQL at all."""

    code = "sql_parse_error"


# --------------------------------------------------------------------------- execution


class ExecutionError(MizanError):
    code = "execution_error"


class QueryTimeout(ExecutionError):
    code = "query_timeout"


class RowLimitExceeded(ExecutionError):
    code = "row_limit_exceeded"
