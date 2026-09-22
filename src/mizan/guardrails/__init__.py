"""SQL guardrails: extraction, AST validation, and safe execution."""

from __future__ import annotations

from .executor import QueryResult, execute
from .extract import extract_sql, has_multiple_statements
from .policy import ALLOWED_FUNCTIONS, DENIED_FUNCTIONS, GuardrailPolicy
from .validator import GuardrailReport, Violation, parse_sql, suggest_identifier, validate

__all__ = [
    "ALLOWED_FUNCTIONS",
    "DENIED_FUNCTIONS",
    "GuardrailPolicy",
    "GuardrailReport",
    "QueryResult",
    "Violation",
    "execute",
    "extract_sql",
    "has_multiple_statements",
    "parse_sql",
    "suggest_identifier",
    "validate",
]
