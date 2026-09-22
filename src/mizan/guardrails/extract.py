"""Recover a bare SQL statement from whatever the model actually emitted.

This module exists because of a mundane but unavoidable reality: instruction-tuned models
wrap code in Markdown fences, prepend "Here is the query:", append an explanation, and
sometimes emit a fence with no language tag or an unterminated fence when they hit the token
limit. Feeding any of that to a SQL parser produces a parse error that *looks* like a model
capability failure in the eval numbers when it is really a formatting artifact.

Getting this wrong silently depresses every accuracy metric you report, which is why it has
its own module and its own tests rather than being an inline ``.strip("`")``.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(
    r"```[ \t]*(?:sql|sqlite|postgres|postgresql|mysql)?[ \t]*\r?\n(?P<body>.*?)(?:```|\Z)",
    re.DOTALL | re.IGNORECASE,
)

#: Conversational preamble the model sometimes emits before the statement.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is)[^\n:]*:|sure[^\n:]*:|query:|sql:|answer:)\s*",
    re.IGNORECASE,
)

#: Every keyword that can begin a SQL statement — **not** just the ones we permit.
#:
#: This distinction matters more than it looks. Extraction's job is to recover what the
#: model said; deciding whether that is allowed is the validator's job, and only the
#: validator's. An earlier version matched only ``SELECT|WITH``, which meant a generated
#: ``DROP TABLE orders`` was silently trimmed to an empty string and reported as
#: ``parse_error``. The database was never in danger — but the guardrail telemetry claimed
#: the model had emitted gibberish when it had actually attempted a write, which is exactly
#: the signal you cannot afford to lose. Extraction must never sanitise.
_STATEMENT_KEYWORDS = (
    "WITH|SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|"
    "ATTACH|DETACH|PRAGMA|VACUUM|ANALYZE|REINDEX|GRANT|REVOKE|"
    "BEGIN|COMMIT|ROLLBACK|EXPLAIN|SET|USE"
)
_STATEMENT_START_RE = re.compile(rf"\b({_STATEMENT_KEYWORDS})\b", re.IGNORECASE)
_LEADING_STATEMENT_RE = re.compile(rf"^\s*({_STATEMENT_KEYWORDS})\b", re.IGNORECASE)


def extract_sql(raw: str) -> str:
    """Return the first SQL statement found in ``raw``.

    Handles, in order: fenced blocks (closed or truncated), conversational preambles,
    unfenced prose followed by a statement, and trailing explanatory text after the
    terminating semicolon.

    Returns an empty string when nothing statement-like is present; the caller reports that
    as a parse failure rather than guessing.
    """
    if not raw or not raw.strip():
        return ""

    text = raw.strip()

    # 1. Prefer a fenced block. `\Z` in the pattern means a fence the model opened but never
    #    closed (token limit) still yields its body instead of falling through to prose
    #    handling, which would otherwise swallow the whole response.
    if match := _FENCE_RE.search(text):
        candidate = match.group("body").strip()
        if candidate:
            text = candidate

    text = _PREAMBLE_RE.sub("", text).strip()

    # 2. If the text still does not begin with a statement keyword, find where one starts.
    #    Anything before that point is prose.
    if not _LEADING_STATEMENT_RE.match(text):
        if start := _STATEMENT_START_RE.search(text):
            text = text[start.start() :]
        else:
            return ""

    # 3. Cut at the first semicolon. Everything after it is either explanation or a stacked
    #    statement; both are the caller's problem to report, not ours to silently execute.
    #    The statement-count check in the validator still sees the original text, so a
    #    stacked query is *reported*, never quietly trimmed away here.
    text = _strip_trailing_prose(text)

    return text.strip().rstrip(";").strip()


def _strip_trailing_prose(text: str) -> str:
    """Drop explanatory text that follows the statement's terminating semicolon.

    Semicolons inside string literals must not count. A single pass tracking quote state is
    enough: SQLite escapes a quote by doubling it, which this handles naturally because the
    doubled quote toggles the flag off and immediately back on.
    """
    in_single = False
    in_double = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == ";" and not in_single and not in_double:
            return text[:i]
    return text


def has_multiple_statements(text: str) -> bool:
    """Whether ``text`` contains more than one statement, ignoring quoted semicolons.

    Used by the validator to report stacked queries explicitly. ``extract_sql`` trims them,
    so without this check a stacked-query attempt would be scored as a clean success.
    """
    remainder = text.strip()
    body = _strip_trailing_prose(remainder)
    tail = remainder[len(body) :].lstrip(";").strip()
    return bool(tail) and bool(re.search(r"[A-Za-z]", tail))
