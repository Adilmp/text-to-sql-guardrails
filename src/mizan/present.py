"""Presenting a result to a person: a one-line answer and the chart that fits it.

The one-line answer is built from the result, never written by the model
--------------------------------------------------------------------------
Asking the model to phrase "188 orders were delivered late" would cost another generation
(seconds on a CPU) and give it a chance to get the number wrong. Instead the question is
restated around the number the query actually returned, with templates per language. Urdu is
the easy case: "how many" (کتنے) sits in front of its noun, so replacing it with the number
reads naturally ("کتنے آرڈر تاخیر سے ڈیلیور ہوئے؟" → "188 آرڈر تاخیر سے ڈیلیور ہوئے۔").
When no template fits, the answer is still a plain line in the question's language
("Answer: 188").

One cost, stated plainly: a sentence makes an answer more convincing, including a wrong one.
The sentence never adds anything the result doesn't contain, and the page shows it above the
SQL and the confidence it came from (DECISIONS.md D41).

Charts are chosen here, drawn in the browser
--------------------------------------------
A result with one label column and a number column, and between 2 and 50 rows, gets a chart:
a line when the labels are months or years, bars otherwise. Deciding here keeps the rule in
Python, where it is tested, and the page only draws.
"""

from __future__ import annotations

import re
from typing import Any

from .guardrails import QueryResult
from .nl import Script, clean_for_model

#: Labels that read as time, so the chart should be a line: 2025, 2025-01, 2025-01-31, 01..12.
_TIME_LABEL_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$|^(0?[1-9]|1[0-2])$")
_TIME_COLUMN_RE = re.compile(r"month|year|date|day|week", re.IGNORECASE)

#: How a line opens when no template fits, and the words for row summaries, per language.
_WORDS = {
    "en": {
        "answer": "Answer",
        "none": "No matching rows.",
        "rows": "{n} rows",
        "highest": "highest",
        "lowest": "lowest",
        "more": "and {n} more",
        "first": "first {n} rows shown",
        "sep": ", ",
        "end": ".",
    },
    "ar": {
        "answer": "الجواب",
        "none": "لا توجد نتائج مطابقة.",
        "rows": "عدد النتائج: {n}",
        "highest": "الأعلى",
        "lowest": "الأدنى",
        "more": "و{n} أخرى",
        "first": "تُعرض أول {n} نتيجة",
        "sep": "، ",
        "end": ".",
    },
    "ur": {
        "answer": "جواب",
        "none": "کوئی نتیجہ نہیں ملا۔",
        "rows": "{n} نتائج",
        "highest": "سب سے زیادہ",
        "lowest": "سب سے کم",
        "more": "اور {n} مزید",
        "first": "پہلے {n} نتائج دکھائے گئے",
        "sep": "، ",
        "end": "۔",
    },
}

# English: "how many orders were delivered late?" -> "188 orders were delivered late."
# "do/does/did" are left out on purpose: "how many orders did Gulf Express handle" has no
# simple statement form.
_EN_THERE = re.compile(r"^how many (?P<thing>.+?) (?:are|is|were|was) there$", re.IGNORECASE)
_EN_HOW_MANY = re.compile(
    r"^how many (?P<thing>.+?) (?P<verb>are|were|is|was|have|has|had) (?P<rest>.+)$",
    re.IGNORECASE,
)
_EN_WHAT_IS = re.compile(r"^what (?:is|was) (?P<thing>the .+)$", re.IGNORECASE)
# Arabic: "كم عدد الطلبات المتأخرة؟" -> "عدد الطلبات المتأخرة: 188"
_AR_HOW_MANY = re.compile(r"^كم (?P<thing>عدد .+)$")
_AR_WHAT = re.compile(r"^(?:ما|كم) (?:هو |هي )?(?P<thing>.+)$")
# Urdu: replace the first "how many / how much" with the number.
_UR_HOW_MANY = re.compile(r"(?<!\S)(?:کتنے|کتنی|کتنا)(?!\S)")
_FILLER_RE = re.compile(r"^(?:please|kindly)[,\s]+", re.IGNORECASE)


def format_number(value: Any) -> str:
    """``1234`` -> ``1,234``; ``763.0690909`` -> ``763.07``; ``18376.5`` -> ``18,376.5``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _language(script: Script) -> str:
    return {"en": "en", "ur": "ur"}.get(script.prompt_language, "ar")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_columns(result: QueryResult) -> list[int]:
    return [
        i
        for i in range(len(result.columns))
        if result.rows
        and all(_is_number(row[i]) or row[i] is None for row in result.rows)
        and any(row[i] is not None for row in result.rows)
    ]


def summarize(question: str, script: Script, result: QueryResult) -> str:
    """A one-line answer in the question's language, built only from ``result``."""
    lang = _language(script)
    words = _WORDS[lang]
    if not result.rows:
        return words["none"]

    numeric = _numeric_columns(result)
    if len(result.rows) == 1 and len(result.columns) == 1:
        value = result.rows[0][0]
        if value is None:
            return words["none"]
        return _restate(question, lang, format_number(value)) or (
            f"{words['answer']}: {format_number(value)}"
        )

    labels = [i for i in range(len(result.columns)) if i not in numeric]
    if len(result.rows) == 1:
        row = result.rows[0]
        label = " ".join(str(row[i]) for i in labels if row[i] is not None)
        values = words["sep"].join(format_number(row[i]) for i in numeric)
        return f"{label} — {values}" if label and values else label or values

    line = _rows_line(result, labels, numeric, words)
    if result.truncated:
        line += f" ({words['first'].format(n=len(result.rows))})"
    return line


def _rows_line(
    result: QueryResult, labels: list[int], numeric: list[int], words: dict[str, str]
) -> str:
    n = len(result.rows)

    def label_of(row: tuple[Any, ...]) -> str:
        return " ".join(str(row[i]) for i in labels if row[i] is not None)

    if len(labels) == 1 and len(numeric) == 1:
        lab, num = labels[0], numeric[0]
        pairs = [(label_of(r), r[num]) for r in result.rows if r[num] is not None]
        if n <= 3:
            return words["sep"].join(f"{a} ({format_number(v)})" for a, v in pairs)
        top = max(pairs, key=lambda p: p[1])
        low = min(pairs, key=lambda p: p[1])
        del lab
        return (
            f"{words['rows'].format(n=n)} · {words['highest']}: {top[0]} "
            f"({format_number(top[1])}) · {words['lowest']}: {low[0]} ({format_number(low[1])})"
        )
    if len(result.columns) == 1:
        names = [str(r[0]) for r in result.rows[:5]]
        line = words["sep"].join(names)
        return f"{line} {words['more'].format(n=n - 5)}" if n > 5 else line
    return words["rows"].format(n=n)


def _restate(question: str, lang: str, number: str) -> str | None:
    """The question turned into a statement around ``number``, or ``None``."""
    text = _FILLER_RE.sub("", clean_for_model(question)).rstrip(" ?؟.۔").strip()
    if lang == "en":
        if match := _EN_THERE.match(text):
            return f"There are {number} {match['thing']}."
        if match := _EN_HOW_MANY.match(text):
            return f"{number} {match['thing']} {match['verb']} {match['rest']}."
        if match := _EN_WHAT_IS.match(text):
            thing = match["thing"]
            return f"{thing[0].upper()}{thing[1:]} is {number}."
        return None
    if lang == "ur":
        if _UR_HOW_MANY.search(text):
            return _UR_HOW_MANY.sub(number, text, count=1) + "۔"
        return None
    if match := _AR_HOW_MANY.match(text):
        return f"{match['thing']}: {number}"
    if match := _AR_WHAT.match(text):
        return f"{match['thing']}: {number}"
    return None


def chart_spec(result: QueryResult) -> dict[str, Any] | None:
    """Which chart fits ``result``, if any: ``{"type", "label", "value"}`` (column indexes)."""
    n = len(result.rows)
    if not 2 <= n <= 50 or len(result.columns) < 2:
        return None
    numeric = _numeric_columns(result)
    labels = [i for i in range(len(result.columns)) if i not in numeric]
    if len(labels) == 1 and numeric:
        label = labels[0]
    elif not labels and len(numeric) >= 2 and _TIME_COLUMN_RE.search(result.columns[0]):
        label = 0  # a numeric month or year column: CAST(strftime('%m', …) AS INTEGER)
        numeric = numeric[1:]
    else:
        return None
    value = numeric[-1] if numeric else None
    if value is None:
        return None
    is_time = _TIME_COLUMN_RE.search(result.columns[label]) or all(
        _TIME_LABEL_RE.match(str(row[label])) for row in result.rows
    )
    return {"type": "line" if is_time else "bar", "label": label, "value": value}
