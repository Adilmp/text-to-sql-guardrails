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

import contextlib
import re
from typing import Any

import sqlglot
from sqlglot import exp

from .errors import MizanError
from .guardrails import QueryResult, parse_sql
from .nl import Script, clean_for_model
from .schema.catalog import Catalog

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


# ------------------------------------------------------------------------------ explain

#: Words for the explanation, per language. Fragments joined with " · ", not sentences:
#: short enough to read at a glance, and no grammar to get wrong in three languages.
_EXPLAIN = {
    "en": {
        "count": "counted {x}",
        "count_distinct": "counted different {x}",
        "sum": "added up {x}",
        "avg": "averaged {x}",
        "max": "highest {x}",
        "min": "lowest {x}",
        "list": "listed {x}",
        "with": "with {x}",
        "only": "only {x}",
        "per": "per {x}",
        "top": "top {n}",
        "bottom": "bottom {n}",
        "empty": "{x} is empty",
        "set": "{x} is set",
        "calc": "a calculated value",
        "sep": ", ",
    },
    "ar": {
        "count": "عدّ {x}",
        "count_distinct": "عدّ {x} المختلفة",
        "sum": "جمع {x}",
        "avg": "متوسط {x}",
        "max": "أعلى {x}",
        "min": "أدنى {x}",
        "list": "عرض {x}",
        "with": "مع {x}",
        "only": "فقط {x}",
        "per": "لكل {x}",
        "top": "أعلى {n}",
        "bottom": "أدنى {n}",
        "empty": "{x} فارغ",
        "set": "{x} موجود",
        "calc": "قيمة محسوبة",
        "sep": "، ",
    },
    "ur": {
        "count": "{x} گنے",
        "count_distinct": "مختلف {x} گنے",
        "sum": "{x} کا مجموعہ",
        "avg": "{x} کی اوسط",
        "max": "سب سے زیادہ {x}",
        "min": "سب سے کم {x}",
        "list": "{x} کی فہرست",
        "with": "{x} کے ساتھ",
        "only": "صرف {x}",
        "per": "ہر {x} کے لیے",
        "top": "سب سے زیادہ {n}",
        "bottom": "سب سے کم {n}",
        "empty": "{x} خالی",
        "set": "{x} موجود",
        "calc": "حساب شدہ قدر",
        "sep": "، ",
    },
}
#: Names for time buckets, which queries alias in English ("month") whatever the question's
#: language, and "<bucket> of <column>" for strftime().
_TIME_WORDS: dict[str, Any] = {
    "year": {
        "en": "year",
        "ar": "السنة",
        "ur": "سال",
        "of": {"en": "year of {x}", "ar": "سنة {x}", "ur": "{x} کا سال"},
    },
    "month": {
        "en": "month",
        "ar": "الشهر",
        "ur": "مہینہ",
        "of": {"en": "month of {x}", "ar": "شهر {x}", "ur": "{x} کا مہینہ"},
    },
    "day": {
        "en": "day",
        "ar": "اليوم",
        "ur": "دن",
        "of": {"en": "day of {x}", "ar": "يوم {x}", "ur": "{x} کا دن"},
    },
}
_OPERATORS = {exp.EQ: "=", exp.NEQ: "≠", exp.GT: ">", exp.LT: "<", exp.GTE: "≥", exp.LTE: "≤"}
_AGGREGATES = {exp.Count: "count", exp.Sum: "sum", exp.Avg: "avg", exp.Max: "max", exp.Min: "min"}
_QUALIFIER_RE = re.compile(r"\b\w+\.(\w+)")


def explain(sql: str, catalog: Catalog, script: Script) -> str | None:
    """The query as short plain steps in the question's language, or ``None``.

    "counted orders · with customers · only late orders · city = 'Dubai' · per courier name
    · top 1". Built from the validated syntax tree and the glossary, never by the model: an
    explanation that could be wrong in its own way would defeat its purpose, which is to let
    someone who doesn't read SQL check what was actually computed (D41).
    """
    try:
        tree = parse_sql(sql)
    except MizanError:
        return None
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None or not isinstance(select, exp.Select):
        return None
    lang = _language(script)
    words = _EXPLAIN[lang]
    aliases = {
        (t.alias or t.name).lower(): t.name.lower()
        for t in select.find_all(exp.Table)
        if catalog.has_table(t.name)
    }
    # sqlglot 30 renamed the FROM argument; accept both spellings.
    from_ = select.args.get("from_") or select.args.get("from")
    main = from_.this if from_ is not None else None
    main_table = main.name.lower() if isinstance(main, exp.Table) else None

    parts = [_action(select, main_table, catalog, lang, words, aliases)]
    joined = [
        _table_label(catalog, j.this.name, lang)
        for j in select.args.get("joins") or []
        if isinstance(j.this, exp.Table) and catalog.has_table(j.this.name)
    ]
    if joined:
        parts.append(words["with"].format(x=words["sep"].join(joined)))
    if (where := select.args.get("where")) is not None:
        parts += _conditions(where.this, catalog, lang, words, aliases)
    group = select.args.get("group")
    if group is not None:
        labels = [_expr_label(e, catalog, lang, words, aliases) for e in group.expressions]
        parts.append(words["per"].format(x=words["sep"].join(labels)))
    limit = select.args.get("limit")
    order = select.args.get("order")
    if limit is not None and order is not None and order.expressions:
        try:
            n = int(limit.expression.name)
        except (AttributeError, TypeError, ValueError):
            n = 0
        if 0 < n < 200:
            descending = bool(order.expressions[0].args.get("desc"))
            parts.append(words["top" if descending else "bottom"].format(n=n))
    return " · ".join(p for p in parts if p)


def _action(
    select: exp.Select,
    main_table: str | None,
    catalog: Catalog,
    lang: str,
    words: dict[str, str],
    aliases: dict[str, str],
) -> str:
    thing = _table_label(catalog, main_table, lang) if main_table else ""
    for projection in select.expressions:
        for kind, name in _AGGREGATES.items():
            node = projection.find(kind)
            if node is None:
                continue
            inner = node.this
            if name == "count":
                if isinstance(inner, exp.Distinct):
                    target = inner.expressions[0]
                    if (table := _key_table(target, catalog, aliases)) is not None:
                        # COUNT(DISTINCT o.order_id) counts orders.
                        return words["count"].format(x=_table_label(catalog, table, lang))
                    label = _expr_label(target, catalog, lang, words, aliases)
                    return words["count_distinct"].format(x=label)
                return words["count"].format(x=thing)
            return words[name].format(x=_expr_label(inner, catalog, lang, words, aliases))
    return words["list"].format(x=thing)


def _conditions(
    node: exp.Expression,
    catalog: Catalog,
    lang: str,
    words: dict[str, str],
    aliases: dict[str, str],
) -> list[str]:
    """Glossary definitions recognised by name, then the remaining simple comparisons."""
    text = _unqualified(node.sql(dialect="sqlite"))
    parts: list[str] = []
    covered: list[str] = []
    # Longest first, so "late order" wins over the "delivered order" it contains.
    for definition in sorted(catalog.definitions, key=lambda d: -len(d.sql)):
        body = _unqualified(definition.sql)
        if any(body in done for done in covered):
            continue
        if body and body in text and not definition.sql.upper().startswith(("SUM(", "AVG(")):
            term = {"ar": definition.aliases_ar, "ur": definition.aliases_ur}.get(lang)
            label = term[0] if term else _plural(definition.term)
            parts.append(words["only"].format(x=label))
            covered.append(body)
    for condition in _flatten_and(node):
        if any(_unqualified(condition.sql(dialect="sqlite")) in body for body in covered):
            continue
        phrase = _comparison(condition, catalog, lang, words, aliases)
        if phrase:
            parts.append(phrase)
        if len(parts) >= 5:
            break
    return parts


def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    if isinstance(node, exp.Paren):
        return _flatten_and(node.this)
    return [node]


def _comparison(
    node: exp.Expression,
    catalog: Catalog,
    lang: str,
    words: dict[str, str],
    aliases: dict[str, str],
) -> str | None:
    negated = isinstance(node, exp.Not)
    target = node.this if negated else node
    if isinstance(target, exp.Is) and isinstance(target.expression, exp.Null):
        label = _expr_label(target.this, catalog, lang, words, aliases)
        return words["set" if negated else "empty"].format(x=label)
    if negated:
        return None
    for kind, symbol in _OPERATORS.items():
        if isinstance(target, kind):
            left = _expr_label(target.this, catalog, lang, words, aliases)
            right = _expr_label(target.expression, catalog, lang, words, aliases)
            return f"{left} {symbol} {right}"
    if isinstance(target, exp.In):
        values = ", ".join(e.sql(dialect="sqlite") for e in target.expressions[:4])
        return f"{_expr_label(target.this, catalog, lang, words, aliases)} ∈ ({values})"
    return None


def _expr_label(
    node: exp.Expression | None,
    catalog: Catalog,
    lang: str,
    words: dict[str, str],
    aliases: dict[str, str],
) -> str:
    if node is None:
        return ""
    if isinstance(node, exp.Literal):
        return node.sql(dialect="sqlite")
    if isinstance(node, exp.Column):
        if (key_table := _key_table(node, catalog, aliases)) is not None:
            # Grouping by a key means "per courier", not "per courier_id".
            return _singular_label(catalog, key_table, lang)
        table = _owner(node, catalog, aliases)
        if table is None and node.name.lower() in _TIME_WORDS:
            return str(_TIME_WORDS[node.name.lower()][lang])
        return _column_label(catalog, table, node.name, lang)
    if isinstance(node, exp.TimeToStr) or (
        isinstance(node, exp.Anonymous) and node.name.lower() == "strftime"
    ):
        fmt = node.args.get("format") or (node.expressions[0] if node.expressions else None)
        column = node.find(exp.Column)
        unit = {"%Y": "year", "%Y-%m": "month", "%m": "month", "%Y-%m-%d": "day"}.get(
            fmt.name if isinstance(fmt, exp.Literal) else ""
        )
        if unit and column is not None:
            of = _expr_label(column, catalog, lang, words, aliases)
            return str(_TIME_WORDS[unit]["of"][lang]).format(x=of)
    # A business definition used as an expression, e.g. revenue = SUM(quantity * price).
    text = _unqualified(node.sql(dialect="sqlite"))
    for definition in catalog.definitions:
        body = _unqualified(definition.sql)
        if body.startswith(("SUM(", "AVG(")) and text in body:
            term = {"ar": definition.aliases_ar, "ur": definition.aliases_ur}.get(lang)
            return term[0] if term else definition.term.split(" / ")[0]
    if isinstance(node, exp.Func) and (column := node.find(exp.Column)) is not None:
        return _expr_label(column, catalog, lang, words, aliases)
    return words["calc"]


def _owner(node: exp.Column, catalog: Catalog, aliases: dict[str, str]) -> str | None:
    """The real table a column comes from, when that can be told."""
    table = aliases.get((node.table or "").lower())
    if table is None:
        owners = {t for t in aliases.values() if catalog.has_column(t, node.name)}
        table = owners.pop() if len(owners) == 1 else None
    return table


def _key_table(node: exp.Expression, catalog: Catalog, aliases: dict[str, str]) -> str | None:
    """The table whose primary key ``node`` is, or ``None``."""
    if not isinstance(node, exp.Column):
        return None
    table = _owner(node, catalog, aliases)
    if table is None:
        return None
    column = catalog.table(table).column(node.name)
    return table if column is not None and column.primary_key else None


def _plural(term: str) -> str:
    """ "late order" -> "late orders" (glossary terms are singular)."""
    return term if term.endswith("s") else term + "s"


def _singular_label(catalog: Catalog, table: str, lang: str) -> str:
    """ "courier" for couriers in English; in Urdu the shortest glossary alias, which is the
    singular ("کورئیر", not "کورئیرز"); in Arabic the first alias."""
    if lang == "ur" and (names := catalog.table(table).aliases_ur):
        return min(names, key=len)
    if lang != "en":
        return _table_label(catalog, table, lang)
    name = catalog.table(table).name
    return (name[:-1] if name.endswith("s") else name).replace("_", " ")


def _table_label(catalog: Catalog, table: str | None, lang: str) -> str:
    if not table or not catalog.has_table(table):
        return table or ""
    entry = catalog.table(table)
    names = entry.aliases(lang)
    return names[0] if names else entry.name.replace("_", " ")


def _column_label(catalog: Catalog, table: str | None, column: str, lang: str) -> str:
    entry = catalog.table(table).column(column) if table and catalog.has_table(table) else None
    if entry is not None and (names := entry.aliases(lang)):
        return names[0]
    label = column.lower()
    for suffix, replacement in (("_en", ""), ("_aed", " (AED)"), ("_m3", " (m³)")):
        if label.endswith(suffix):
            label = label[: -len(suffix)] + replacement
    label = label.replace("is_", "").replace("_", " ").strip()
    if label == "name" and table:
        singular = table[:-1] if table.endswith("s") else table
        label = f"{singular.replace('_', ' ')} name"
    return label


def _sorted_operands(node: exp.Expression) -> exp.Expression:
    """``price * quantity`` and ``quantity * price`` are the same revenue: order the two
    sides of ``*`` and ``+`` so that either spelling matches the glossary's definition."""
    if isinstance(node, (exp.Mul, exp.Add)):
        left, right = node.this, node.expression
        key = _QUALIFIER_RE.sub(r"\1", left.sql()), _QUALIFIER_RE.sub(r"\1", right.sql())
        if key[0] > key[1]:
            return type(node)(this=right.copy(), expression=left.copy())
    return node


def _unqualified(sql: str) -> str:
    """SQL text without table qualifiers or case, for matching a definition's body.

    Rendered through sqlglot first, so both sides are spelled the same way: the parser
    writes ``x IS NOT NULL`` as ``NOT x IS NULL``, and a definition typed the first way would
    otherwise never match a query printed the second.
    """
    with contextlib.suppress(sqlglot.errors.ParseError):
        tree = sqlglot.parse_one(sql, dialect="sqlite").transform(_sorted_operands)
        sql = tree.sql(dialect="sqlite")
    return re.sub(r"\s+", " ", _QUALIFIER_RE.sub(r"\1", sql)).strip().upper()
