"""Prompt construction.

Design notes that materially changed accuracy during this build
---------------------------------------------------------------
* **The schema is rendered as DDL, not JSON.** Models have seen far more ``CREATE TABLE``
  during pretraining than any bespoke schema format, and DDL is also the most token-dense
  way to express the same information.
* **Low-cardinality text columns carry their value set.** Telling the model that ``status``
  is one of ``{'cancelled','delivered','in_transit','pending','returned'}`` eliminates the
  single most common silent failure, where it emits ``status = 'shipped'`` — valid SQL,
  valid identifiers, zero rows, and no error anywhere to tell you it was wrong.
* **The date format is stated explicitly.** SQLite has no date type; these columns are
  ``TEXT`` in ``YYYY-MM-DD HH:MM:SS``. Without that sentence the model reaches for
  ``DATEDIFF`` and other dialects' date functions.
* **Arabic questions get an Arabic-language instruction block.** Instructing a model in the
  language of the question measurably improves instruction-following. The *output* contract
  is unchanged — identifiers are always the English ones from the schema, because those are
  what the database actually contains.
"""

from __future__ import annotations

from ..nl.detect import Script
from ..schema.catalog import Catalog

_RULES_EN = """\
You translate a question into exactly one SQLite SELECT statement.

Rules:
1. Output ONLY the SQL. No markdown fences, no explanation, no trailing commentary.
2. Exactly one statement. Never use a semicolon to chain statements.
3. SELECT (or WITH ... SELECT) only. Never INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH or PRAGMA.
4. Use only the tables and columns given in the schema. Never invent an identifier.
5. SQLite dialect only. Date/time columns are TEXT formatted 'YYYY-MM-DD HH:MM:SS';
   compare them as strings or with date()/strftime(), never with DATEDIFF or INTERVAL.
6. When a column lists its possible values, use one of those values exactly as written.
7. An order is late when delivered_at IS NOT NULL AND delivered_at > promised_at.
   An order that has not been delivered is not late.
8. Prefer explicit JOIN ... ON over comma joins. Alias tables when you join more than one."""

_RULES_AR = """\
مهمتك تحويل السؤال إلى جملة SELECT واحدة بلغة SQLite.

القواعد:
١. أخرج SQL فقط. بدون علامات تنسيق، بدون شرح، بدون أي نص إضافي.
٢. جملة واحدة فقط. لا تستخدم الفاصلة المنقوطة لربط جمل متعددة.
٣. SELECT فقط. ممنوع INSERT أو UPDATE أو DELETE أو DROP أو ALTER أو ATTACH أو PRAGMA.
٤. استخدم فقط الجداول والأعمدة الموجودة في المخطط. لا تخترع أي اسم.
٥. أسماء الجداول والأعمدة تبقى بالإنجليزية كما وردت في المخطط، حتى لو كان السؤال بالعربية.
٦. أعمدة التاريخ نصية بصيغة 'YYYY-MM-DD HH:MM:SS'. استخدم date() أو strftime() فقط.
٧. الطلب متأخر عندما delivered_at IS NOT NULL AND delivered_at > promised_at.
٨. عند وجود قائمة قيم محتملة لعمود، استخدم إحدى تلك القيم كما هي بالضبط."""

#: Few-shot examples. Kept deliberately small — two examples that demonstrate the two
#: hardest conventions (the late-order relationship and Arabic-in/English-identifiers-out)
#: beat a dozen that repeat the easy cases and cost context.
_EXAMPLES = """\
Example 1
Question: how many orders were delivered late?
SQL: SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at

Example 2
Question: كم عدد العملاء في الرياض؟
SQL: SELECT COUNT(*) FROM customers WHERE city = 'Riyadh'"""


def build_system_prompt(
    catalog: Catalog, script: Script, *, include_examples: bool = True
) -> str:
    """Assemble the system prompt for a question written in ``script``."""
    rules = _RULES_AR if script.prompt_language == "ar" else _RULES_EN
    schema = catalog.to_prompt(include_arabic=script.prompt_language == "ar")

    parts = [rules, "", "Schema:", schema]
    if include_examples:
        parts += ["", _EXAMPLES]
    return "\n".join(parts)


def build_user_prompt(question: str, script: Script) -> str:
    """Wrap the question itself.

    The trailing ``SQL:`` cue matters: it puts the model in completion position, which
    measurably reduces the rate of conversational preambles like "Sure, here's the query".
    The extractor handles those anyway, but not producing them is cheaper than repairing
    them.
    """
    label = "السؤال" if script.prompt_language == "ar" else "Question"
    return f"{label}: {question}\nSQL:"
