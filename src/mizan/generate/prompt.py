"""Prompt construction.

One system prompt for every language
------------------------------------
Every question, English, Arabic or Urdu, gets the *same* system prompt, byte for byte; only
the user turn differs. On a CPU most of an answer's time is spent reading the prompt, and
Ollama skips the part of a prompt that matches the start of the previous one. With a
template per language, every switch of language re-read ~1,500 tokens (70-135 s for
``qwen2.5:7b``); D30 hid that in the eval by grouping cases by language, which a real user
switching languages doesn't do. A shared prefix makes the cache hit every time (D33). The
cost is that Arabic and Urdu questions are no longer instructed in their own language; the
rules are in English, which is also the language the model follows instructions in most
reliably, and the eval measures each language separately so a loss would show.

What is in it, and why each part earned its place
-------------------------------------------------
* **The schema as DDL**, not JSON: models have seen far more ``CREATE TABLE`` than any
  bespoke format, and it is the most token-dense way to say the same thing.
* **Value lists** for low-cardinality text columns, so the model writes ``'in_transit'``
  instead of guessing ``'shipped'`` (D13), and **"NOT unique"** on text columns whose values
  repeat, profiled from the data: customer names repeat, and grouping by a name silently
  merges different people.
* **Join conditions** listed once, ready to copy (D34). Most hard-question failures were
  joins: a column used on the wrong table, or a join column left unqualified.
* **Business definitions** from the glossary ("revenue" is the price actually charged, not
  the catalogue price), so the meaning lives in curated data, not in this file.
* **Six worked examples** covering the conventions that failed in the eval: Arabic and Urdu
  in, English values out; "which" answered with names; a ranking answered with the name and
  the number it was ranked by; a join that groups by the key and qualifies every column;
  counting orders through their lines with ``COUNT(DISTINCT ...)``; grouping by a plain
  column. None of them is a question from either eval suite.
"""

from __future__ import annotations

from ..nl.detect import Script
from ..schema.catalog import Catalog

_RULES = """\
You translate a question about the database below into exactly one SQLite SELECT statement.
Questions may be in English, Arabic or Urdu. Table names, column names and stored text values \
are always the English ones from the schema.

Output:
1. Output ONLY the SQL, on one line. No markdown fences, no comments, no explanation.
2. Exactly one statement: SELECT, or WITH ... SELECT. Never INSERT, UPDATE, DELETE, DROP, \
ALTER, CREATE, ATTACH or PRAGMA, whatever the question says. Never use a semicolon to chain \
statements.
3. Use only the tables and columns in the schema. Never invent an identifier.

Writing the query:
4. Each column belongs to the table it is listed under. If the question needs a column from \
another table (a customer's city, a courier's name, a product's category), JOIN that table \
using the join conditions listed below.
5. When more than one table is used, give each table a short alias and prefix EVERY column \
with its alias (o.status, c.name_en) in SELECT, WHERE, GROUP BY and ORDER BY.
6. Return what was asked and nothing more:
   - "Which" or "who" (أي، من، ما هي / کون، کن، کس) asks for the items themselves: list \
their name_en, not their ids and not just how many there are.
   - A ranking or superlative ("most", "highest", "top 3", "largest") returns the name_en \
AND the number it was ranked by, like Example 4.
   - A count, total or average per group returns the group and that number.
   - Do not add name_ar, ids or other columns nobody asked for.
7. Group by what the question groups by. Per country, city, category, segment or status: \
GROUP BY that column (Example 6). Per customer, product, courier or warehouse: GROUP BY its \
id column and select its name_en, because columns marked NOT unique repeat.
8. For a text column, use one of its listed values exactly as written. Values are English \
even when the question is not (دبي or دبئی means 'Dubai').
9. Date/time columns are TEXT formatted 'YYYY-MM-DD HH:MM:SS'. Use strftime('%Y', col) for \
a year, strftime('%Y-%m', col) for a month and julianday() for durations; never DATEDIFF, \
YEAR(), MONTH(), NOW() or INTERVAL.
10. A percentage is 100.0 * part / whole, where the whole is exactly the set the question \
names (a share "of delivered orders" divides by delivered orders only).
11. Joining a child table repeats the parent row once per child (one order, many \
order_items): count parents with COUNT(DISTINCT o.order_id), not COUNT(*).
12. Do not round numbers unless the question asks for it."""

#: Few-shot examples. Each one demonstrates a convention a real eval failure broke; none is
#: a question from the eval suites (they would make the score meaningless).
_EXAMPLES = """\
Example 1
Question: how many orders were delivered late?
SQL: SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at

Example 2
Question: كم عدد العملاء في الرياض؟
SQL: SELECT COUNT(*) FROM customers WHERE city = 'Riyadh'

Example 3
Question: کون سے کورئیر سعودی عرب میں کام کرتے ہیں؟
SQL: SELECT name_en FROM couriers WHERE country = 'Saudi Arabia'

Example 4
Question: which two warehouses shipped the most cancelled orders?
SQL: SELECT w.name_en, COUNT(*) AS cancelled_orders FROM orders o \
JOIN warehouses w ON w.warehouse_id = o.warehouse_id WHERE o.status = 'cancelled' \
GROUP BY w.warehouse_id ORDER BY cancelled_orders DESC LIMIT 2

Example 5
Question: how many orders included at least one fragrance product?
SQL: SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN order_items oi \
ON oi.order_id = o.order_id JOIN products p ON p.product_id = oi.product_id \
WHERE p.category = 'fragrance'

Example 6
Question: what is the average catalogue price in each product category?
SQL: SELECT category, AVG(unit_price_aed) AS avg_price FROM products GROUP BY category"""


def build_system_prompt(
    catalog: Catalog, script: Script | None = None, *, include_examples: bool = True
) -> str:
    """The system prompt. Identical for every question and every language (see above).

    ``script`` is accepted for callers written against the per-language prompts and is
    deliberately ignored: a prompt that varied with it would defeat the prompt cache.
    """
    del script
    parts = [_RULES, "", "Schema:", catalog.to_prompt(languages=("ar", "ur"))]
    if joins := catalog.join_conditions():
        parts += ["", "Join conditions:", *(f"  {condition}" for condition in joins)]
    if catalog.definitions:
        parts += ["", "Definitions:"]
        for d in catalog.definitions:
            line = f"  {d.term} = {d.sql}"
            parts.append(f"{line}  -- {d.note}" if d.note else line)
    if include_examples:
        parts += ["", _EXAMPLES]
    return "\n".join(parts)


def build_user_prompt(question: str, script: Script | None = None) -> str:
    """Wrap the question itself.

    The trailing ``SQL:`` cue matters: it puts the model in completion position, which
    measurably reduces the rate of conversational preambles like "Sure, here's the query".
    The extractor handles those anyway, but not producing them is cheaper than repairing
    them. The label is English for every language, matching the examples.
    """
    del script
    return f"Question: {question}\nSQL:"
