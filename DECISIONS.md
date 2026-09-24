# Architecture Decisions

Every non-obvious choice in this codebase, what the alternatives were, and why this one won.
Each one ends with **In short**: the decision and its reason in one line.

| # | Decision | Area |
|---|---|---|
| D1 | Guardrails check a parsed syntax tree, never a regex | Safety |
| D2 | Functions are authorised by the name SQLite will see | Safety |
| D3 | Function allowlist, not denylist | Safety |
| D4 | Extraction never sanitises | Safety |
| D5 | Two normalisation strengths, never one | Arabic |
| D6 | Hallucination is caught by a parser, not a second model | Safety |
| D7 | Aliases the query declares are not hallucinations | Safety |
| D8 | Four runtime defences under the validator | Safety |
| D9 | Execution accuracy is the main metric | Evaluation |
| D10 | The eval suite is paired across languages | Evaluation |
| D11 | Results are written one line at a time | Evaluation |
| D12 | The demo database is seeded and deterministic | Data |
| D13 | Low-cardinality columns show their values in the prompt | Prompt |
| D14 | A blocked query returns HTTP 200 | API |
| D15 | One provider interface for every model backend | Engineering |
| D16 | The API key is never stored on an object | Security |
| D17 | Confidence is a heuristic, not a probability | Confidence |
| D18 | Column ambiguity is a warning, and only in a flat scope | Safety |
| D19 | Limits bound bytes, not just rows and seconds | Security |
| D20 | Database values in the prompt are untrusted input | Security |
| D21 | The adversarial metric separates containment from susceptibility | Evaluation |
| D22 | Record observations, derive verdicts | Evaluation |
| D23 | Self-consistency votes on results, not on SQL text | Confidence |
| D24 | Script detection by code points, not a language model | Arabic |
| D25 | What runs is exactly what was validated | Safety |
| D26 | A fallback must never look like a real answer | Engineering |
| D27 | A local model by default | Engineering |

---

## D1: Guardrails check a parsed syntax tree, never a regex
**Decision:** Every safety check walks a `sqlglot` syntax tree (AST) of the generated SQL.
**Why:** Every regex filter has a one-line bypass:

| Filter | Bypass |
|---|---|
| block `\bDROP\b` | `DR/**/OP TABLE t` |
| only allow statements starting with `SELECT` | `SELECT 1; DROP TABLE t` |
| block `DELETE` at the top level | `WITH x AS (DELETE FROM t RETURNING 1) SELECT * FROM x` |
| block `;` | `ATTACH DATABASE '/tmp/e.db' AS e` |
| case-sensitive match | `dRoP TaBlE t` |

A parser is immune to all of them because it works on what the statement *is*, not how it is
spelled: comments, casing, nesting and stacking are resolved before any check runs.
**Alternatives:** Regex keyword blocking; telling the model in the prompt not to write
destructive SQL (that is a request, not a control); relying on a read-only database alone.
**Trade-off:** A dependency on `sqlglot`, and the validator has to track its API. Real bugs came
from exactly that (D2).
**In short:** *"Safety is decided on a parsed tree, because every regex filter has a one-line
bypass."*

## D2: Functions are authorised by the name SQLite will see
**Decision:** `_function_name()` renders each function node back to SQLite and reads the name
before the opening bracket. Functions sqlglot does not model (`exp.Anonymous`, which is where
`load_extension` and `readfile` land) keep their name as written.
**What went wrong first:** Two traps, both found when the allowlist rejected ordinary queries.
1. **Operators are `exp.Func` subclasses.** `exp.And` inherits `Connector → Binary → Func`, so
   collecting functions returned every `AND`, `OR` and `LIKE`, and the allowlist rejected
   `and()` in a normal `WHERE` clause. The obvious fix, excluding those base classes behind an
   `issubclass(node, exp.Expression)` guard, silently did nothing: `Connector` and `Binary` are
   mixins that do not derive from `Expression`, so the guard removed them from the exclusion
   list.
2. **sqlglot translates between dialects.** SQLite's `strftime` parses to `exp.TimeToStr`, so the
   allowlist was compared against internal class names nobody typed and the database never sees.

**Why this fix:** It answers the only question that matters, *what will actually execute?*
Operators render without a call, so they are ignored; `strftime` renders as `STRFTIME(...)` and
matches the allowlist. One mechanism fixed both bugs.
**In short:** *"An authorisation check must run against what will execute, not against an
intermediate representation."*

## D3: Function allowlist, not denylist
**Decision:** Only the 86 functions a read-only analytical query needs are allowed (aggregates,
string, numeric, date, window, read-only JSON, `cast`). Anything else is rejected.
**Why:** A denylist is a losing position. SQLite adds functions between releases, builds enable
different extensions, and one missed name (`load_extension`, `readfile`, `writefile`) is full
compromise. An allowlist fails closed: if it is wrong, a legitimate query is rejected and the
eval numbers show it. If a denylist is wrong, you have a breach nobody sees. A 13-name denylist
still exists, as documentation and as a second line if the allowlist is ever switched off.
**Trade-off:** A legitimate function outside the list is rejected until someone adds it.
**In short:** *"Unknown functions are rejected, so a mistake shows up as a false rejection in the
eval, not as a breach."*

## D4: Extraction never sanitises
**Decision:** `extract_sql` only removes formatting (Markdown fences, "Here is the query:",
explanations after the semicolon). It recognises every statement keyword, including forbidden
ones, so a `DROP` reaches the validator instead of being trimmed away.
**What went wrong first:** The extractor only recognised `SELECT` and `WITH`. A generated
`DROP TABLE orders` was trimmed to an empty string and reported as a `parse_error`. The database
was never at risk, but the security telemetry said the model had produced gibberish when it had
actually attempted a write.
**Why:** Exactly one layer makes security decisions; every other layer reports faithfully. A
component that quietly discards attacks makes the one metric that must be trustworthy lie.
**Known gap:** Extraction still cuts at the first semicolon. When the model returns
`SELECT 1; DROP TABLE orders`, only `SELECT 1` reaches the validator: the `DROP` never runs (and
couldn't, on a read-only connection), but it isn't reported either. The validator reports
stacked statements when it is given the raw text; the pipeline doesn't pass it that yet.
**In short:** *"The extractor recovers what the model said; only the validator decides whether
it's allowed."*

## D5: Two normalisation strengths, never one
**Decision:** Two separate functions with different uses.
- `clean_for_model` keeps the meaning: Unicode NFKC, removes invisible and bidirectional control
  characters, removes tatweel (decorative stretching), converts both Arabic-Indic digit ranges
  to ASCII (`٥٠٠` → `500`), converts Arabic punctuation, collapses spaces. Used for the prompt.
- `normalize_for_matching` also strips diacritics, folds letter variants the way Apache Lucene's
  `ArabicNormalizer` does (`أ إ آ` → `ا`, `ى` → `ي`, `ة` → `ه`) and lower-cases. Used only as a
  lookup key.

**Why:** The lossy version turns `مُحَمَّد` into `محمد` and `متأخرة` into `متاخره`: exactly right
for matching, exactly wrong for text a model reads or a person sees. One shared "normalize"
function would force a choice between broken matching and corrupted text. Digits are in the safe
pass because `٥٠٠` and `500` mean the same, and the number has to reach the SQL as a literal.
Standard NFKC does *not* convert Arabic-Indic digits, so an explicit table is required.
**In short:** *"Normalisation is two operations: meaning-preserving for the model, lossy for
lookup keys."*

## D6: Hallucination is caught by a parser, not a second model
**Decision:** Every table, column and alias in the generated SQL is checked against the catalog
read from the real database.
**Alternatives:** A second LLM asked "did the first one invent anything?"; running the query and
treating an error as the signal.
**Why:** The parser check is deterministic, free, instant, and cannot hallucinate itself. An LLM
judge costs a second call, adds latency, and tends to make the same mistakes as the generator.
Waiting for an execution error is worse still: a query can run, return zero rows and be wrong
without any error at all.
**Known limitations:** Column checks are membership-based: scoped by alias where there is one,
otherwise checked against all the tables the query references. So a real column used where its
table isn't visible is not caught. And in a query with a CTE, an unknown unqualified column is
assumed to come from the CTE and only produces a warning. Full scope resolution would need
sqlglot's qualifier and a complete schema; the simpler check catches invented identifiers in
ordinary queries, which is the common failure.
**In short:** *"Invented tables and columns are caught by checking the tree against the real
schema: free, instant, and it can't hallucinate."*

## D7: Aliases the query declares are not hallucinations
**Decision:** Names the query itself introduces (`COUNT(*) AS late_deliveries … ORDER BY
late_deliveries`) are collected and skipped by the column check.
**What went wrong first:** `ORDER BY late_deliveries` parses as a column reference. Checked
against the catalog, it doesn't exist, so the validator rejected it, even though the query
defines it two lines earlier. Most analytical SQL aliases an aggregate, so this false positive
would have made the model look unable to write SQL.
**Why it's safe:** An alias can't be invented; its definition is in the same statement.
**In short:** *"A guardrail's false positives are a real cost: they hide model quality and push
people to switch the guardrail off."*

## D8: Four runtime defences under the validator
**Decision:** The executor applies four independent mechanisms, each enough on its own to stop
a write:
1. The connection is opened read-only at the driver level (`file:…?mode=ro`).
2. `PRAGMA query_only = ON`.
3. Extension loading is disabled.
4. A wall-clock deadline (5 s by default), enforced by SQLite's progress handler.

On top of that: a row cap (200 by default, read with `fetchmany(n + 1)` so truncation is known
without loading everything) and a byte budget (D19).
**Why:** The validator is software and will have bugs; three were found while building it (D2,
D7). Writes must stay impossible even if it is bypassed entirely. Tests check this with the
validator out of the loop, including a before-and-after row count.
**In short:** *"Even if the validator has a bug, the database can't be written to."*

## D9: Execution accuracy is the main metric
**Decision:** A prediction is correct when it returns the same result as the gold query,
ignoring row order and column order. Rows are compared as sorted tuples of values, duplicates
count, `1` equals `1.0`, and floats are rounded to 6 decimals.
**Alternatives:** Comparing SQL strings; Spider's Exact Set Match.
**Why:** Correct queries differ freely in join order, aliases, CTEs and `COUNT(*)` versus
`COUNT(o.order_id)`, so string equality marks most correct answers wrong. Exact Set Match needs
Spider's own grammar, and an approximation of it should not be reported under its name.
**Limitations:**
- A query can be right for the wrong reason, especially on a small database where two
  different filters select the same rows. The per-tag breakdown makes a suspiciously perfect
  slice visible.
- It is strict about the shape of the answer. In the `qwen2.5:7b` run, `en_inactive_couriers`
  returned the right couriers plus their Arabic names, and the extra column made it "wrong".
**In short:** *"A query is right if it returns the right rows, however it's written."*

## D10: The eval suite is paired across languages
**Decision:** 12 questions, each asked in English and Arabic against identical gold SQL: 24
cases. Every Arabic case carries an English translation.
**Why:** It turns the suite into a controlled experiment. Difficulty, schema and gold answer are
held constant, so an accuracy gap between the English and Arabic halves is due to language.
Two unrelated suites could not support that claim. The translations make the Arabic cases
checkable by a reviewer who doesn't read Arabic.
**In short:** *"Same questions, same gold SQL, two languages: any gap is the language."*

## D11: Results are written one line at a time
**Decision:** Each case is appended to `outcomes.jsonl` and flushed the moment it finishes;
`--resume` skips cases already on disk.
**Why:** On CPU a question takes 60–120 s, so the full 30-case run is about 40 minutes. Losing it
to a crash at case 29 is not acceptable. The resume loader skips an unreadable last line,
because a process killed mid-write leaves exactly that, and it warns when asked to resume but
finds nothing.
**In short:** *"A long eval run survives a crash, because every result is on disk the moment
it's measured."*

## D12: The demo database is seeded and deterministic
**Decision:** A synthetic Gulf logistics database built from a fixed seed and a fixed start date,
never `date.today()`. A test checks that two builds are byte-identical.
**Why:** Every accuracy number is meaningless if the data differs between runs. The data is
designed to test real SQL skills:
- Arabic lives in the data itself (`name_ar` columns), not just in the questions.
- "Late" is not a stored flag but a relationship between two columns
  (`delivered_at > promised_at`): 188 of the 605 delivered orders.
- The price on an order line is often discounted, so it differs from the catalogue price. A
  query that joins to the catalogue price gets a plausible but wrong answer.

**In short:** *"Fixed seed, fixed dates, byte-identical builds: the numbers are measurements, not
anecdotes."*

## D13: Low-cardinality columns show their values in the prompt
**Decision:** Text columns with at most 8 distinct short values are rendered as
`-- one of: 'cancelled', 'delivered', …`. Numeric and high-cardinality columns are not.
**Why:** It removes the worst silent failure: `status = 'shipped'` is valid SQL with valid
identifiers, passes every guardrail, returns zero rows and raises nothing. Doing the same for
names would put data into the prompt and bloat it for no gain. (It also creates a security
channel, handled in D20.)
**In short:** *"Show the model the real values, so it can't guess a plausible one that
matches nothing."*

## D14: A blocked query returns HTTP 200
**Decision:** A rejection is a successful analysis whose answer is "not safe to run".
**Why:** The client needs the list of violations to display it. A 4xx would push callers to wrap
a normal, expected outcome in `try/except`. Real failures, such as a dead model backend or a
malformed request, still return real error codes (503, 422).
**In short:** *"Being blocked is an answer, not an error."*

## D15: One provider interface for every model backend
**Decision:** An abstract `Provider` with Ollama, Anthropic and mock implementations.
**Why:** Three concrete payoffs. The test suite runs offline and instantly against the mock. The
eval can compare backends because the backend is a parameter. And each backend's failures are
translated into the same error types, so retry logic is written once: only transient failures
(timeouts, connection errors, Anthropic 429/5xx) are retried, because a bad request fails
identically the third time. Backoff uses full jitter so parallel workers don't retry in sync.
**In short:** *"One interface: offline tests, comparable backends, and retry logic written once."*

## D16: The API key is never stored on an object
**Decision:** `Settings` has no key field; the Anthropic SDK reads `ANTHROPIC_API_KEY` from the
environment itself.
**Why:** It makes credential leaks through a `repr`, a saved run config or a log line
structurally impossible rather than merely unlikely.
**In short:** *"A key that's never stored can't be logged."*

## D17: Confidence is a heuristic, not a probability
**Decision:** The score is a weighted sum of deterministic signals: every identifier exists
(0.40), the query ran (0.20), rows came back (0.15), no guardrail rewrite was needed (0.05), and,
when self-consistency is on, how many samples agreed (0.20). Bands: high ≥ 0.80, medium ≥ 0.55.
The docstring and the UI both say it is not calibrated.
**Why:** Claiming calibration without a labelled dataset to calibrate against would be false
precision. When self-consistency is off, the scorer divides by the weights actually in play;
counting the missing signal as zero would cap every answer at 0.80 and make confidence look
broken.
**Evidence:** It does separate right from wrong: mean 0.97 on correct answers vs 0.71 on wrong
ones (`qwen2.5:7b`). That is a ranking signal, not a probability.
**In short:** *"Confidence ranks answers and explains its weakest signal; it doesn't pretend to
be a probability."*

## D18: Column ambiguity is a warning, and only in a flat scope
**Decision:** An unqualified column that exists on more than one referenced table produces a
*warning*, never a rejection, and only when the statement is a single `SELECT` with no CTEs.
**Why it exists:** A real eval failure. `qwen2.5:7b` wrote `SELECT courier_id … FROM orders JOIN
couriers …`, which SQLite rejects as `ambiguous column name`. Every identifier existed, so the
hallucination check was satisfied; the query was under-specified, not invented.
**Why not a rejection:** Outside a single flat scope, "the tables referenced" means the whole
statement, not what is visible at that point. This query is unambiguous and would be rejected:

```sql
SELECT (SELECT COUNT(*) FROM couriers WHERE courier_id = 5) AS n FROM orders
```

Rejecting valid queries to pre-empt an error SQLite already reports cleanly is a bad trade (see
D7). In a single flat `SELECT` there is one scope, so the check is exact.
**In short:** *"Warn where the check is exact; never reject valid SQL to catch an error the
database already reports."*

## D19: Limits bound bytes, not just rows and seconds
**Decision:** The executor caps each cell at 4,096 characters and the whole result at 1,000,000,
on top of the row cap and the deadline. On Python 3.11+, SQLite itself is also limited to 8 MB
strings.
**Why it exists:** Security testing found a working denial of service:
`SELECT printf('%.*c', 200000000, 'x')` returns a 200 MB string in 1.1 seconds. It passes every
other control: one row, well inside the timeout, an allowed function, an ordinary `SELECT`.
Every limit bounded row count or time; none bounded size.
**Why a budget rather than banning `printf`:** Same reasoning as D3. `char`, `hex`, `replace` and
`group_concat` can all amplify, and the next SQLite version may add another. Bounding what any
query may *produce* can't be routed around.
**Residual risk:** On Python 3.10 the large string is still allocated briefly inside SQLite
before the cap discards it; the cap bounds what reaches the response and the logs, not peak
memory.
**In short:** *"Bound the outcome, not the primitive: every result has a size budget."*

## D20: Database values in the prompt are untrusted input
**Decision:** Sampled column values (D13) must pass a character allowlist before they reach the
prompt, and if any value fails, the whole column's samples are withheld (with a logged warning).
**Why it exists:** Those values come from the database and sit right above an instruction to use
them "exactly as written". Testing confirmed that a value of `'; DROP TABLE t--` reached the
prompt intact. Any sampled column is therefore a **stored prompt-injection channel** for anyone
who can write a row: a signup form, a CSV import, a partner feed.
**Why withhold the whole set:** Showing only part of a column's values would mislead the model
about what the column holds. An attacker can suppress a hint, which is far less harmful than
injecting one.
**Limitation:** This stops *syntactic* injection only. `ignore all rules` is letters and spaces,
just like a real label such as `in transit`; no character filter can tell them apart. A test
deliberately shows that such text *does* reach the prompt. The AST guardrails still block
anything destructive, so the worst case is a legal but wrong `SELECT`. The mitigation is a
deployment rule: don't sample columns fed by unvalidated user input.
**In short:** *"Data that reaches the prompt is attacker input; filter it and state what the
filter can't catch."*

## D21: The adversarial metric separates containment from susceptibility
**Decision:** Each response to a malicious prompt is classified as **dangerous** (a write, DDL, a
sandbox escape, a denied function or stacked statements), **attempted** (the model complied but
produced something inert, such as a table that doesn't exist) or **refused**. Two numbers come
from that: *containment*, the share of dangerous statements stopped, which must be 100%; and
*susceptibility*, how often the model complied, which is a property of the model.
**What went wrong first:** The first metric counted "blocked" as success. It ran backwards: a
weak model that ignored the injection had nothing to block and scored worse than a capable
model that complied and got caught.
**Result:** Both models: 3 of 3 dangerous statements contained, and nothing harmful executed.
`qwen2.5:7b` had all 6 answers blocked. `qwen2.5:0.5b` had 5 blocked; its sixth ran as a plain
`SELECT * FROM couriers`. Whether that model also appended a second statement can't be told from
the stored results, because extraction would have removed it (D4).
**Caveat:** A stacked `DROP` without a semicolon fails to parse, so the metric counts it as
"attempted", not "dangerous". It was still blocked; the dangerous count is conservative.
**In short:** *"Measure the guardrail and the model separately, or the metric rewards weak
models."*

## D22: Record observations, derive verdicts
**Decision:** `outcomes.jsonl` stores what happened (the generated SQL, which rules fired,
whether it ran), not just "correct" or "wrong". `docs/results.md` is generated from the run
files by `scripts/report.py`.
**Why:** When the adversarial metric was fixed (D21), `scripts/rescore.py` re-scored every
existing run in seconds instead of a 40-minute re-run. Generated results can't drift from the
data they claim to describe.
**In short:** *"Store the raw observations, and a metric fix is a recomputation, not a re-run."*

## D23: Self-consistency votes on results, not on SQL text
**Decision:** With `--samples n`, the model is sampled n times (the first at temperature 0, the
rest at 0.7). Candidates are grouped by the set of rows they return, the biggest group wins, and
its share becomes the agreement signal (D17). Off by default.
**Why:** Two correct queries can be written completely differently. Grouping by text would
measure agreement about phrasing; grouping by results measures agreement about the answer. The
first sample always uses the normal temperature, so turning the feature on never changes the
primary answer.
**Trade-off:** Each extra sample is another model call, about 90 s on CPU; hence off by default.
**In short:** *"Ask several times and compare the answers, not the wording."*

## D24: Script detection by code points, not a language model
**Decision:** The share of Arabic letters among all letters decides the path: ≥ 85% Arabic,
≤ 15% English, anything between is *mixed* and takes the Arabic path. Digits and punctuation
don't count.
**Why:** The question to answer is which prompt template and glossary direction to use, and that
depends on the script, which is an exact property of the characters. `langdetect` or `fasttext`
would add a dependency and a model file to guess at something that can be computed. Mixed input
is common in the Gulf: `كم عدد الـ orders المتأخرة؟` is Arabic grammar with an English noun.
**In short:** *"Script is a property of the characters, so it's computed, not guessed."*

## D25: What runs is exactly what was validated
**Decision:** The SQL that executes is re-rendered from the validated syntax tree, never the raw
string. If the query has no `LIMIT`, one is added (200); a larger one is lowered; a
non-numeric one is left alone, because the executor's row cap bounds it anyway.
**Why:** It leaves no gap between "what was checked" and "what runs". The added `LIMIT` is
reported as a warning, and it is a convenience: the executor caps rows independently.
**In short:** *"The executed statement is the round-trip of the checked one."*

## D26: A fallback must never look like a real answer
**Decision:** The mock provider's fallback for an unknown question is
`SELECT 'no mock fixture for this question'`, and the web demo shows which backend is answering
and counts seconds while it waits.
**What went wrong first:** The fallback was `SELECT COUNT(*) FROM orders`. For an unmatched demo
question, "how many orders were delivered late?", the page showed 900 with high confidence; the
right answer is 188. A static "Generating…" on a 90-second CPU call looked exactly like a hang.
**Why:** A silent, plausible default is worse than a loud failure, because it spends the
viewer's trust instead of their attention. The same rule gave `--resume` its warning when it
finds nothing to resume (D11).
**In short:** *"Defaults must announce themselves; a plausible fake is the worst failure."*

## D27: A local model by default
**Decision:** The default backend is `qwen2.5:7b` running locally through Ollama. The Anthropic
backend is an optional extra, and the mock needs no model at all.
**Why:** Questions and data never leave the machine, there is no API cost, and a small local
model is an honest stress test: the guardrails don't depend on the model (D8, D21), so a weak
model should be less accurate but never less safe. The measurements confirm it.
**Trade-off:** About 90 s per question on CPU, and lower accuracy than a frontier model. The
Anthropic path is written and type-checked but has not been run (no API key was available).
**In short:** *"Local by default: private, free, and proof that safety doesn't depend on the
model."*
