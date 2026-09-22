# Decision log

Every non-obvious choice in this codebase, what the alternatives were, and why this one
won. Written so that an interviewer asking "why did you do it that way?" gets a real answer
rather than "that's how the tutorial did it".

---

## D1 — Guardrails validate a parsed AST, never a regex

**Decision.** All SQL safety checks walk a `sqlglot` syntax tree.

**Alternatives.** Regex keyword blocking; a prompt instruction telling the model not to
write destructive SQL; running against a read-only replica and accepting the risk.

**Why.** Every regex filter has a one-line bypass, and they compose badly:

| Filter | Bypass |
|---|---|
| block `\bDROP\b` | `DR/**/OP TABLE t` |
| prefix-match `SELECT` | `SELECT 1; DROP TABLE t` |
| block `DELETE` | `WITH x AS (DELETE FROM t RETURNING 1) SELECT * FROM x` |
| block `;` | `ATTACH DATABASE '/tmp/e.db' AS e` |
| case-sensitive match | `dRoP TaBlE t` |

A parser is immune to all of them because it reasons about what a statement *is*, not how
it is spelled. Prompt instructions are not a control at all — they are a request.

**Cost.** A dependency on `sqlglot`, and the validator must track its API. Two real bugs
came from exactly that (see D2, D3).

---

## D2 — Function authorisation uses the dialect-rendered name, not sqlglot's class name

**Decision.** `_function_name()` renders each `exp.Func` node back to SQLite and reads the
identifier before the opening paren.

**What went wrong first.** Two separate traps, both found by the allowlist rejecting
ordinary queries:

1. **Operators are `exp.Func` subclasses.** `exp.And` has the MRO
   `And → Connector → Binary → Func`, so `find_all(exp.Func)` yields every `AND`, `OR`,
   `LIKE` and comparison in the query. The first implementation reported a function called
   `and` and rejected almost every non-trivial `WHERE` clause. Worse, the obvious fix —
   excluding those base classes — *silently did nothing*, because `Connector` and `Binary`
   are mixins that do not derive from `exp.Expression`, so a
   `issubclass(node, exp.Expression)` guard filtered them out of the exclusion list.
2. **sqlglot canonicalises across dialects.** SQLite's `strftime` parses to
   `exp.TimeToStr`, with its argument wrapped in `exp.TsOrDsToTimestamp`. Checking an
   allowlist against sqlglot's internal class names compares against a vocabulary the user
   never typed and the database never sees.

**Why the render approach.** It answers the only question that matters: *what function will
actually execute?* Operators render without a call form and are correctly ignored;
`strftime` renders as `STRFTIME(...)` and matches the allowlist entry.

---

## D3 — Function policy is an allowlist, not a denylist

**Decision.** Unknown functions are rejected.

**Why.** A denylist is a losing position. SQLite adds functions between point releases,
different builds enable different extensions, and one missed name (`load_extension`,
`readfile`, `writefile`, `edit`) is full compromise. An allowlist fails closed: being wrong
costs a false rejection that shows up in the eval numbers, not a breach that does not.

**Cost.** Legitimate functions outside the list are rejected until added. Accepted, because
that failure is *visible*; the denylist's failure is not.

---

## D4 — Extraction must never sanitise

**Decision.** `extract_sql` recognises every statement keyword, including the forbidden
ones, and hands whatever it finds to the validator.

**What went wrong first.** The original version matched only `SELECT|WITH`. A generated
`DROP TABLE orders` was trimmed to an empty string and reported as `parse_error`. The
database was never at risk — but the guardrail telemetry claimed the model had emitted
gibberish when it had actually attempted a write. Security decisions belong in exactly one
place, and a component that quietly discards attacks makes the one metric that must be
trustworthy lie.

---

## D5 — Two normalization strengths, never one

**Decision.** `clean_for_model` (meaning-preserving) and `normalize_for_matching` (lossy)
are separate functions with different call sites.

**Why.** `normalize_for_matching` folds `مُحَمَّد` to `محمد` and `متأخرة` to `متاخره`. That is
exactly right for a lookup key and exactly wrong for anything a human will read or a model
will reason over. One shared "normalize" function forces a choice between broken matching
and corrupted display. Digit folding (`٥٠٠` → `500`) sits in the *safe* pass because it
preserves meaning and materially helps the model emit correct literals.

---

## D6 — Hallucination is detected by a parser, not a second model

**Decision.** Every table, column and alias in generated SQL is checked against the
introspected catalog.

**Alternatives.** An LLM-as-judge pass asking "did the first model invent anything?";
executing and treating an error as the signal.

**Why.** The parser check is deterministic, free, instant, and cannot itself hallucinate.
An LLM judge costs a second call, adds latency, and is wrong in correlated ways with the
generator. Execution-as-signal is strictly worse: `SELECT status FROM orders` where `status`
exists but `city` does not would need the query to actually run first, and a query that
*succeeds* while being wrong (`status = 'shipped'` returning zero rows) produces no error at
all.

**Known limitation, stated honestly.** Column checking is membership-based, scoped by alias
where one is present and otherwise checked against the union of referenced tables. It does
not catch a column that exists on table A being referenced in a scope where only B is
visible. Full scope resolution would need sqlglot's qualifier and a complete schema; the
trade was deliberate, because this catches every *invented* identifier, which is the failure
mode that matters.

---

## D7 — Declared aliases are exempt from the unknown-column check

**Decision.** Names introduced by the query itself (`SELECT COUNT(*) AS n ... ORDER BY n`)
are collected and skipped.

**What went wrong first.** `ORDER BY n` parses as an `exp.Column`, and checking it against
the catalog reported `n` as a hallucinated column — true, and completely wrong, because the
query defines it two lines earlier. This was a false-positive class that would have
destroyed accuracy on any query using an aggregate alias. Exempting declared aliases is
safe: an alias cannot be invented, because its definition is in the same statement.

---

## D8 — Defence in depth at execution time

**Decision.** Four independent runtime mechanisms, each sufficient alone:

1. Connection opened read-only at the driver level (`file:...?mode=ro`).
2. `PRAGMA query_only = ON`.
3. Extension loading explicitly disabled.
4. A wall-clock deadline via `set_progress_handler`.

**Why.** The validator is software and will have bugs — three were found during this build.
Writes must remain impossible even if it is bypassed entirely. The executor tests assert
this *without* the validator in the loop.

---

## D9 — Execution accuracy as the primary metric

**Decision.** A prediction is correct when it returns the same result set as the gold query,
ignoring row and column order.

**Alternatives.** SQL string equality; Spider's Exact Set Match.

**Why.** Two correct answers to "which courier was late most often" differ in join order,
alias names, CTE usage and `COUNT(*)` vs `COUNT(o.order_id)`. String equality scores almost
every correct query wrong. Spider's ESM is better but still penalises a different route to
the same answer, and cannot be computed without Spider's own grammar — implementing an
approximation and calling it ESM would be dishonest.

**Limitation, stated.** A query can be right for the wrong reason, especially on a small
database where two different filters select the same rows. That is why the suite reports a
per-tag breakdown: a suspiciously perfect score on `date_logic` is visible rather than
hidden in an aggregate.

---

## D10 — The eval suite is paired across languages

**Decision.** Every question exists in English and Arabic with identical gold SQL.

**Why.** It converts the suite into a controlled experiment. An accuracy gap between `en_*`
and `ar_*` cases is attributable to language, because difficulty, schema and gold answer are
held constant. Two separate unpaired suites could not support that claim.

---

## D11 — Results are written incrementally, one JSON line per case

**Decision.** Each case is appended and flushed the moment it completes; `--resume` skips
cases already on disk.

**Why.** CPU inference takes 60–120 s per question, so a 30-case run is 40 minutes. Losing
that to a crash at case 29 is unacceptable. The resume loader also tolerates an unparseable
final line, because a process killed mid-write leaves exactly that.

---

## D12 — The database is seeded and fully deterministic

**Decision.** Fixed seed, fixed epoch, no `date.today()` anywhere.

**Why.** Every accuracy number is meaningless if the data differs between runs. A test
asserts that two builds are byte-identical.

**Design detail.** `order_items.unit_price_aed` is deliberately a discounted price that
differs from `products.unit_price_aed`. A query that naively joins to the catalogue price
gets a different answer than one reading the line price — which is exactly the distinction
between a correct query and a plausible one.

---

## D13 — Low-cardinality columns carry their value set into the prompt

**Decision.** Text columns with few distinct short values render as
`-- one of: 'delivered', 'in_transit', ...`; high-cardinality and numeric columns do not.

**Why.** It removes the worst silent failure mode: `status = 'shipped'` is valid SQL with
valid identifiers, passes every guardrail, returns zero rows, and raises nothing anywhere.
Doing the same for names would leak data into the prompt and bloat it for no gain.

---

## D14 — Guardrail rejections return HTTP 200

**Decision.** A blocked query is a successful analysis whose answer is "not safe to run".

**Why.** The client needs the violation list to render it. Returning 4xx pushes callers to
wrap a normal, expected outcome in `try/except`. Genuine failures — dead backend, malformed
request — still return real error codes.

---

## D15 — Provider adapter layer rather than one HTTP call

**Decision.** An abstract `Provider` with Ollama, Anthropic and Mock implementations.

**Why.** Three concrete payoffs: the test suite runs offline and instantly against the mock;
the eval harness can compare backends because the backend is a parameter; and per-backend
failure modes are normalised so retry logic is written once. Only transient failures
(timeout, connection) are retried — a 400 fails identically on the third attempt and
retrying it just distorts latency measurements. Backoff uses full jitter so parallel eval
workers do not re-synchronise into a thundering herd.

---

## D16 — The API key is never stored on an object

**Decision.** `Settings` has no key field; the Anthropic SDK reads `ANTHROPIC_API_KEY` from
the environment itself.

**Why.** It makes accidental credential logging structurally impossible rather than merely
unlikely — the key cannot appear in a `repr`, a serialised run config, or a log line that
dumps `self.__dict__`.

---

## D17 — Confidence is labelled a heuristic, not a probability

**Decision.** The score is a weighted combination of deterministic signals, and both the
docstring and the UI say it is not calibrated.

**Why.** Claiming calibration without a labelled dataset to calibrate against is exactly
the kind of overstatement an interviewer should catch. The weight of a disabled signal is
redistributed across the rest, so turning self-consistency off does not silently cap every
score at 0.7 and make the feature look broken.

---

## D18 — Column ambiguity is a warning, and only in a flat scope

**Decision.** An unqualified column that exists on more than one referenced table produces a
*warning*, never a rejection — and only when the statement is a single `SELECT` with no CTEs.

**Why this exists.** A real eval failure, observed during the `qwen2.5:7b` run: the model
emitted `SELECT courier_id ... FROM orders JOIN couriers ...`, which SQLite rejects at
execution with `ambiguous column name: courier_id`. Every identifier existed, so the
hallucination check was satisfied — the query was under-specified, not wrong.

**Why not a rejection.** The tempting fix is to make ambiguity a violation. It would be
wrong, because outside a flat scope `referenced_real_tables` is the union across the *whole
statement*, not the tables visible at that point. This query is perfectly unambiguous and
would be rejected:

```sql
SELECT (SELECT COUNT(*) FROM couriers WHERE courier_id = 5) AS n FROM orders
```

Rejecting valid queries to pre-empt an error the executor already reports cleanly is a bad
trade — and D7 is the cautionary tale about what a false-positive class costs.

**Why the flat-scope restriction.** In a single `SELECT` with no CTEs there is exactly one
scope, so "referenced anywhere" and "visible here" are the same set, and the check is exact.
Restricting it to where it can be exact is what makes it safe to ship at all; the
alternative was a guess or nothing.

---

## D19 — Resource limits must bound bytes, not just rows and seconds

**Decision.** The executor enforces a per-cell character cap and a total-result byte budget,
in addition to the row cap and wall-clock deadline.

**Why this exists.** Security testing found a working denial of service:
`SELECT printf('%.*c', 200000000, 'x')` returns a 200 MB string in 1.1 seconds. It passes
*every* other control — one row, well inside the timeout, an allowlisted function, an
ordinary `SELECT` AST. Every limit in the system bounded row count or elapsed time; none
bounded size, and a single row can be arbitrarily large.

**Why a budget rather than removing `printf`.** Identical reasoning to D3. `char`, `hex`,
`replace`, `group_concat` and `zeroblob` can all amplify, and the next SQLite release may
add another primitive. Blacklisting amplifiers is a losing game; bounding what any query may
*produce* is not routable around.

**Residual risk, documented rather than hidden.** On Python 3.10 the oversized value is
still allocated transiently inside SQLite before the cap discards it. The cap bounds what
propagates — API responses, logs, retained memory — not peak RSS. Python 3.11+ exposes
`Connection.setlimit(SQLITE_LIMIT_LENGTH, ...)`, which prevents the allocation outright and
is applied automatically when present.

---

## D20 — Database content that reaches the prompt is untrusted input

**Decision.** Sampled column values are filtered by a content allowlist before rendering
into the system prompt, and a column's entire sample set is withheld if any value fails.

**Why this exists.** D13 renders low-cardinality values into the prompt to stop the model
inventing literals — a real accuracy win. But those values are read from the database and
placed directly above an instruction to use them *"exactly as written"*. Testing confirmed a
value of `'; DROP TABLE t--` reached the prompt intact.

That makes any sampled column a **stored prompt-injection channel** for anyone who can write
a row — a signup form, a CSV import, a partner feed. Attacker and victim are different
users, which is what makes it worth taking seriously.

**Why withhold the whole set.** Showing a partial value domain would misinform the model
about what the column can contain, which is a correctness bug introduced by a security fix.
The trade is explicit: an attacker can *suppress* a hint, which is far cheaper than letting
them *inject* one. A warning is logged so the suppression is visible rather than silent.

**The limitation, stated rather than obscured.** This stops *syntactic* injection only. A
payload of `ignore all rules` is letters and spaces — character-class-identical to a
legitimate label like `in transit`. No charset filter can separate them, because the
difference is meaning rather than form. A test deliberately asserts that such a payload
*does* reach the prompt, so the gap stays visible.

The AST guardrails still reject destructive SQL regardless of what persuaded the model, so
semantic injection cannot cause data loss. It could cause a legal-but-wrong `SELECT`.
Nothing syntactic catches that; the mitigation is a deployment rule — do not sample columns
fed by unvalidated user input.
