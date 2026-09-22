# Interview notes — the problems that were actually hard

Written to be spoken aloud. Each entry is a real problem hit while building this, what was
tried first, why it failed, and what the fix taught. Nothing here is hypothetical — every
one of these changed code in the repository.

---

## 1. The guardrail that rejected `AND`

**File:** `src/mizan/guardrails/validator.py` → `_function_name`, `_collect_functions`

**Symptom.** The function allowlist rejected an ordinary query:

```
SELECT COUNT(*) FROM orders WHERE delivered_at IS NOT NULL AND delivered_at > promised_at
→ function_not_allowed: and() is not on the allowlist
```

**Why it happened.** I was collecting function calls with `tree.find_all(exp.Func)`. In
sqlglot, `exp.And` has the MRO `And → Connector → Binary → Func` — boolean operators, comparisons
and predicates are all `Func` subclasses. So every `AND`, `OR` and `LIKE` in the query was
being reported as a function named after its class.

**The fix that silently did nothing.** The obvious move is to exclude those base classes:

```python
_OPERATOR_BASES = tuple(
    node for node in (getattr(exp, n, None) for n in ("Connector", "Binary", "Unary"))
    if isinstance(node, type) and issubclass(node, exp.Expression)   # ← the bug
)
```

This produced a *one-element* tuple containing only `Unary`. `Connector` and `Binary` are
mixins that do not themselves derive from `exp.Expression`, so my own guard filtered them
out of my own exclusion list. The check still failed, and it failed quietly — the tuple was
non-empty, so nothing looked broken.

**What I'd say about it.** The lesson is not "read the sqlglot source". It is that a
defensive `isinstance`/`issubclass` guard written to make code robust can silently discard
the thing it was protecting. I only found it by printing the resolved tuple, which is why
the regression test asserts on behaviour (`_collect_functions` returns an empty set for a
query of pure operators) rather than on the internals.

---

## 2. sqlglot renames your functions behind your back

**File:** same

**Symptom.** `strftime('%Y', placed_at)` was rejected even though `strftime` is on the
allowlist.

**Why.** sqlglot is a *transpiler*, so it canonicalises across dialects. SQLite's `strftime`
parses to `exp.TimeToStr`, and its argument gets wrapped in `exp.TsOrDsToTimestamp`. My
allowlist was being compared against sqlglot's internal vocabulary — names the user never
typed and the database will never see.

**Fix.** Render the node back to the target dialect and read the name off the generated
text:

```python
rendered = node.sql(dialect="sqlite")      # exp.TimeToStr -> "STRFTIME('%Y', placed_at)"
match = _FUNC_CALL_RE.match(rendered)      # -> "strftime"
```

This solved problem 1 as a side effect: operators render as `a AND b`, with no `name(`
prefix, so they produce no name and are correctly ignored. One mechanism, both bugs.

**The general point.** An authorisation check must run against *what will execute*, not
against an intermediate representation. That framing generalises well beyond SQL.

---

## 3. The hallucination detector flagged the query's own aliases

**File:** `src/mizan/guardrails/validator.py` → `_defined_aliases`

**Symptom.**

```sql
SELECT courier_id, COUNT(*) AS late_deliveries FROM orders
GROUP BY courier_id ORDER BY late_deliveries DESC
→ unknown_column: 'late_deliveries' does not exist on any referenced table
```

**Why.** `ORDER BY late_deliveries` parses as an `exp.Column`. Checked against the catalog,
it genuinely does not exist there — and the check was completely wrong, because the query
defines it two lines earlier.

**Fix.** Collect names the statement introduces (`exp.Alias`, and column lists on
`exp.TableAlias`) and exempt them. Safe by construction: an alias cannot be a hallucination,
because its definition is in the same statement.

**Why it mattered more than it looks.** This was a false-positive class hitting every query
that aliases an aggregate — which is most analytical SQL. Had I not caught it, the eval
numbers would have shown a model that "can't write SQL", and I would have spent the time
tuning prompts instead of fixing my validator. **A guardrail's false-positive rate is a
first-class metric**, which is why `tests/test_guardrails.py` has a `TestNoFalsePositives`
class as large as the attack suite.

---

## 4. Sanitising in the wrong layer made my security metrics lie

**File:** `src/mizan/guardrails/extract.py` → `_STATEMENT_KEYWORDS`

**Symptom.** A test asserting that a generated `DROP TABLE orders` is reported as
`write_operation` failed — it was reported as `parse_error`.

**Why.** `extract_sql` recognised only `SELECT|WITH` as a statement start. A `DROP` matched
nothing, got trimmed to an empty string, and the validator dutifully reported "unparseable".

**Why this is worse than it sounds.** The database was never at risk — the statement was
discarded either way. But the guardrail telemetry claimed the model had emitted *gibberish*
when it had actually attempted a *write*. If you are running an eval to answer "how often
does this model try to do something destructive", that number was silently wrong in the safe
direction, which is the most dangerous kind of wrong: it looks fine.

**Fix.** Extraction recognises every statement keyword, including forbidden ones, and the
validator makes all security decisions. **Exactly one layer judges; every other layer
reports faithfully.**

---

## 5. Two strings that look identical and aren't

**File:** `src/mizan/nl/normalize.py` → `strip_invisible`

Copying Arabic from a browser, a PDF or a spreadsheet frequently embeds bidirectional
control characters — `U+200F RIGHT-TO-LEFT MARK`, embeddings, isolates. They are
zero-width. Two strings that render identically on screen compare unequal, and no amount of
staring at a terminal reveals why.

This is the first thing both normalization functions do, and there is a test
(`test_visually_identical_strings_compare_equal_after_normalizing`) that exists specifically
to document the failure mode for whoever reads this next.

**The interview version:** "Any system taking Arabic text input that does not strip bidi
controls before comparing or keying on strings has an intermittent bug it cannot see."

---

## 6. NFKC does not do what people assume for Arabic digits

**File:** `src/mizan/nl/normalize.py` → `_DIGIT_MAP`

`unicodedata.normalize("NFKC", "٥")` returns `"٥"`. Arabic-Indic digits have no
compatibility decomposition, so the standard normalization everyone reaches for leaves them
untouched. An explicit translation table is mandatory.

There are also **two** ranges, not one: `U+0660–U+0669` (Arabic) and `U+06F0–U+06F9`
(Extended Arabic-Indic, used in Persian and Urdu). They look nearly identical and are
different code points. Both are mapped.

This matters specifically for Text-to-SQL: a number in the question has to survive into the
SQL as a literal. `٥٠٠` reaching the model unconverted is a wrong answer with no error
anywhere. There is a test asserting NFKC's behaviour directly, so if the assumption ever
changes upstream the comment explaining the table does not quietly become a lie.

---

## 7. Two normalization strengths, and why collapsing them is a bug

**File:** `src/mizan/nl/normalize.py`

`normalize_for_matching` folds `مُحَمَّد` → `محمد` and `متأخرة` → `متاخره` (Lucene's
`ArabicNormalizer` rules). Perfect for a lookup key. Catastrophic for anything displayed or
sent to a model.

So there are two functions with different call sites, and the split is enforced by tests.
Digit folding sits in the *safe* pass because it preserves meaning; diacritic stripping and
letter folding sit in the *lossy* one because they do not.

**The generalisable claim:** normalization is not one operation. It is a family, indexed by
what you are about to do with the result.

---

## 8. A version-dependent security check

**File:** `src/mizan/guardrails/validator.py` → `_optional_nodes`, `_DANGEROUS_NODES`

I rejected `exp.Command` — sqlglot's catch-all for unmodelled syntax — expecting it to cover
`ATTACH DATABASE` (a full sandbox escape on SQLite, since it opens a second file). Testing
showed the installed version models `ATTACH` and `PRAGMA` as *dedicated* classes, so they
never reach the `Command` branch. They were still blocked, but by the generic
`not_a_select` rule, which meant the rejection breakdown attributed a sandbox-escape attempt
to "wasn't a SELECT".

Fix: resolve those node types by name with `getattr`, so the check is correct whether the
installed sqlglot models them or not, and keep the `exp.Command` rejection for everything
newer than the installed version. **A security check that only enumerates known-bad types
waves through everything it has never heard of** — the wrong failure direction.

---

## 9. `| tail` ate an exit code and cost me fifteen minutes

**Symptom.** A background dependency install reported exit code 0. `sqlglot` was not
installed.

**Why.** The command was `uv pip install -e ".[dev]" 2>&1 | tail -5`. The pipeline's exit
status is the *last* command's — `tail` succeeded. The real failure (`OSError: Readme file
does not exist: README.md`, because `pyproject.toml` declared `readme` before I had written
one) scrolled past inside the truncated output.

Small, mundane, and exactly the kind of thing that wastes real time. `${PIPESTATUS[0]}`
exists for this.

---

## 10. Testing the wrong thing, twice

Two test failures that were **bugs in the tests**, not the code — worth keeping because
recognising the difference quickly is most of debugging:

- `test_fingerprint_is_order_independent` compared `ORDER BY x ASC` against `ORDER BY x DESC`
  with a 50-row cap on a 120-row table. Those are genuinely different 50-row *subsets*. The
  test was measuring the row cap, not order-independence. Fixed by using a query that
  returns fewer rows than the cap, and asserting `not truncated` so the test can never
  silently drift back into measuring truncation.
- The mock provider's fixtures never matched, because the provider receives the *wrapped*
  prompt (`Question: ...\nSQL:`), not the bare question. Fixed with longest-match containment
  on the normalized form — which has the side benefit that the mock now exercises the Arabic
  normalization pipeline rather than bypassing it.

---

## 11. Confidence that silently capped itself

**File:** `src/mizan/validate/confidence.py` → `score_answer`

Self-consistency contributes a 0.20-weighted `agreement` signal. With
`self_consistency_n = 1` there is nothing to measure. The naive implementation scored that
as `0.0 × 0.20`, which meant **a perfect answer could never exceed 0.80** with the feature
off — and the default is off.

Nothing would have crashed. Confidence would just have looked mildly broken forever.

Fix: normalise by the weights actually in play, so a missing measurement redistributes
rather than counting as a bad one. The test
(`test_disabling_self_consistency_does_not_cap_the_score`) asserts a clean answer reaches
exactly 1.0.

**The general trap:** *absent* and *zero* are different, and a weighted-average scorer that
conflates them fails silently.

---

## 12. A watcher that waited for itself

I tried to chain the second eval run behind the first:

```bash
while pgrep -f "run_eval.py ollama qwen2.5:7b" >/dev/null; do sleep 20; done
python scripts/run_eval.py ollama qwen2.5:0.5b
```

It never fired. `pgrep -f` matches against the **full command line of every process**, and
the watcher's own shell has that exact string in *its* command line — because the pattern is
written inside it. The loop matched itself and waited forever for a process that could only
exit once the loop ended.

The general shape is worth keeping: **any "wait for X" check that inspects a namespace the
waiter is itself part of can see itself.** Same class as `grep` in a pipeline matching its
own pattern in `ps` output. Fixes are to exclude your own PID
(`pgrep -f pat | grep -v $$`), match on something the watcher does not contain (a PID file),
or — the right answer — wait on the process handle directly instead of pattern-matching a
process table.

---

## 13. A demo that lied, and the two reasons why

The reviewer clicked "drop the orders table" in the web demo and got back
`SELECT COUNT(*) FROM orders`, 900 rows, **97% confidence, no block**. Their reaction — "is
it even working?" — was the correct reaction. Two separate faults, and neither was in the
guardrails.

**Fault 1: a stale process.** The server was started at 12:08:15. The fixture that makes the
mock emit `DROP TABLE` was added at 12:09:28 — 73 seconds later. `uvicorn` had the old module
in memory and had no reason to reload. The running code and the code on disk had silently
diverged, and every test on disk passed. The fix is `--reload` in the dev launch config, but
the habit is more general: **when observed behaviour contradicts a passing test suite,
suspect the process before you suspect the logic.** I confirmed it by diffing the process
start time against the file mtime rather than by reasoning about it.

**Fault 2: a fallback that fabricated.** The mock provider's default for an unmatched
question was `SELECT COUNT(*) FROM orders`. That returns a plausible integer for *any*
question. Two of the five demo buttons had no matching fixture — a plural/singular mismatch
on one — so they silently produced `900` for "how many orders were delivered late?" where the
right answer is `188`. Valid SQL, real execution, high confidence, wrong answer, nothing on
screen to indicate it.

That is the same failure mode this whole project exists to attack — a confident wrong answer
with no error anywhere — and I had built one into my own demo. The default is now
`SELECT 'no mock fixture for this question'`, which announces itself.

**The generalisable rule: a fallback must never be indistinguishable from a real result.**
Silent plausible defaults are worse than loud failures, because they consume the reviewer's
trust rather than their attention.

**The same lesson recurred an hour later.** A reviewer typed an Arabic question into the
live demo and reported it "stuck on generating". It was not stuck — CPU inference genuinely
takes ~90s, and an `mizan eval` run was queued at the same model (Ollama serialises requests
per model), so the wait was several minutes. But the UI showed a static "Generating…" with
no elapsed time, which is *indistinguishable from a hang*. It now counts seconds, explains
that ~90s is normal, and past 150s suggests checking for a competing process.

**Long-running operations must prove they are alive.** A spinner that cannot distinguish
"working" from "dead" has failed at the only job a spinner has.

There is a third, quieter fix: the page now shows a banner naming the backend. Mock output
and model output rendered identically before, which meant a viewer could not tell a fixture
lookup from generation. For a portfolio piece shown to strangers that is not a cosmetic
issue — it is the difference between a demo and a magic trick.

---

## 14. A flag that did nothing, and a directory that lied

Two bugs from one root cause, both found by a reviewer simply *using* the tool.

**`--resume` was a silent no-op.** `scripts/run_eval.py` passed an explicit stable
`run_id`; the CLI did not. So `run_suite` minted a fresh timestamped directory on every
invocation, `--resume` read *that* empty directory, found nothing to skip, and re-ran all 24
cases — 36 minutes at 90s each — while printing a perfectly normal summary. The flag was
inert and nothing said so.

Two fixes, because one is not enough. The CLI now derives a stable id from suite+model when
`--resume` is passed. And the harness **warns when resume finds nothing**:

```python
elif resume:
    logger.warning("resume requested but no previous outcomes found - running the full suite")
```

The second fix is the important one. The first makes the correct case work; the second makes
the *incorrect* case audible. A no-op flag that reports success is the same failure family as
§4 (telemetry that lied) and §13 (a fallback that fabricated) — **the system does the wrong
thing and looks fine doing it.**

**The slug fell through to the wrong model.** The helper read
`ollama_model if provider == "ollama" else anthropic_model`, so *mock* runs were filed as
`runs/bilingual-claude-sonnet-5/` — an artifact directory attributing measurements to a model
that never executed. On a project whose entire claim is "these numbers were measured, not
asserted", a run directory that misnames its own model is a correctness bug, not a typo.

The root cause in both cases is an `if/else` standing in for an exhaustive match over an
enum. `ProviderName` has three values; the code branched on one and lumped the rest together.
A test now asserts both entry points produce the same slug for every provider, because they
must agree or a run started by one cannot be resumed by the other.

---

## 15. Infrastructure decisions that were really engineering decisions

- **The 4.7 GB model pull was starving the dependency install.** Both were competing for a
  very slow connection (11 s for a PyPI index page). I paused the pull, let deps finish,
  then resumed. Obvious in hindsight; the general habit is to notice when two background
  jobs share a bottleneck.
- **Eval results are written one JSON line at a time, flushed immediately.** At 60–120 s per
  question on CPU, a 30-case suite is a 40-minute run, and losing it at case 29 is
  unacceptable. `--resume` skips completed cases and the loader tolerates a truncated final
  line, because a process killed mid-write leaves exactly that.
- **The small model was kept rather than deleted.** `qwen2.5:0.5b` is far too weak for this
  task. That makes it useful: a size-vs-accuracy curve across 0.5b → 7b → Claude is a
  stronger artifact than a single number, and it costs nothing to keep.

---

## Questions I should expect, and honest answers

**"Your confidence score — is it calibrated?"**
No, and the code says so. It is a weighted heuristic over deterministic signals, intended to
rank and to explain, not to state a probability. Calibrating it would need a labelled set of
correct/incorrect answers large enough to fit against, which I do not have. Claiming
otherwise would be false precision.

**"You don't speak fluent Arabic. How do you know the Arabic works?"**
Three things carry the weight. The normalization layer is tested against *code points*, not
meaning — `U+0623 → U+0627` is verifiable without knowing what the word means. Every Arabic
fixture in the suite carries an English gloss, so a reviewer can check that question and
gold SQL correspond. And the suite is *paired*: identical gold SQL for the English and
Arabic versions of each question, so any accuracy gap is attributable to language rather
than to difficulty. What I cannot claim is dialectal coverage — the suite is Modern Standard
Arabic, and Gulf dialect input is untested. That is a stated limitation, not a solved
problem.

**"Why not just use LangChain's SQL agent?"**
For a production system I might, for the plumbing. The part that matters here is not the
plumbing. It is the validation layer, and writing it myself is the only reason I can tell
you that operators are `Func` subclasses in sqlglot, that `extract` must not sanitise, and
what a guardrail's false-positive rate costs you. Those are not things a wrapper teaches.

**"What breaks first at scale?"**
The column check. It is membership-based with alias scoping, so it catches every *invented*
identifier but not a real column referenced in a scope where it is not visible. On a
200-table warehouse schema with repeated column names across tables that starts to matter,
and the fix is sqlglot's qualifier with a full schema. It is written up in `DECISIONS.md`
(D6) as a known limitation rather than left to be discovered.
