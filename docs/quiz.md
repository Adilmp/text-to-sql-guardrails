# Quiz — do you actually know your own codebase?

Answer out loud before scrolling. If you can't answer one, open the file named beside it.

Three tiers:
**A — the code you shipped.** You must be able to answer all of these.
**B — the reasoning behind it.** These are what a good interviewer actually asks.
**C — the CS underneath.** These generalise past this project.

---

## Tier A — your code

**A1.** `src/mizan/nl/normalize.py` has two public normalization functions. Name both, and
give one concrete input where using the wrong one would be a bug.

**A2.** What does `_INVISIBLE_RE` remove, and why can you not find the bug it prevents by
looking at a terminal?

**A3.** `unicodedata.normalize("NFKC", "٥")` returns what? What does that tell you about
why `_DIGIT_MAP` exists?

**A4.** Why does `clean_for_model` fold digits but *not* strip diacritics?

**A5.** In `guardrails/validator.py`, what is `exp.Command` and why is it rejected outright
rather than inspected?

**A6.** A query arrives as `SELECT 1; DROP TABLE orders`. Trace it: which function sees it
first, which rule fires, and what does `GuardrailReport.sql` contain at the end?

**A7.** `executor.py` lists four independent safety mechanisms. Name them. Which ones still
work if the entire validator is deleted?

**A8.** Why does `execute()` call `fetchmany(max_rows + 1)` instead of `fetchmany(max_rows)`?

**A9.** What is `QueryResult.fingerprint()` for, and why is it a `frozenset` rather than a
list?

**A10.** In `metrics.py`, `_normalise` returns a `Counter`, not a `set`. Give the specific
input where that difference changes the answer.

**A11.** Why does `MockProvider` match fixtures by containment rather than dictionary
lookup?

**A12.** What happens to the `agreement` signal's 0.20 weight when `self_consistency_n == 1`,
and what breaks if you skip that handling?

---

## Tier B — the reasoning

**B1.** Give three distinct bypasses of a regex-based SQL filter, and explain why an AST is
immune to all three.

**B2.** You chose an allowlist for SQL functions. Argue the opposite position as strongly as
you can, then say why you still chose allowlist.

**B3.** `exp.And` is a subclass of `exp.Func`. Explain how you discovered this, why the
obvious fix silently failed, and what the eventual fix was.

**B4.** Why is a guardrail's *false-positive* rate a security-relevant metric and not just a
quality one?

**B5.** Explain why `extract_sql` deliberately recognises `DROP` even though `DROP` is never
allowed. What broke when it didn't?

**B6.** Your hallucination detector is a parser, not a second LLM. Give three reasons, and
one thing the LLM judge would catch that yours does not.

**B7.** Why is execution accuracy the primary metric instead of comparing SQL strings? Give
a concrete pair of queries that are textually different and semantically identical.

**B8.** What is the weakness of execution accuracy? Construct an example where a wrong query
scores as correct.

**B9.** Why is the eval suite *paired* across languages? What claim does pairing let you
make that two separate suites would not?

**B10.** You are asked: "you don't speak fluent Arabic, how do you know this works?"
Give your three-part answer, and state the limitation you cannot defend away.

**B11.** Why does a blocked query return HTTP 200 rather than 400?

**B12.** The API key never lands on the `Settings` object. What class of bug does that
structurally prevent?

---

## Tier C — the CS underneath

**C1.** What is a parser AST and why is "parse, don't validate with regex" a general
security principle? Name another domain where the same rule applies.

**C2.** Define *fail-closed* vs *fail-open*. Which is the allowlist, and which is
`exp.Command` rejection?

**C3.** What is Unicode normalization? Distinguish NFC, NFD, NFKC and NFKD. Why is NFKC the
right choice here and where would it be wrong?

**C4.** What are bidirectional control characters for, legitimately? Why are they a security
concern beyond string comparison? (Look up "Trojan Source".)

**C5.** What is *defence in depth*? Why is "the validator already checks this" not a reason
to skip the read-only connection?

**C6.** `time.monotonic()` vs `time.perf_counter()` vs `time.time()` — the executor uses two
of the three. Which, where, and why is `time.time()` wrong for a deadline?

**C7.** What is a `contextvar`, and why is it correct for a run ID where a global variable
would be wrong? What specifically does it do across `async` tasks?

**C8.** Explain exponential backoff with *full jitter*. What failure mode does the jitter
prevent, and what is that failure mode called?

**C9.** Why retry a timeout but not a 400? State the general rule for what is safe to retry.

**C10.** What does *idempotent* mean, and which parts of this codebase rely on it?

**C11.** In a bash pipeline `cmd | tail -5`, whose exit status does `$?` report? How do you
get the first command's? (You lost fifteen minutes to this.)

**C12.** What is the difference between a *heuristic score* and a *calibrated probability*?
What would you need to turn the confidence score into the latter?

**C13.** Why does `dataclass(frozen=True)` matter for `GuardrailPolicy` specifically, beyond
general good practice?

**C14.** Self-consistency groups candidates by result rather than by SQL text. Frame this in
terms of *equivalence classes*: what relation are you quotienting by, and why is the
text-equality relation the wrong one?

**C15.** `while pgrep -f "run_eval.py qwen2.5:7b"; do sleep 20; done` never exited, even
after that process died. Why? Name the general class of bug, and give two fixes.

**C16.** Your eval reports 79.2% overall but 25% on `hard` cases and 33% on `join`-tagged
cases. Why is reporting the sliced numbers alongside the headline not just "being thorough"
— what specific way of being wrong does it protect you from?

---

## Answer key

<details>
<summary>Tier A</summary>

**A1.** `clean_for_model` (meaning-preserving: NFKC, invisible removal, tatweel, digits,
punctuation) and `normalize_for_matching` (adds diacritic stripping, letter folding, case
folding). Using the lossy one on model input would send `محمد` where the user wrote
`مُحَمَّد`; using the safe one as a dictionary key means `متأخرة` and `متاخره` miss each other.

**A2.** Zero-width and bidirectional formatting characters (`U+200B–200F`, `U+202A–202E`,
`U+2066–2069`, BOM, soft hyphen). They have no visual width, so two strings that differ by
several of them render identically — the difference exists only in the bytes.

**A3.** `"٥"` — unchanged. Arabic-Indic digits have no compatibility decomposition, so the
normalization everyone reaches for does nothing here and an explicit table is mandatory.

**A4.** Digit folding preserves meaning (`٥٠٠` and `500` are the same number) and materially
helps the model emit the right literal. Diacritics change what the text says, and the model
reads `أحمد` fine as written.

**A5.** sqlglot's catch-all node for syntax it does not model (`VACUUM`, `REINDEX`, anything
newer than the installed version). It is rejected because "we don't know what this does" is
not a safe basis for execution — enumerating known-bad types would wave through everything
unrecognised.

**A6.** `extract_sql` sees it first and trims at the semicolon; `has_multiple_statements`
(checked on the *raw* text, before trimming) fires `stacked_statements`, and parsing the
whole string also fires `parse_error`. `GuardrailReport.sql` is `""` — a rejected report
never returns runnable SQL.

**A7.** (1) `file:...?mode=ro` URI, (2) `PRAGMA query_only = ON`, (3) extension loading
disabled, (4) progress-handler deadline. All four still work with the validator deleted —
that is the point, and `tests/test_executor.py` asserts it without the validator in the loop.

**A8.** Fetching one extra row is how you distinguish "exactly `max_rows` rows existed" from
"more existed and we truncated", without materialising the full result set.

**A9.** Order-independent identity of a result set, used for self-consistency grouping. A
`frozenset` because it must be hashable to serve as a dict key when grouping candidates.

**A10.** `[a, a, b]` vs `[a, b]`. A `set` calls those equal; they are different answers. A
`GROUP BY` returning duplicate rows would be scored correct against a gold query that does
not.

**A11.** The provider receives the wrapped prompt (`Question: ...\nSQL:`), never the bare
question, so exact lookup never matches. Longest-match containment on the *normalized* form
also means the mock exercises the Arabic pipeline instead of bypassing it.

**A12.** The weight is redistributed: the score is normalised by the weights actually in
play. Skip it and a perfect answer caps at 0.80 with the default settings — nothing crashes,
confidence just looks quietly broken forever.

</details>

<details>
<summary>Tier B</summary>

**B1.** `SELECT 1; DROP TABLE t` (stacking defeats prefix matching); `DR/**/OP TABLE t`
(comments defeat keyword matching); `WITH x AS (DELETE FROM t RETURNING 1) SELECT * FROM x`
(nesting defeats top-level inspection). An AST is immune because it encodes what the
statement *is* after lexing and parsing, so spelling, casing, comments and nesting are
already resolved.

**B2.** *For denylist:* it never rejects a legitimate query, so you never lose accuracy to
your own guardrail; the set of genuinely dangerous functions is small and well known; an
allowlist needs constant maintenance as analysts want new functions. *Why allowlist anyway:*
the failure modes are asymmetric. A denylist that is wrong is a breach, and you find out
from an incident. An allowlist that is wrong is a rejected query, and you find out from your
own eval numbers. Choose the mistake you can see.

**B3.** Found by the allowlist rejecting `and()` on a normal `WHERE`. The obvious fix —
excluding `Connector`/`Binary` base classes — was written with an
`issubclass(node, exp.Expression)` guard, and those two are mixins that do not derive from
`Expression`, so the guard filtered them out of the exclusion list and produced a
one-element tuple. Real fix: render each node to SQLite and read the name off the generated
text, which ignores operators automatically because they render without a call form.

**B4.** Because a guardrail nobody can ship with gets turned off. High false-positive rates
create pressure to loosen the policy or bypass it, and a disabled control is worth zero.
Separately, the false positives masquerade as model failure, so you spend your effort tuning
prompts while the real bug sits in the validator.

**B5.** So the validator, not the extractor, makes every security decision. When extraction
recognised only `SELECT|WITH`, a generated `DROP TABLE` was trimmed to `""` and reported as
`parse_error` — the database was safe, but the telemetry said "model emitted gibberish" when
the truth was "model attempted a write". Safe *and* wrong is the worst combination, because
nothing alerts you.

**B6.** Deterministic; free and instant; cannot itself hallucinate. What an LLM judge would
catch that this does not: *semantic* wrongness with valid identifiers — `WHERE status =
'pending'` when the question asked about delivered orders. Every name exists, so the parser
is satisfied.

**B7.** Because join order, alias names, CTE-vs-subquery and `COUNT(*)` vs `COUNT(o.id)` all
vary freely among correct answers. Example: `SELECT COUNT(*) FROM orders WHERE delivered_at
> promised_at` and `SELECT COUNT(order_id) FROM orders WHERE promised_at < delivered_at`.

**B8.** Right rows, wrong reason. On this database, `WHERE segment = 'enterprise'` and
`WHERE customer_id IN (...)` could select the same rows by coincidence on a small table. The
per-tag breakdown exists so a suspiciously perfect slice is visible rather than averaged
away.

**B9.** Identical gold SQL, identical schema, identical difficulty — only language varies. So
an accuracy gap between `en_*` and `ar_*` is attributable to *language*. Two unpaired suites
could not support that claim, because the difference might be that one suite is harder.

**B10.** (1) Normalization is tested against code points, not meaning — `U+0623 → U+0627` is
verifiable without reading Arabic. (2) Every Arabic fixture carries an English gloss, so
question/gold-SQL correspondence is reviewable. (3) The suite is paired, isolating language
as the variable. The limitation: it is all Modern Standard Arabic. Gulf dialect input is
untested, and you should say so rather than imply coverage you do not have.

**B11.** A blocked query is a *successful analysis* whose result is "not safe to run". The
client needs the violation list to display it. A 4xx pushes callers to wrap an expected
outcome in `try/except` and conflates it with transport failures.

**B12.** Credential leakage through incidental serialisation — a `repr` in a traceback, a
pickled config, a run artifact, a log line that dumps `self.__dict__`. Not storing it makes
those impossible rather than merely unlikely.

</details>

<details>
<summary>Tier C</summary>

**C1.** A tree representing a program's grammatical structure after lexing and parsing.
"Parse, don't validate" generalises because a parser produces a *typed, unambiguous*
representation, whereas a regex tests a surface property of the text — and surface
properties have infinitely many spellings. Same rule: HTML sanitisation (parse the DOM,
never regex tags), path traversal (resolve and normalise, never string-match `..`), email
validation, URL host checks.

**C2.** Fail-closed denies on uncertainty; fail-open permits. The allowlist is fail-closed
(unknown function → reject) and so is the `exp.Command` rejection (unknown statement →
reject). Security controls should fail closed; availability-critical ones sometimes should
not, which is a real trade-off, not a slogan.

**C3.** Mapping equivalent Unicode sequences to a canonical form. NFC composes, NFD
decomposes (both lossless, canonical equivalence). NFK* adds *compatibility* mappings —
ligatures, presentation forms, superscripts — which is lossy in formatting. NFKC is right
here because Arabic presentation forms (`U+FB50–FEFF`) must fold back to ordinary letters.
It would be wrong where the compatibility distinction carries meaning: `x²` → `x2` destroys
an exponent, and `ﬁ` → `fi` changes a typographic record.

**C4.** Legitimately, they control the display order of mixed LTR/RTL text. The security
concern is **Trojan Source**: overrides can make source code *render* in a different order
than it *compiles*, so a reviewer sees one program and the compiler sees another. That is
why stripping them is a correctness measure, not just a normalization tidy-up.

**C5.** Multiple independent controls so that no single failure is fatal. "The validator
checks this" assumes the validator is correct — and three validator bugs were found while
building this. The read-only connection does not depend on the validator being right, which
is the entire value.

**C6.** `time.monotonic()` for the deadline (never goes backwards; immune to NTP adjustment
and DST), `time.perf_counter()` for measuring elapsed latency (highest resolution).
`time.time()` is wall-clock and can jump backwards, which would make a deadline either fire
early or never.

**C7.** A variable whose value is scoped to the current execution context, with correct
isolation across threads *and* `async` tasks — each task inherits a copy at creation and its
writes do not leak to siblings. A global would be shared across concurrent requests, so two
simultaneous API calls would overwrite each other's run ID.

**C8.** Retry after a delay drawn uniformly from `[0, 2^attempt)`. Without jitter, clients
that failed together retry together, re-synchronising into repeated simultaneous bursts —
the **thundering herd**, which turns a transient blip into a sustained outage.

**C9.** A timeout may be transient — the condition can differ on the next attempt. A 400
means the request itself is malformed, so attempt three fails identically. General rule:
retry only when the failure is plausibly *transient* and the operation is *idempotent* or
safely repeatable.

**C10.** An operation with the same effect applied once or many times. Relied on by:
`logging.configure` (guarded so double-calling does not duplicate handlers — otherwise every
line prints twice), `build()` with `overwrite`, and the eval `--resume` path, which must be
safe to re-run over a partially completed run.

**C11.** `$?` is the *last* command's — `tail`. Use `${PIPESTATUS[0]}` in bash, or
`set -o pipefail` to make the pipeline fail if any stage does.

**C12.** A heuristic ranks; a calibrated probability means that among all answers scored
0.8, about 80% are actually correct. To calibrate you need a labelled set of
correct/incorrect outcomes, then fit a mapping (Platt scaling or isotonic regression) from
raw score to observed frequency, and validate it on held-out data with a reliability diagram
or Brier score.

**C13.** Because the serialised policy written into each run directory must describe the
rules that actually applied. A mutable policy could be changed mid-run, making the recorded
artifact a lie about the run it claims to document — and eval artifacts that misrepresent
their own configuration are worse than no artifacts.

**C14.** You are quotienting the set of candidate queries by the relation "produces the same
result set", and picking the largest equivalence class. Text equality is a *finer* relation
— it splits semantically identical queries into separate classes — so voting under it
measures agreement about phrasing rather than agreement about the answer, and a model that
is consistently right in three different spellings would score as maximally inconsistent.

**C15.** `pgrep -f` matches the full command line of *every* process, and the watching
shell's own command line contains the pattern — because the pattern is written inside it. So
it matched itself and waited for a process that could only exit once the wait ended. General
class: **a check that inspects a namespace the checker is itself part of can observe
itself** (same family as `grep` matching its own pattern in `ps` output). Fixes: exclude your
own PID (`pgrep -f pat | grep -v $$`), match on something the watcher does not contain (a PID
file), or wait on the process handle directly rather than pattern-matching a process table.

**C16.** It protects against an aggregate that is true but *unactionable* — and against
fooling yourself. 79.2% sounds like "mostly works"; 25% on hard and 33% on joins says
precisely what to fix next and what not to promise. It also guards the reverse error: a
suspiciously perfect slice (100% on `null_handling`) is a prompt to check whether those cases
are actually easy or whether the metric is being satisfied for the wrong reason — which
matters because execution accuracy can be right for the wrong reason (see B8). An average
hides both failure modes.

</details>
