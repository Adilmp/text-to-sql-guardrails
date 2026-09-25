# Security model

What this system defends against, what it does not, and what testing actually found.

Written to be read by someone deciding whether to trust it — so the limitations are as
prominent as the controls.

---

## Threat model

The adversary is **the language model itself**, or anyone able to influence it. That
includes a user typing a malicious question, a prompt-injection payload arriving through
data, and the ordinary case of a model that is simply wrong in a dangerous way.

The model is treated as **untrusted input**, not as a component. Nothing it produces is
executed without passing an independent check.

Out of scope: network security, authentication and multi-tenancy. This is a local
single-user tool. If it were exposed to real users, §6 lists what would have to be added
first.

---

## 1. Controls

| Layer | Control | Defeats |
|---|---|---|
| Extraction | Recognises all statement keywords, never sanitises | Telemetry that under-reports attacks |
| AST | `sqlglot` parse; reject non-`SELECT` anywhere in the tree | `DROP`, `DELETE`, CTE-hidden DML, stacked queries |
| AST | Reject `exp.Command` (unmodelled syntax) | `VACUUM`, and anything newer than the pinned sqlglot |
| AST | Reject `Attach`/`Pragma`/`Detach` node types | `ATTACH DATABASE` sandbox escape |
| AST | Function **allowlist**, matched on the dialect-rendered name | `load_extension`, `readfile`, `writefile` |
| AST | Every identifier checked against the live catalog | Hallucinated tables/columns; reading `sqlite_master` |
| AST | Join / subquery-depth / union ceilings | Accidental cartesian explosions |
| Prompt | Content allowlist on sampled database values | Stored **syntactic** prompt injection |
| Runtime | Read-only URI (`file:...?mode=ro`) | Writes, even with the validator bypassed |
| Runtime | `PRAGMA query_only = ON` | Writes on a connection that opened read-write |
| Runtime | Extension loading disabled | `load_extension` if the allowlist were bypassed |
| Runtime | Wall-clock deadline via progress handler | Runaway recursion, cartesian products |
| Runtime | Row cap (`fetchmany(n+1)`) | Unbounded row counts |
| Runtime | **Per-cell and total byte budget** | Memory amplification (see §2.1) |
| Introspection | **`quote_identifier()`** on every interpolated name | Injection via crafted table/column names (§2.3) |
| HTTP | CSP `default-src 'none'` + `connect-src 'self'` | Exfiltration if client escaping ever fails |
| HTTP | `frame-ancestors 'none'`, `X-Frame-Options: DENY` | Clickjacking a localhost-bound tool |
| HTTP | No CORS middleware + JSON-only bodies | Cross-origin reads; stands in for CSRF |
| Supply chain | `uv.lock` (56 pinned packages) installed exactly in CI; `pip-audit` on the hashed lockfile | Unreviewed dependency drift |
| CI | `ruff --select S` (flake8-bandit) on every lint | Regressions in the above |

The runtime controls are asserted by tests that **do not involve the validator at all**
(`tests/test_security.py::TestWriteProtectionWithoutValidator`). If every static check were
deleted, writes would still fail.

---

## 2. What testing found

Three real vulnerabilities, plus one telemetry failure (§2.4). Two vulnerabilities were found
by probing, one by static analysis plus an adversarial test that caught what the scanner
missed.

### 2.1 Memory amplification — every limit bounded the wrong dimension

```sql
SELECT printf('%.*c', 200000000, 'x')
```

Returned a **200 MB** string in **1.1 seconds**. It passed every control: one row (row cap
satisfied), 1.1 s of a 5 s budget (timeout satisfied), `printf` is a legitimate formatting
function (allowlist satisfied), and the AST is an ordinary `SELECT`.

**Root cause.** Every limit bounded *row count* or *wall-clock time*. None bounded *bytes*.

**Fix.** A per-cell cap (4 096 chars) and a total-result budget (1 000 000 chars), applied
in the executor and reported via `QueryResult.cells_truncated`. The 200 MB response now
serialises to ~4 KB.

The fix bounds the **outcome**, not the primitive. Blacklisting `printf` would have been
whack-a-mole — `char`, `hex`, `replace`, `group_concat` and `zeroblob` all amplify, and the
next SQLite release may add another. A byte budget cannot be routed around.

**Residual risk, stated plainly.** On Python 3.10 the oversized string is still allocated
transiently inside SQLite before being discarded; the cap bounds what *propagates* (API
response, logs, retained memory), not peak RSS during the query. Python 3.11+ supports
`Connection.setlimit(SQLITE_LIMIT_LENGTH, ...)`, which prevents the allocation itself and is
applied automatically when available.

### 2.2 Stored prompt injection through sampled data

Low-cardinality column values are rendered into the system prompt as
`-- one of: 'delivered', 'pending'`, immediately above an instruction to use them
*"exactly as written"*. A value of `'; DROP TABLE t--` reached the prompt intact.

Any column sampled this way is an injection channel for **anyone able to write a row** — a
signup form, a CSV import, a partner feed. This is second-order: the attacker and the victim
are different users.

**Fix.** A content allowlist (`catalog._is_prompt_safe`) withholds a column's samples if any
value contains SQL-meaningful characters or sequences. One bad value suppresses the whole
set, because a partial value domain would mislead the model about what the column can hold.

**What the fix does not cover — and cannot.** A payload of `ignore all rules` is letters and
spaces, character-class-identical to a legitimate label like `in transit`. **No charset
filter can separate them, because the difference is meaning, not form.** A test
(`test_natural_language_instructions_are_a_documented_residual_risk`) deliberately asserts
that such a payload *does* reach the prompt, so nobody later mistakes this control for
semantic coverage.

What bounds the damage: the AST guardrails reject destructive SQL regardless of what
persuaded the model to emit it, so semantic injection **cannot cause data loss**. What it
could cause is a legal-but-wrong `SELECT` returning the wrong rows. Nothing syntactic
catches that. The practical mitigation is a deployment decision: **do not sample columns
that accept unvalidated user input.**

### 2.3 SQL injection through database *identifiers*

Introspection must interpolate table and column names into SQL, because identifiers cannot
be bound as query parameters. They were wrapped in double quotes — necessary but not
sufficient. SQLite permits a double quote *inside* an identifier, written doubled:
`CREATE TABLE "a""b" (x)` is legal and stored in `sqlite_master` as `a"b`.

Naive interpolation of that name produces `SELECT COUNT(*) FROM "a"b"` — an injection point
that runs **before any guardrail**, on a connection the validator never sees. `MIZAN_DB_PATH`
accepts any SQLite file, so a crafted database would execute attacker-chosen SQL during
catalog loading.

**Fix.** `catalog.quote_identifier()` doubles embedded quotes, applied at all three
interpolation sites.

**How it was found, and why that matters.** Static analysis (`ruff --select S608`) flagged
two of the three sites. It missed `PRAGMA table_info("{table}")` entirely, because the rule
only matches SELECT-shaped strings. The third site was found by an adversarial test that
built a real database with a hostile table name and watched introspection fail.

Scanners and adversarial tests find *different* bugs. Neither alone was sufficient here.

### 2.4 A stacked statement that vanished from the telemetry

Not a way into the database, but a failure of the part that reports attacks. Extraction keeps
the first statement of the model's reply and cuts at the semicolon, and the pipeline validated
only that. `SELECT * FROM couriers; DROP TABLE couriers` therefore ran as the SELECT, and the
`DROP` appeared in no log and no rule count. The database was never at risk: the `DROP` was never
executed, and the connection is read-only. But the adversarial results said the 0.5b model had
*refused* the attack.

**Fix.** The validator also checks the model's whole reply for a second statement
(`extract.find_stacked_statement`: text after the first statement that starts with a statement
keyword and parses as SQL). Explanations after the semicolon still pass. Eval records now keep
the model's raw reply, and the runs were measured again.

**How it was found:** reviewing the documentation's claims against the code. The validator's
own tests passed, because they called it with raw text; the gap was in how the pipeline called
it. A test now covers the pipeline end to end.

---

## 3. Confirmed not exploitable

Tested and blocked. Tests pin each so a refactor cannot quietly regress them.

- **Schema exfiltration** — `sqlite_master`, `sqlite_schema`, and access via `UNION`,
  subquery or CTE. Blocked because introspection excludes `sqlite_%`, so those tables are
  not in the catalog and fail the identifier check.
- **PRAGMA table-valued functions** — `pragma_table_info('orders')`, `pragma_database_list`.
  Blocking the `PRAGMA` *statement* alone would not have caught these; they are ordinary
  `FROM` clauses reaching the same data.
- **Obfuscation** — mixed case, `/**/` splitting, leading/trailing comments, newline-stacked
  statements, full-width homoglyphs. A parser is indifferent to spelling.
- **Zip Slip** — path-traversal entries in the Spider archive are refused before extraction.
- **Write attempts at the driver level** — verified with the validator entirely out of the
  loop, including a before/after row count proving the database is unchanged.

---

## 4. Known limitations

1. **Semantic prompt injection** (§2.2) — cannot be caught syntactically.
2. **Column scope resolution** is membership-based. It catches invented identifiers in ordinary
   queries, but not a real column referenced where it is not visible, and inside a query with a
   CTE an unknown unqualified column only produces a warning. See `DECISIONS.md` D6.
3. **Peak allocation on Python 3.10** (§2.1).
4. **No authentication, authorisation or rate limiting.** Local single-user tool.
5. **`/api/schema` deliberately exposes** the schema and low-cardinality sample values. It
   is a demo endpoint; it would not ship as-is.
6. **Dialectal Arabic is untested.** The suite is Modern Standard Arabic.
7. **The demo page's CSP allows inline script and style** (`'unsafe-inline'`), because the page
   is a single self-contained file. `connect-src 'self'` still stops injected script from
   reaching another host; moving the script and styles into separate files would allow dropping
   `'unsafe-inline'`.

---

## 5. Running the security tests

```bash
uv run pytest tests/test_security.py -v
```

53 tests (one needs Python 3.11+) across schema exfiltration, resource exhaustion, stored
prompt injection, driver-level write protection, obfuscation, identifier quoting, API input
bounds and security headers.

---

## 6. What would be required before exposing this to real users

- Authentication, per-user authorisation, and rate limiting on `/api/ask`
- Row-level access control — the guardrails control *what statements run*, not *whose data
  they return*
- Python 3.11+ so `SQLITE_LIMIT_LENGTH` is enforced at the engine
- Removing `/api/schema`, or gating it behind authorisation
- Audit logging tied to user identity (structured logs exist; identity does not)
- A policy decision on which columns may be sampled into prompts (§2.2)
