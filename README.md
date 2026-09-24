# text-to-sql-guardrails

**Bilingual Arabic/English Text-to-SQL with AST-level guardrails and deterministic
hallucination detection.**

Ask a database a question in Arabic or English. The generated SQL is parsed into an AST,
validated against the real schema, and executed inside a defense-in-depth sandbox — before
any result is returned.

## What this is

Most Text-to-SQL demos do one thing: send a schema and a question to an LLM and run
whatever comes back. This project is about the three things that have to happen *around*
that call before it is safe to put in front of a user.

**1. Arabic input against an English schema.**
Arabic NLP is the largest AI cluster in the Gulf market, and Arabic orthography is
genuinely hard: the same word admits several spellings (hamza carriers, ya vs. alef
maqsura, ta marbuta), diacritics are optional, digits come from two different Unicode
ranges, and copy-pasted text is riddled with invisible bidirectional control characters.
`mizan.nl` handles all of it — and draws a hard line between *meaning-preserving* cleanup
(safe for the model) and *lossy* folding (for matching only).

**2. Guardrails on a parsed syntax tree, never a regex.**
Regex SQL filtering is the most common security hole in Text-to-SQL systems and every
bypass is a one-liner (`SELECT 1; DROP TABLE t`, `DR/**/OP`, `WITH x AS (DELETE ...)`,
`ATTACH DATABASE`). `mizan.guardrails` parses with `sqlglot` and validates the tree, backed
by four independent runtime defences.

**3. Hallucination caught by a parser, not by another model.**
Every table, column, alias and function in the generated SQL is checked against the real
catalog. This is deterministic, costs nothing, and is strictly more reliable than asking a
second LLM whether the first one invented a column.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com/) with a model pulled (e.g. `ollama pull qwen2.5:7b`).

```bash
uv venv --python 3.10 && uv pip install -e ".[dev]"
```

```bash
uv run mizan build-db && uv run mizan schema
```

```bash
uv run mizan ask "which couriers delivered late most often?"
```

```bash
uv run mizan ask "كم عدد الطلبات المتأخرة في دبي؟"
```

No Ollama? Use the mock provider to explore the guardrails without a model:

```bash
MIZAN_PROVIDER=mock uv run mizan serve
```

## Architecture

```
question ──► nl.detect_script ──► nl.clean_for_model
                                        │
                                        ▼
                          schema.Catalog.to_prompt  (DDL + Arabic aliases + value samples)
                                        │
                                        ▼
                          providers.{ollama,anthropic,mock}.generate
                                        │
                                        ▼
                          guardrails.extract_sql   (strip fences / prose)
                                        │
                                        ▼
                          guardrails.validate      ◄── Catalog  (AST checks + schema check)
                                        │
                                        ▼
                          guardrails.execute       (read-only, query_only, deadline, row cap)
                                        │
                                        ▼
                          validate.confidence      (agreement · schema · execution)
```

## Results

Local CPU inference, 24-case paired bilingual suite + 6 adversarial prompts. Full breakdown
in [`docs/results.md`](docs/results.md), regenerated from run artifacts by
`scripts/report.py`. Every number is measured, not estimated — what was *not* measured is
listed explicitly in that document.

### Execution accuracy

| Model | Overall | English | Arabic | Latency |
|---|---:|---:|---:|---:|
| `qwen2.5:7b` | **79.2%** (19/24) | 75% (9/12) | 83% (10/12) | 88 s |
| `qwen2.5:0.5b` | **45.8%** (11/24) | 58% (7/12) | 33% (4/12) | 9 s |

**The difficulty cliff is the honest headline.** For 7b: easy 92% · medium 88% · **hard
25%**. Every failure involved a multi-table join (`join`-tagged cases: 2/6). Aggregation,
filtering, null handling and Arabic-digit parsing were all ≥88%. This is a system that
handles single-table analytics well and multi-table reasoning poorly, and no average should
be allowed to hide that.

**Arabic degrades faster than English as the model shrinks.** At 7b the two languages are
level (83% vs 75% — one case apart, which is noise at n=12; the defensible claim is *no
measurable degradation on Arabic*). At 0.5b the gap opens to 33% vs 58%. Multilingual
capability is the first thing a small model loses, so *the Arabic path is the one that
needs the larger model*, not the one that can be cheaply downgraded.

### The guardrails are what make a cheap model deployable

| Model | Dangerous statements generated | Contained |
|---|---:|---:|
| `qwen2.5:7b` | 3 | **3/3 (100%)** |
| `qwen2.5:0.5b` | 3 | **3/3 (100%)** |

The 0.5b model hallucinated non-existent identifiers **4×** against 7b's **1×**, and
attempted writes just as readily. Every instance was caught deterministically, before
execution. That is the argument for this architecture in one line: **the validation layer is
model-independent, so a cheap, weak, 10×-faster model is still safe to put in front of a
database — it is only less accurate.**

**Confidence discriminates:** mean **0.97** on correct answers vs **0.71** on wrong ones.
Real signal — but still not a calibrated probability, see [D17](DECISIONS.md).

> The adversarial metric had a subtle bug during development — it rewarded model weakness
> instead of guardrail strength. The fix and reasoning are in
> [`eval/metrics.py::classify_injection`](src/mizan/eval/metrics.py).

## Documentation

| Document | What's in it |
|---|---|
| [`SECURITY.md`](SECURITY.md) | Threat model, controls, **three real vulnerabilities found by testing**, and what is still uncovered |
| [`DECISIONS.md`](DECISIONS.md) | Every non-obvious engineering decision and why |

## Licence

MIT
