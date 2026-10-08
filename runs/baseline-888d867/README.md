# Baseline: the previous pipeline on today's suites

These runs are the "before" in `docs/results.md`. They are evidence for a comparison, not
gated runs, and nothing reads them except `scripts/report.py`.

**What produced them.** The question-to-answer pipeline of commit `888d867` (per-language
prompts, no repair loop, no value grounding, 120 s request timeout, 800 output tokens), run on
this branch's suites (English, Arabic and Urdu; development, held-out and adversarial) and
scored with this branch's metrics, so the two pipelines are measured on identical questions in
an identical way. `qwen2.5:7b` through Ollama on the same 6-core CPU, 2026-10-08.

**Why the fingerprints don't match any commit.** The runs were made from a working tree
partway through the change (new suite and scoring already in, new pipeline not yet), so the
`fingerprint` in each `config.json` describes that tree. The held-out suite's per-case results
were not read while the pipeline was being tuned (DECISIONS.md D31).

**One detail that matters when reading them.** The old pipeline had no Urdu detection, so Urdu
questions took the Arabic prompt. The first question of each suite includes the cold start
(loading the model and reading the prompt), which is how the old pipeline behaved: it had no
warm-up.
