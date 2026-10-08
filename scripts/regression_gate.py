#!/usr/bin/env python
"""Regression gate: did this change make the model's answers worse?

Exit codes: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (broken or stale evidence). The rules are in
``mizan.eval.gate``; the tolerance is in ``regression-gate.json``.

Usage:
    # Before pushing a prompt, model or guardrail change. Runs both suites into .gate/ and
    # compares them with the committed runs in runs/ (about 10 minutes for 7b on a CPU).
    python scripts/regression_gate.py run qwen2.5:7b
    python scripts/regression_gate.py promote qwen2.5:7b          # on PASS, make it the baseline
    python scripts/regression_gate.py run qwen2.5:7b --promote    # both in one go

    # Compare any two run directories (for example a cheaper model against the baseline):
    python scripts/regression_gate.py compare \
        runs/multilingual-qwen2.5-7b runs/multilingual-qwen2.5-0.5b

    # What CI runs. Needs no model: the evidence is the committed runs.
    python scripts/regression_gate.py ci --base-ref origin/main

Why the model runs on a laptop and not in CI: a CI runner has no Ollama, and a 7b pass takes
~10 minutes even on a 6-core laptop. So the laptop produces the evidence and commits it, and
CI checks two things about it: that it was produced by the code being merged (behaviour
fingerprint), and that it is no worse than the evidence on the base branch.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402

from mizan.eval.fingerprint import behaviour_fingerprint  # noqa: E402
from mizan.eval.gate import (  # noqa: E402
    GateInputError,
    Policy,
    Verdict,
    check_fresh,
    compare,
    empty_run,
    exists_at,
    load_run,
    render_markdown,
    worst,
)

RUNS = REPO_ROOT / "runs"
CANDIDATES = REPO_ROOT / ".gate"
RUN_FILES = ("config.json", "outcomes.jsonl", "summary.json")
SUITES = ("multilingual", "injection", "holdout")
#: What `run` evaluates unless told otherwise. The held-out suite is run on purpose, not by
#: habit (DECISIONS.md D31).
DEFAULT_SUITES = ("multilingual", "injection")
#: What `git` prints for "no previous commit" in a push event (a branch's first push).
NULL_SHA = "0" * 40

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the eval into .gate/ and gate it against runs/")
    p_run.add_argument("model", help="model name, e.g. qwen2.5:7b")
    p_run.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "mock"])
    p_run.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES), choices=SUITES)
    p_run.add_argument("--resume", action="store_true", help="continue an interrupted run")
    p_run.add_argument("--promote", action="store_true", help="on PASS, replace the baseline")

    p_pro = sub.add_parser("promote", help="gate the runs in .gate/ again and, on PASS, promote")
    p_pro.add_argument("model", help="model name, e.g. qwen2.5:7b")
    p_pro.add_argument("--provider", default="ollama", choices=["ollama", "anthropic", "mock"])
    p_pro.add_argument("--suites", nargs="+", default=list(DEFAULT_SUITES), choices=SUITES)

    p_cmp = sub.add_parser("compare", help="compare two run directories")
    p_cmp.add_argument("baseline", type=Path)
    p_cmp.add_argument("candidate", type=Path)
    p_cmp.add_argument("--baseline-ref", help="read the baseline from this git ref")

    p_fresh = sub.add_parser("check-fresh", help="were these runs produced by the current code?")
    p_fresh.add_argument("runs", type=Path, nargs="+")

    p_ci = sub.add_parser("ci", help="freshness + comparison with the base branch, for CI")
    p_ci.add_argument("--base-ref", default="", help="base commit or branch (empty: skip)")

    for p in (p_run, p_pro, p_cmp, p_ci):
        p.add_argument("--report", type=Path, help="also write the Markdown report here")

    args = parser.parse_args()
    try:
        policy = Policy.load()
        if args.command == "run":
            sections, verdict = cmd_run(args, policy)
        elif args.command == "promote":
            sections, verdict = cmd_promote(args, policy)
        elif args.command == "compare":
            report = compare(
                load_run(args.baseline, ref=args.baseline_ref), load_run(args.candidate), policy
            )
            sections, verdict = [render_markdown(report)], report.verdict
        elif args.command == "check-fresh":
            sections, verdict = cmd_fresh(args.runs)
        else:
            sections, verdict = cmd_ci(args.base_ref, policy)
    except GateInputError as exc:
        sections, verdict = [f"### ⚠️ INCONCLUSIVE\n\n> {exc}\n"], Verdict.INCONCLUSIVE

    output = "\n".join(sections)
    console.print(Markdown(output))
    console.print(f"\n[bold]Gate verdict: {verdict.value.upper()}[/] (exit {verdict.exit_code})")
    for target in filter(None, [getattr(args, "report", None), _step_summary()]):
        with Path(target).open("a", encoding="utf-8") as handle:
            handle.write(output + "\n")
    return verdict.exit_code


# ---------------------------------------------------------------------------------- run


def _settings(args: argparse.Namespace) -> Any:
    from mizan.config import Settings

    overrides: dict[str, object] = {
        "provider": args.provider,
        "db_path": REPO_ROOT / "data" / "gulf_logistics.sqlite",
        "run_dir": CANDIDATES,
    }
    if args.provider in ("ollama", "anthropic"):
        overrides[f"{args.provider}_model"] = args.model
    return Settings.from_env(**overrides)


def _run_names(args: argparse.Namespace) -> list[str]:
    from mizan.eval.harness import _slug  # the same naming the harness uses for runs/

    return [f"{suite}-{_slug(_settings(args))}" for suite in args.suites]


def cmd_run(args: argparse.Namespace, policy: Policy) -> tuple[list[str], Verdict]:
    """Run the eval into .gate/, compare each suite with its baseline, optionally promote."""
    from mizan.eval import get_suite, run_suite
    from mizan.logging import configure

    settings = _settings(args)
    settings.ensure_dirs()
    configure(settings.log_level, settings.log_dir)
    for suite, name in zip(args.suites, _run_names(args), strict=True):
        if not (RUNS / name / "outcomes.jsonl").exists():
            raise GateInputError(f"no baseline at runs/{name}: nothing to compare the run with")
        console.print(f"[bold]Running the {suite} suite with {args.model} into .gate/{name}[/]")
        cases = get_suite(suite)
        run_suite(settings, cases=cases, suite_name=suite, run_id=name, resume=args.resume)
    return _gate_candidates(args, policy, promote_on_pass=args.promote)


def cmd_promote(args: argparse.Namespace, policy: Policy) -> tuple[list[str], Verdict]:
    return _gate_candidates(args, policy, promote_on_pass=True)


def _gate_candidates(
    args: argparse.Namespace, policy: Policy, *, promote_on_pass: bool
) -> tuple[list[str], Verdict]:
    """Compare every candidate in .gate/ with its baseline; promote all of them or none."""
    names = _run_names(args)
    reports = [compare(load_run(RUNS / n), load_run(CANDIDATES / n), policy) for n in names]
    sections = [render_markdown(r) for r in reports]
    verdict = worst(r.verdict for r in reports)
    stale = [
        p for n in names if (p := check_fresh(load_run(CANDIDATES / n), behaviour_fingerprint()))
    ]
    if stale:
        # The code changed after the run started: this evidence describes other code.
        sections += [f"> ⚠️ {problem}" for problem in stale]
        verdict = worst([verdict, Verdict.INCONCLUSIVE])
    if promote_on_pass:
        if verdict is Verdict.PASS:
            for name in names:
                promote(CANDIDATES / name, RUNS / name)
            _regenerate_results()
            sections.append(
                "**Promoted:** the new runs are now the baseline in `runs/`, and "
                "`docs/results.md` was regenerated. Review the diff and commit both."
            )
        else:
            sections.append(f"**Not promoted:** the verdict is {verdict.value.upper()}.")
    return sections, verdict


def promote(candidate: Path, baseline: Path) -> None:
    """Replace the baseline's files with the candidate's (all three, or none)."""
    missing = [f for f in RUN_FILES if not (candidate / f).exists()]
    if missing:
        raise GateInputError(f"cannot promote {candidate}: missing {', '.join(missing)}")
    baseline.mkdir(parents=True, exist_ok=True)
    for name in RUN_FILES:
        shutil.copy2(candidate / name, baseline / name)


def _regenerate_results() -> None:
    """Numbers in the docs are derived from runs/, never typed (scripts/report.py)."""
    subprocess.run(  # noqa: S603 - this interpreter running this repo's own script
        [sys.executable, str(REPO_ROOT / "scripts" / "report.py")], check=True
    )


# ------------------------------------------------------------------------ fresh and ci


def cmd_fresh(run_dirs: list[Path]) -> tuple[list[str], Verdict]:
    expected = behaviour_fingerprint()
    lines, stale = [f"### Evidence freshness (current code: `{expected}`)", ""], False
    for run_dir in run_dirs:
        problem = check_fresh(load_run(run_dir), expected)
        stale = stale or problem is not None
        lines.append(f"- ⚠️ {problem}" if problem else f"- ✅ `{run_dir}` was produced by this code")
    return ["\n".join(lines) + "\n"], Verdict.INCONCLUSIVE if stale else Verdict.PASS


def cmd_ci(base_ref: str, policy: Policy) -> tuple[list[str], Verdict]:
    """Freshness of every gated run, then a comparison with the base branch's evidence."""
    if not policy.gated_runs:
        raise GateInputError("regression-gate.json lists no gated_runs")
    run_dirs = [RUNS / name for name in policy.gated_runs]
    sections, fresh = cmd_fresh(run_dirs)
    verdicts = [fresh]

    if not base_ref or base_ref == NULL_SHA:
        sections.append("No base commit to compare with (first push of a branch): skipped.\n")
        return sections, worst(verdicts)
    for run_dir in run_dirs:
        candidate = load_run(run_dir)
        report = compare(_baseline_at(run_dir, base_ref, policy, candidate), candidate, policy)
        sections.append(render_markdown(report))
        verdicts.append(report.verdict)
    return sections, worst(verdicts)


def _baseline_at(run_dir: Path, base_ref: str, policy: Policy, candidate: Any) -> Any:
    """The base branch's version of a gated run: same name, a declared older name, or none.

    A run that exists under neither name on the base is new: every case in it is reported
    as added and safety-checked, and nothing can count as a regression.
    """
    if exists_at(run_dir, base_ref):
        return load_run(run_dir, ref=base_ref)
    if (old := policy.renamed_runs.get(run_dir.name)) and exists_at(RUNS / old, base_ref):
        return load_run(RUNS / old, ref=base_ref)
    return empty_run(f"{base_ref}:(no {run_dir.name})", candidate.suite)


def _step_summary() -> str | None:
    """GitHub Actions shows Markdown appended to this file on the run's summary page."""
    return os.environ.get("GITHUB_STEP_SUMMARY") or None


if __name__ == "__main__":
    raise SystemExit(main())
