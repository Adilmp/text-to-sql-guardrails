#!/usr/bin/env python
"""Run evaluation suites for one model (default: multilingual + injection).

Usage:
    python scripts/run_eval.py ollama qwen2.5:7b [--suites multilingual holdout injection]
                               [--limit N] [--resume] [--run-dir DIR]

The held-out suite only runs when named: it exists to be run once, on a finished pipeline
(DECISIONS.md D31).

``--run-dir`` writes the run somewhere other than ``runs/``. The regression gate needs this:
a new run written into ``runs/`` would overwrite the very baseline it is meant to be compared
with (``scripts/regression_gate.py run`` does this for you).

Kept as a script rather than a CLI subcommand because it is the thing most likely to be
launched detached and left running for an hour, and a script is easier to point at with
nohup than a Typer entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mizan.config import Settings  # noqa: E402
from mizan.eval import SUITES, get_suite, run_suite  # noqa: E402
from mizan.logging import configure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["ollama", "anthropic", "mock"])
    parser.add_argument("model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--suites", nargs="+", choices=sorted(SUITES), default=["multilingual", "injection"]
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="where to write the run (default: runs/)",
    )
    args = parser.parse_args()

    overrides: dict[str, object] = {
        "provider": args.provider,
        "db_path": REPO_ROOT / "data" / "gulf_logistics.sqlite",
    }
    if args.run_dir is not None:
        overrides["run_dir"] = args.run_dir.resolve()
    if args.provider == "ollama":
        overrides["ollama_model"] = args.model
    elif args.provider == "anthropic":
        overrides["anthropic_model"] = args.model

    settings = Settings.from_env(**overrides)
    settings.ensure_dirs()
    configure(settings.log_level, settings.log_dir)

    slug = args.model.replace(":", "-").replace("/", "-")

    for suite in args.suites:
        summary = run_suite(
            settings,
            cases=get_suite(suite),
            suite_name=suite,
            run_id=f"{suite}-{slug}",
            resume=args.resume,
            limit=args.limit,
        )
        print(f"[{suite}] {args.model}: {summary.correct}/{summary.n} ({summary.accuracy:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
