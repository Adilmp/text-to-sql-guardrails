#!/usr/bin/env python
"""Run the bilingual + injection suites for one model.

Usage:
    python scripts/run_eval.py ollama qwen2.5:7b [--limit N] [--resume]

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
from mizan.eval import build_suite, injection_suite, run_suite  # noqa: E402
from mizan.logging import configure  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["ollama", "anthropic", "mock"])
    parser.add_argument("model")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-injection", action="store_true")
    args = parser.parse_args()

    overrides: dict[str, object] = {
        "provider": args.provider,
        "db_path": REPO_ROOT / "data" / "gulf_logistics.sqlite",
    }
    if args.provider == "ollama":
        overrides["ollama_model"] = args.model
    elif args.provider == "anthropic":
        overrides["anthropic_model"] = args.model

    settings = Settings.from_env(**overrides)
    settings.ensure_dirs()
    configure(settings.log_level, settings.log_dir)

    slug = args.model.replace(":", "-").replace("/", "-")

    accuracy_summary = run_suite(
        settings,
        cases=build_suite(),
        suite_name="bilingual",
        run_id=f"bilingual-{slug}",
        resume=args.resume,
        limit=args.limit,
    )
    print(
        f"[bilingual] {args.model}: "
        f"{accuracy_summary.correct}/{accuracy_summary.n} "
        f"({accuracy_summary.accuracy:.1%})"
    )

    if not args.skip_injection:
        injection_summary = run_suite(
            settings,
            cases=injection_suite(),
            suite_name="injection",
            run_id=f"injection-{slug}",
            resume=args.resume,
        )
        print(
            f"[injection] {args.model}: "
            f"{injection_summary.correct}/{injection_summary.n} blocked "
            f"({injection_summary.accuracy:.1%})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
