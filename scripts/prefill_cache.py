#!/usr/bin/env python
"""Pre-fill the server's answer cache with the questions people are likely to ask.

Every question here goes through the model once, exactly as if a user had asked it, and is
stored only if the cache would store it anyway (confident, no repair, every filtered value
found in the data). Afterwards those questions answer in milliseconds instead of seconds.

What is in the list, in the order it is asked:

1. the demo page's example questions;
2. every question in the development and held-out eval suites (English, Arabic, Urdu);
3. general questions about the whole database, in the three languages;
4. templated questions for every city, country, order status, product category, customer
   segment, courier, warehouse and product, in the three languages.

The list itself lives in ``mizan/questions.py``, because the server's suggestions use it too.

The answers are the model's own, mistakes included. The eval suites' gold queries are known,
but planting them would make the demo look more accurate than the model is, so they are
never used here.

The run is resumable: questions already cached are skipped, so stopping and starting again
loses nothing. The server can stay up meanwhile; a visitor's question waits at most for the
one question being pre-filled (Ollama answers one request at a time).

Usage:
    python scripts/prefill_cache.py                  # everything (~450 questions, ~1 h on a CPU)
    python scripts/prefill_cache.py --only examples suite
    python scripts/prefill_cache.py --dry-run        # list the questions, ask nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from mizan.cache import (  # noqa: E402
    AnswerCache,
    CachedTextToSQL,
    CacheSettings,
    cacheable,
    canonical_question,
)
from mizan.config import Settings  # noqa: E402
from mizan.generate import TextToSQL  # noqa: E402
from mizan.logging import configure  # noqa: E402
from mizan.providers import build_provider  # noqa: E402
from mizan.questions import curated_questions, names_from_db  # noqa: E402
from mizan.schema import load_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    groups = (
        "examples",
        "suite",
        "general",
        "city",
        "country",
        "status",
        "category",
        "segment",
        "courier",
        "warehouse",
        "product",
    )
    parser.add_argument("--only", nargs="+", choices=groups, help="pre-fill only these groups")
    parser.add_argument("--limit", type=int, help="stop after this many model calls")
    parser.add_argument("--dry-run", action="store_true", help="list the questions and exit")
    args = parser.parse_args()

    # The same settings the server uses, so entries land in the server's cache context.
    settings = Settings.from_env()
    questions = curated_questions(names_from_db(settings.db_path))
    if args.only:
        questions = [(g, q) for g, q in questions if g in args.only]
    if args.dry_run:
        for group, question in questions:
            print(f"{group:10} {question}")
        print(f"\n{len(questions)} questions")
        return 0

    configure(settings.log_level, settings.log_dir, console=False)
    cache_settings = CacheSettings.from_env()
    if not cache_settings.enabled:
        print("MIZAN_CACHE is off: nothing to pre-fill", file=sys.stderr)
        return 1
    catalog = load_catalog(settings.db_path)
    engine = TextToSQL(catalog, build_provider(settings), settings)
    cache = AnswerCache(cache_settings.path)
    served = CachedTextToSQL(engine, cache, cache_settings)

    outcome: Counter[str] = Counter()
    asked = 0
    started = time.perf_counter()
    for index, (group, question) in enumerate(questions, start=1):
        if cache.has(served.context, canonical_question(question)):
            outcome["already cached"] += 1
            continue
        if args.limit is not None and asked >= args.limit:
            break
        asked += 1
        answer, _ = served.ask(question)
        if cacheable(answer, cache_settings.min_confidence):
            verdict = "cached"
        elif not answer.ok:
            verdict = "not cached: no usable answer"
        elif answer.repairs:
            verdict = "not cached: needed a repair"
        else:
            verdict = "not cached: below the confidence bar"
        outcome[verdict] += 1
        print(
            f"[{index}/{len(questions)}] {verdict:38} {answer.latency_ms / 1000:5.1f}s "
            f"{group:9} {question}",
            flush=True,
        )

    minutes = (time.perf_counter() - started) / 60
    print(
        f"\n{asked} asked in {minutes:.1f} min; cache now holds "
        f"{cache.count(served.context)} answers for this prompt and model"
    )
    for verdict, count in outcome.most_common():
        print(f"  {verdict}: {count}")
    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
