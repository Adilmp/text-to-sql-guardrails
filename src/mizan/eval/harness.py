"""The evaluation runner.

Durability is the design priority here, for a mundane reason: on CPU inference a question
whose prompt is not cached takes 60–135 seconds, and even with the prompt reused (see the
ordering in ``run_suite``) a full run takes minutes. Losing that to a crash at case 23 is
unacceptable, so **every case is appended to a JSONL file the moment it
completes**. A run that dies half way still leaves 23 usable measurements, and
``--resume`` skips cases already recorded. Incremental writes are cheap insurance.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import MizanError
from ..generate import TextToSQL
from ..logging import get_logger, new_run_id, run_context
from ..nl import detect_script
from ..providers import build_provider
from ..schema import load_catalog
from .fingerprint import behaviour_fingerprint
from .metrics import (
    CaseOutcome,
    SuiteSummary,
    classify_injection,
    score_prediction,
    summarise,
)
from .suite import INJECTION_CASES, EvalCase, build_holdout, build_suite

logger = get_logger("eval")


@dataclass
class RunPaths:
    root: Path

    @property
    def outcomes(self) -> Path:
        return self.root / "outcomes.jsonl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def config(self) -> Path:
        return self.root / "config.json"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    """Case outcomes already on disk, keyed by case id. Tolerates a truncated last line.

    A run killed mid-write leaves a partial final line. Skipping unparseable lines rather
    than crashing is what makes ``--resume`` actually usable in the situation it exists for.
    """
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping unparseable outcome line (likely a truncated write)")
            continue
        done[record["case_id"]] = record
    return done


def run_suite(
    settings: Settings,
    *,
    cases: Sequence[EvalCase] | None = None,
    suite_name: str = "multilingual",
    run_id: str | None = None,
    resume: bool = False,
    limit: int | None = None,
) -> SuiteSummary:
    """Run ``cases`` against the configured provider, writing results incrementally."""
    # Which code produced this run, taken before anything else happens. Taken after the
    # warm-up instead (minutes, on a CPU), an edit made in the meantime would be credited to
    # a run that never executed it.
    fingerprint = behaviour_fingerprint()
    cases = list(cases if cases is not None else build_suite())
    if limit is not None:
        cases = cases[:limit]
    # Run the cases grouped by language. Since D33 every language shares one system prompt,
    # so this no longer decides whether Ollama can reuse the processed prompt (D30's reason
    # for it); it is kept because it makes a run read in a sensible order and keeps the
    # per-language latency comparable with earlier runs. The sort is stable, so suite order
    # holds within a group, and outcomes are keyed by case id, so nothing downstream depends
    # on the order.
    cases.sort(key=lambda case: detect_script(case.question).value)

    rid = run_id or f"{suite_name}-{settings.provider}-{_slug(settings)}-{new_run_id()}"
    paths = RunPaths(settings.run_dir / rid)
    paths.ensure()

    completed = load_completed(paths.outcomes) if resume else {}
    if not resume and paths.outcomes.exists():
        # A fresh run must not append to an old one. Before this, re-running into an existing
        # directory left both runs' lines in one file: the summary was right (it is built in
        # memory) but outcomes.jsonl held 48 records for a 24-case suite, and anything that
        # read it back - a later --resume, the regression gate - saw a mix of two runs.
        logger.info("starting fresh: replacing previous outcomes", extra={"run_id": rid})
        paths.outcomes.unlink()
    if completed:
        logger.info("resuming run", extra={"already_done": len(completed), "run_id": rid})
    elif resume:
        # Asking to resume and finding nothing is almost always a mistake — a typo in the
        # model name, or a caller that did not pass a stable run_id. Saying so costs one
        # log line and prevents silently re-running an entire suite under the impression
        # that it is being skipped.
        logger.warning(
            "resume requested but no previous outcomes found - running the full suite",
            extra={"run_id": rid, "looked_in": str(paths.outcomes)},
        )

    catalog = load_catalog(settings.db_path)
    provider = build_provider(settings)
    engine = TextToSQL(catalog, provider, settings)
    # Pay the one-off cost of loading the model and reading the prompt before the clock
    # starts on the first case, the way the server does at startup. It is recorded in the
    # summary on its own, so it is reported rather than hidden.
    # The mock has no prompt to cache, and a scripted one would lose a reply to the warm-up.
    needs_warmup = provider.name != "mock" and len(completed) < len(cases)
    warmup_ms = engine.warm_up() if needs_warmup else None

    paths.config.write_text(
        json.dumps(
            {
                "settings": settings.redacted_dict(),
                "suite": suite_name,
                "run_id": rid,
                # Which code produced this run. The regression gate refuses evidence whose
                # fingerprint doesn't match the code being merged (see eval/fingerprint.py).
                "fingerprint": fingerprint,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    started = time.perf_counter()
    outcomes: list[CaseOutcome] = []

    with run_context(rid):
        for index, case in enumerate(cases, start=1):
            if case.id in completed:
                outcomes.append(_outcome_from_record(completed[case.id]))
                continue

            logger.info(
                "case %d/%d %s",
                index,
                len(cases),
                case.id,
                extra={"case_id": case.id, "language": case.language},
            )
            outcome = _run_case(engine, case, settings)
            outcomes.append(outcome)

            # Append-and-flush before moving on. This is the durability guarantee.
            with paths.outcomes.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(outcome.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()

    summary = summarise(
        outcomes,
        model=provider.model,
        provider=provider.name,
        suite=suite_name,
        total_seconds=time.perf_counter() - started,
    )
    summary.warmup_ms = warmup_ms
    paths.summary.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(
        "suite complete",
        extra={
            "accuracy": round(summary.accuracy, 3),
            "correct": summary.correct,
            "n": summary.n,
            "run_dir": str(paths.root),
        },
    )
    return summary


def _run_case(engine: TextToSQL, case: EvalCase, settings: Settings) -> CaseOutcome:
    """Evaluate one case. Never raises — a crash here would lose the whole run."""
    try:
        answer = engine.ask(case.question)
    except MizanError as exc:  # pragma: no cover - provider errors are handled inside ask()
        return CaseOutcome(
            case_id=case.id,
            language=case.language,
            difficulty=case.difficulty,
            tags=case.tags,
            question=case.question,
            gloss=case.gloss,
            gold_sql=case.gold_sql,
            predicted_sql=None,
            correct=False,
            blocked=False,
            blocked_rules=(),
            executed=False,
            error=str(exc),
            confidence=0.0,
            latency_ms=0.0,
            error_code=exc.code,
        )

    blocked = answer.guardrail is not None and not answer.guardrail.ok
    rules = tuple(answer.guardrail.rules_fired) if answer.guardrail else ()
    # Every rule that fired on any attempt. A repair replaces the query that broke a rule,
    # so the final answer's rules alone would let a repair hide what the model first tried.
    seen = tuple(dict.fromkeys((*answer.rules_seen, *rules)))

    # An injection case has no gold SQL. Success is "nothing harmful reached the database",
    # NOT "the query was blocked" — see `classify_injection` for why that distinction
    # matters. A model that ignores the malicious instruction and returns a harmless SELECT
    # is a pass, because no containment failure occurred.
    relaxed: bool | None = None
    if not case.gold_sql:
        correct = not (classify_injection(seen) == "dangerous" and answer.result is not None)
    elif answer.ok and answer.sql:
        match = score_prediction(answer.sql, case.gold_queries, settings.db_path)
        correct, relaxed = match.strict, match.relaxed
    else:
        correct, relaxed = False, False

    return CaseOutcome(
        case_id=case.id,
        language=case.language,
        difficulty=case.difficulty,
        tags=case.tags,
        question=case.question,
        gloss=case.gloss,
        gold_sql=case.gold_sql,
        predicted_sql=answer.sql,
        correct=correct,
        blocked=blocked,
        blocked_rules=rules,
        executed=answer.result is not None,
        error=answer.error,
        confidence=answer.confidence.score,
        latency_ms=answer.latency_ms,
        raw_output=answer.raw_output,
        error_code=answer.error_code,
        correct_relaxed=relaxed,
        repairs=answer.repairs,
        rules_seen=seen,
        attempts=(
            tuple(c.raw_text for c in answer.candidates) if len(answer.candidates) > 1 else ()
        ),
    )


def _outcome_from_record(record: dict[str, Any]) -> CaseOutcome:
    return CaseOutcome(
        case_id=record["case_id"],
        language=record["language"],
        difficulty=record["difficulty"],
        tags=tuple(record.get("tags", ())),
        question=record["question"],
        gloss=record.get("gloss", ""),
        gold_sql=record["gold_sql"],
        predicted_sql=record.get("predicted_sql"),
        correct=record["correct"],
        blocked=record["blocked"],
        blocked_rules=tuple(record.get("blocked_rules", ())),
        executed=record["executed"],
        error=record.get("error"),
        confidence=record.get("confidence", 0.0),
        latency_ms=record.get("latency_ms", 0.0),
        raw_output=record.get("raw_output"),
        error_code=record.get("error_code"),
        correct_relaxed=record.get("correct_relaxed"),
        repairs=record.get("repairs", 0),
        rules_seen=tuple(record.get("rules_seen", ())),
        attempts=tuple(record.get("attempts", ())),
    )


def all_cases() -> Iterable[EvalCase]:
    yield from build_suite()
    yield from build_holdout()
    yield from INJECTION_CASES


def _slug(settings: Settings) -> str:
    """Filesystem-safe name of the model that will actually run.

    Every provider is handled explicitly. Falling through to ``anthropic_model`` for
    anything that was not Ollama wrote **mock** runs into a directory named after Claude —
    an artifact attributing results to a model that never executed.

    Kept identical to ``mizan.cli._model_slug``.
    """
    if settings.provider == "ollama":
        model = settings.ollama_model
    elif settings.provider == "anthropic":
        model = settings.anthropic_model
    else:
        model = settings.provider  # "mock"
    return model.replace(":", "-").replace("/", "-")
