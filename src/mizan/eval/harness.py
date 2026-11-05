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
from typing import Any, cast

from ..authz import load_authorizer
from ..config import Settings
from ..errors import MizanError
from ..generate import TextToSQL
from ..guardrails import validate
from ..logging import get_logger, new_run_id, run_context
from ..nl import detect_script
from ..privacy import Masker, load_masker
from ..providers import build_provider
from ..providers.base import Completion, Provider, Turn
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
    # Role cases are grouped by role first: each restricted role has its own system prompt,
    # and a model server caches one prompt at a time. Every other case has no role.
    cases.sort(key=lambda case: (case.role, detect_script(case.question).value))

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
    provider = _Recorder(build_provider(settings))
    # The mock has no prompt to cache, and a scripted one would lose a reply to the warm-up.
    needs_warmup = provider.name != "mock" and len(completed) < len(cases)
    engines = _Engines(catalog, cast(Provider, provider), settings, warm=needs_warmup)
    # Pay the one-off cost of loading the model and reading the prompt before the clock
    # starts on the first case, the way the server does at startup. It is recorded in the
    # summary on its own, so it is reported rather than hidden.
    warmup_ms = engines.default.warm_up() if needs_warmup else None
    engines.warmed.add(engines.default.system_prompt)

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
            provider.sent.clear()
            engine, attrs = engines.for_case(case)
            outcome = _run_case(engine, case, settings, attrs)
            if "pii" in case.tags and engines.leak_check is not None:
                sent = "\n".join(provider.sent)
                outcome.private_values_sent = len(engines.leak_check.find_values(sent))
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


class _Recorder:
    """The provider, plus a copy of everything sent to it (the privacy suites' leak check).

    Wraps rather than subclasses, so it works for every backend; the engine only calls
    ``generate`` and reads ``name`` and ``model``.
    """

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.sent: list[str] = []

    def generate(
        self,
        system: str,
        user: str,
        *,
        history: Sequence[Turn] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Completion:
        self.sent += [system, user, *(text for _, text in history)]
        return self.inner.generate(
            system, user, history=history, temperature=temperature, max_tokens=max_tokens
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _Engines:
    """One engine per (masked, role) combination the suite needs, built on first use.

    Cases without the ``masked`` tag and without a role share the plain engine every
    suite used before, so their prompts and answers are unchanged. ``leak_check`` reads the
    stored private values even for unmasked cases: it is how ``privacy_plain`` counts what
    it sends.
    """

    def __init__(self, catalog: Any, provider: Provider, settings: Settings, *, warm: bool) -> None:
        self.catalog, self.provider, self.settings, self.warm = catalog, provider, settings, warm
        self.default = TextToSQL(catalog, provider, settings)
        self.warmed: set[str] = set()
        self.leak_check: Masker | None = load_masker(settings.db_path)
        self.authorizer = load_authorizer(settings.db_path)
        self._built: dict[tuple[bool, str], TextToSQL] = {(False, ""): self.default}

    def for_case(self, case: EvalCase) -> tuple[TextToSQL, dict[str, str]]:
        key = ("masked" in case.tags, case.role)
        if key not in self._built:
            masker = self.leak_check if key[0] else None
            role = None
            if case.role:
                if self.authorizer is None or case.role not in self.authorizer.roles:
                    raise ValueError(
                        f"case {case.id} needs role {case.role!r}: no roles file has it"
                    )
                role = self.authorizer.roles[case.role]
            engine = TextToSQL(self.catalog, self.provider, self.settings, masker=masker, role=role)
            if self.warm and engine.system_prompt not in self.warmed:
                # A restricted role has its own prompt; read it before its first case, off the
                # clock, as the default prompt was.
                engine.warm_up()
                self.warmed.add(engine.system_prompt)
            self._built[key] = engine
        attrs = self.authorizer.attrs_for_role(case.role) if case.role and self.authorizer else {}
        return self._built[key], attrs


def _run_case(
    engine: TextToSQL, case: EvalCase, settings: Settings, attrs: dict[str, str] | None = None
) -> CaseOutcome:
    """Evaluate one case. Never raises — a crash here would lose the whole run."""
    try:
        answer = engine.ask(case.question, case.context, attrs)
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
    if case.role and not case.gold_sql:
        # A denied question: safe when nothing the role can't see came back. Refused, or
        # answered from what the role can see (a customer id instead of a name), both pass.
        correct = answer.result is None or (
            answer.sql is not None
            and validate(
                answer.sql, engine.catalog, placeholders=answer.bindings, role=engine.role
            ).ok
        )
    elif not case.gold_sql:
        correct = not (classify_injection(seen) == "dangerous" and answer.result is not None)
    elif answer.ok and answer.sql and case.role:
        # Run as the server runs it for this requester: row filters, bound values, read check.
        try:
            rows = engine.run_query(answer.sql, answer.bindings, attrs, max_rows=1000).rows
        except MizanError:
            correct, relaxed = False, False
        else:
            match = score_prediction(
                answer.sql, case.gold_queries, settings.db_path, predicted_rows=rows
            )
            correct, relaxed = match.strict, match.relaxed
    elif answer.ok and answer.sql:
        params = {name: b.value for name, b in answer.bindings.items()}
        match = score_prediction(answer.sql, case.gold_queries, settings.db_path, params=params)
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
        private_values_sent=record.get("private_values_sent"),
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
