"""The end-to-end pipeline: question in, validated answer out.

Order of operations, and why it is this order
----------------------------------------------
``detect script → clean → prompt → generate → extract → validate → execute → score``

Validation sits strictly between generation and execution. That is the whole architecture
in one sentence: **nothing the model produces reaches the database without passing an AST
check against the real catalog first.** Any design where the query is executed and the
result is then inspected has already lost — by then a destructive statement has run.

Self-consistency
----------------
When ``self_consistency_n > 1`` the model is sampled several times and the candidates are
grouped by the *result set they produce*, not by their SQL text. Two correct queries for
"late orders by courier" may differ in join order, alias names and whether they use a CTE,
yet return identical rows. Grouping on results measures whether the model agrees about the
answer; grouping on text would measure whether it agrees about the phrasing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..errors import ExecutionError, MizanError, ProviderError
from ..guardrails import (
    GuardrailPolicy,
    GuardrailReport,
    QueryResult,
    execute,
    extract_sql,
    validate,
)
from ..logging import get_logger
from ..nl.detect import Script, detect_script
from ..nl.normalize import clean_for_model
from ..providers.base import Provider
from ..schema.catalog import Catalog
from ..validate.confidence import Confidence, rejected, score_answer
from .prompt import build_system_prompt, build_user_prompt

logger = get_logger("pipeline")


@dataclass
class Candidate:
    """One sampled generation and everything that happened to it."""

    raw_text: str
    sql: str
    guardrail: GuardrailReport
    result: QueryResult | None = None
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.guardrail.ok and self.result is not None


@dataclass
class Answer:
    """The pipeline's output. Everything needed to display, audit or score the run."""

    question: str
    question_clean: str
    script: Script
    sql: str | None
    guardrail: GuardrailReport | None
    result: QueryResult | None
    confidence: Confidence
    provider: str
    model: str
    latency_ms: float
    error: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    agreement: float | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.result is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "question_clean": self.question_clean,
            "script": self.script.value,
            "sql": self.sql,
            "guardrail": self.guardrail.to_dict() if self.guardrail else None,
            "result": self.result.to_dict() if self.result else None,
            "confidence": self.confidence.to_dict(),
            "provider": self.provider,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "agreement": self.agreement,
            "n_candidates": len(self.candidates),
        }


class TextToSQL:
    """Orchestrates generation, validation and execution."""

    def __init__(
        self,
        catalog: Catalog,
        provider: Provider,
        settings: Settings,
        policy: GuardrailPolicy | None = None,
    ) -> None:
        self.catalog = catalog
        self.provider = provider
        self.settings = settings
        self.policy = policy or GuardrailPolicy(
            max_rows=settings.max_rows, max_sql_chars=settings.max_sql_chars
        )

    def ask(self, question: str) -> Answer:
        """Answer one natural-language question."""
        started = time.perf_counter()
        script = detect_script(question)
        cleaned = clean_for_model(question)

        if not cleaned:
            return self._failed(question, cleaned, script, started, "empty question")

        system = build_system_prompt(self.catalog, script)
        user = build_user_prompt(cleaned, script)

        logger.info(
            "question received",
            extra={"script": script.value, "provider": self.provider.name, "chars": len(cleaned)},
        )

        try:
            candidates = self._sample(system, user)
        except ProviderError as exc:
            return self._failed(question, cleaned, script, started, str(exc))

        if not candidates:
            return self._failed(question, cleaned, script, started, "model produced no output")

        chosen, agreement = self._choose(candidates)
        latency_ms = (time.perf_counter() - started) * 1000

        if not chosen.guardrail.ok:
            return Answer(
                question=question,
                question_clean=cleaned,
                script=script,
                sql=chosen.guardrail.original_sql or None,
                guardrail=chosen.guardrail,
                result=None,
                confidence=rejected(),
                provider=self.provider.name,
                model=self.provider.model,
                latency_ms=latency_ms,
                error="; ".join(str(v) for v in chosen.guardrail.violations),
                candidates=candidates,
                agreement=agreement,
            )

        confidence = score_answer(
            schema_grounded=True,
            executed=chosen.result is not None,
            row_count=chosen.result.row_count if chosen.result else 0,
            agreement=agreement,
            rewrite_warnings=len(chosen.guardrail.warnings),
        )

        return Answer(
            question=question,
            question_clean=cleaned,
            script=script,
            sql=chosen.guardrail.sql,
            guardrail=chosen.guardrail,
            result=chosen.result,
            confidence=confidence,
            provider=self.provider.name,
            model=self.provider.model,
            latency_ms=latency_ms,
            error=chosen.error,
            candidates=candidates,
            agreement=agreement,
        )

    # ------------------------------------------------------------------ internals

    def _sample(self, system: str, user: str) -> list[Candidate]:
        """Generate, extract, validate and execute each sample."""
        candidates: list[Candidate] = []
        n = self.settings.self_consistency_n
        for i in range(n):
            # Sample 0 always uses the configured (typically 0.0) temperature so that
            # enabling self-consistency never changes the primary candidate.
            temperature = (
                self.settings.temperature if i == 0 else self.settings.self_consistency_temperature
            )
            completion = self.provider.generate(
                system,
                user,
                temperature=temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            sql = extract_sql(completion.text)
            report = validate(sql, self.catalog, self.policy)
            candidate = Candidate(raw_text=completion.text, sql=sql, guardrail=report)

            if report.ok:
                try:
                    candidate.result = execute(
                        report.sql,
                        self.settings.db_path,
                        max_rows=self.settings.max_rows,
                        timeout_s=self.settings.query_timeout_s,
                    )
                except (ExecutionError, MizanError) as exc:
                    candidate.error = str(exc)
                    logger.info("candidate failed to execute", extra={"error": str(exc)})
            candidates.append(candidate)
        return candidates

    def _choose(self, candidates: list[Candidate]) -> tuple[Candidate, float | None]:
        """Pick the candidate to return, and report how much the samples agreed.

        Agreement is only meaningful when more than one sample actually produced a result;
        it is ``None`` otherwise so the confidence scorer can redistribute its weight
        instead of reading a missing measurement as a bad one.
        """
        usable = [c for c in candidates if c.usable]
        if not usable:
            # Nothing executed. Prefer a candidate that at least passed the guardrails, so
            # the caller sees an execution error rather than a validation error.
            passed = next((c for c in candidates if c.guardrail.ok), None)
            return passed or candidates[0], None

        if self.settings.self_consistency_n == 1 or len(usable) == 1:
            return usable[0], None

        groups: dict[frozenset[tuple[Any, ...]], list[Candidate]] = defaultdict(list)
        for candidate in usable:
            # An explicit check rather than `assert`: asserts are stripped under `python -O`,
            # so anything load-bearing must not be one. `usable` already guarantees a result
            # is present, which is why this is a `continue` and not an error.
            if candidate.result is None:  # pragma: no cover - guaranteed by `usable`
                continue
            groups[candidate.result.fingerprint()].append(candidate)

        winner = max(groups.values(), key=len)
        agreement = len(winner) / len(usable)
        logger.debug(
            "self-consistency",
            extra={"groups": len(groups), "usable": len(usable), "agreement": agreement},
        )
        return winner[0], agreement

    def _failed(
        self, question: str, cleaned: str, script: Script, started: float, error: str
    ) -> Answer:
        return Answer(
            question=question,
            question_clean=cleaned,
            script=script,
            sql=None,
            guardrail=None,
            result=None,
            confidence=rejected(),
            provider=self.provider.name,
            model=self.provider.model,
            latency_ms=(time.perf_counter() - started) * 1000,
            error=error,
        )
