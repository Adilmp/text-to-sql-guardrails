"""The end-to-end pipeline: question in, validated answer out.

Order of operations, and why it is this order
----------------------------------------------
``detect language → clean → prompt → generate → extract → validate → execute → ground →
(repair → extract → validate → execute → ground)* → score``

Validation sits strictly between generation and execution. That is the whole architecture
in one sentence: **nothing the model produces reaches the database without passing an AST
check against the real catalog first.** Any design where the query is executed and the
result is then inspected has already lost — by then a destructive statement has run.

Repair
------
When the validator rejects a query for a fixable reason (an unknown column, an undefined
alias, a function SQLite doesn't have), SQLite refuses it (an ambiguous column), or it
filters on a text value that isn't in the data but has a close match, the model is shown
its query and the problem and asked again, up to ``max_repairs`` times (``repair.py``).
Queries that tried something dangerous are never repaired. A question that is answered
correctly first time costs exactly what it did before; only broken answers pay for a
second generation.

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
from ..providers.base import Provider, Turn
from ..schema.catalog import Catalog
from ..validate.confidence import Confidence, rejected, score_answer
from ..validate.grounding import GroundingIssue, check_values
from .prompt import build_system_prompt, build_user_prompt
from .repair import Problem, diagnose

logger = get_logger("pipeline")


@dataclass
class Candidate:
    """One sampled generation and everything that happened to it."""

    raw_text: str
    sql: str
    guardrail: GuardrailReport
    result: QueryResult | None = None
    error: str | None = None
    #: Text values the query filters on that aren't in the data (see validate/grounding.py).
    grounding: list[GroundingIssue] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.guardrail.ok and self.result is not None

    @property
    def rank(self) -> int:
        """How good this candidate is, ignoring correctness we can't see: higher is better.

        Ran and grounded > ran with an ungrounded value > failed in SQLite > rejected.
        """
        if not self.usable:
            return 1 if self.guardrail.ok else 0
        return 2 if self.grounding else 3


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
    #: Stable machine-readable code for ``error`` when it came from a ``MizanError`` (for
    #: example ``provider_unavailable``). Lets the eval tell "the model server was down"
    #: from "the model answered wrongly" without parsing messages.
    error_code: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    agreement: float | None = None
    #: The model's reply exactly as received, before extraction. Kept so an answer can be
    #: audited later: what the model *said*, not just what was run.
    raw_output: str | None = None
    #: How many times the model was asked to fix its own query (see ``_repair``).
    repairs: int = 0
    #: Every guardrail rule that fired on any attempt, in order, including attempts that a
    #: repair replaced. Telemetry must not lose what the model first tried.
    rules_seen: tuple[str, ...] = ()

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
            "error_code": self.error_code,
            "agreement": self.agreement,
            "n_candidates": len(self.candidates),
            "raw_output": self.raw_output,
            "repairs": self.repairs,
            "rules_seen": list(self.rules_seen),
        }


class TextToSQL:
    """Orchestrates generation, validation, execution and repair."""

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
        # Built once: it is identical for every question (prompt.py), which is what lets
        # the backend's prompt cache serve every request after the first.
        self.system_prompt = build_system_prompt(catalog)

    def warm_up(self) -> float | None:
        """Make the backend read the system prompt now, so the first user doesn't wait.

        On a CPU the first request pays for loading the model and reading the whole prompt
        (about two minutes for qwen2.5:7b). Everything after it reuses that work. Calling
        this at startup moves the cost off the first user's request. Returns the time it
        took in milliseconds, or ``None`` if the backend failed (logged, never raised: a
        cold first answer is better than a server that won't start).
        """
        started = time.perf_counter()
        try:
            self.provider.generate(
                self.system_prompt,
                build_user_prompt("how many orders are there?"),
                temperature=self.settings.temperature,
                max_tokens=1,
            )
        except ProviderError as exc:
            logger.warning("warm-up failed", extra={"error": str(exc)})
            return None
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("warm-up done", extra={"elapsed_ms": round(elapsed)})
        return elapsed

    def ask(self, question: str) -> Answer:
        """Answer one natural-language question."""
        started = time.perf_counter()
        script = detect_script(question)
        cleaned = clean_for_model(question)

        if not cleaned:
            return self._failed(question, cleaned, script, started, "empty question")

        system = self.system_prompt
        user = build_user_prompt(cleaned)

        logger.info(
            "question received",
            extra={"script": script.value, "provider": self.provider.name, "chars": len(cleaned)},
        )

        try:
            candidates = self._sample(system, user)
        except ProviderError as exc:
            return self._failed(question, cleaned, script, started, str(exc), code=exc.code)

        if not candidates:
            return self._failed(question, cleaned, script, started, "model produced no output")

        chosen, agreement = self._choose(candidates)
        rules_seen = [rule for c in candidates for rule in c.guardrail.rules_fired]
        try:
            chosen, attempts = self._repair(system, user, chosen)
        except ProviderError as exc:
            # The model server failed mid-repair. The first answer still stands; report it
            # rather than throwing away a query that was (at worst) imperfect.
            logger.warning("repair aborted", extra={"error": str(exc)})
            attempts = []
        rules_seen += [rule for c in attempts for rule in c.guardrail.rules_fired]
        repairs = len(attempts)
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
                candidates=candidates + attempts,
                agreement=agreement,
                raw_output=chosen.raw_text,
                repairs=repairs,
                rules_seen=tuple(rules_seen),
            )

        confidence = score_answer(
            schema_grounded=True,
            executed=chosen.result is not None,
            row_count=chosen.result.row_count if chosen.result else 0,
            agreement=agreement,
            rewrite_warnings=len(chosen.guardrail.warnings),
            repairs=repairs,
            ungrounded_values=len(chosen.grounding),
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
            candidates=candidates + attempts,
            agreement=agreement,
            raw_output=chosen.raw_text,
            repairs=repairs,
            rules_seen=tuple(rules_seen),
        )

    # ------------------------------------------------------------------ internals

    def _evaluate(self, raw_text: str) -> Candidate:
        """Extract, validate, execute and ground one model reply."""
        sql = extract_sql(raw_text)
        report = validate(sql, self.catalog, self.policy, raw_output=raw_text)
        candidate = Candidate(raw_text=raw_text, sql=sql, guardrail=report)
        if not report.ok:
            return candidate
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
            return candidate
        candidate.grounding = check_values(report.sql, self.catalog, self.settings.db_path)
        return candidate

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
            candidates.append(self._evaluate(completion.text))
        return candidates

    def _repair(
        self, system: str, user: str, chosen: Candidate
    ) -> tuple[Candidate, list[Candidate]]:
        """Show the model what is wrong with its query, up to ``max_repairs`` times.

        Returns the best candidate seen (by :attr:`Candidate.rank`, the later one on a tie)
        and every repair attempt made. The conversation grows by one exchange per attempt
        and starts with the unchanged system prompt and question, so a cached backend only
        reads the new turns.
        """
        attempts: list[Candidate] = []
        best, current = chosen, chosen
        history: list[Turn] = [("user", user), ("assistant", chosen.raw_text)]
        for _ in range(self.settings.max_repairs):
            problem: Problem | None = diagnose(
                current.guardrail, current.error, current.grounding, self.catalog, current.raw_text
            )
            if problem is None:
                break
            logger.info("repairing", extra={"kind": problem.kind, "attempt": len(attempts) + 1})
            completion = self.provider.generate(
                system,
                problem.feedback,
                history=history,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_output_tokens,
            )
            current = self._evaluate(completion.text)
            attempts.append(current)
            history += [("user", problem.feedback), ("assistant", completion.text)]
            if current.rank >= best.rank:
                best = current
        return best, attempts

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
        self,
        question: str,
        cleaned: str,
        script: Script,
        started: float,
        error: str,
        *,
        code: str | None = None,
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
            error_code=code,
        )
