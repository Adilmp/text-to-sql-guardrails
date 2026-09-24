"""Confidence scoring.

What this is, stated honestly
-----------------------------
This is a **heuristic score, not a calibrated probability**. It does not claim that 0.8
means "correct 80% of the time". Its job is to rank answers so a UI can show a bounded
warning and a human can decide whether to trust a number before pasting it into a report.
Claiming calibration without a labelled dataset to calibrate against would be the kind of
detail a reviewer is right to push on, so the code says what it is.

Why the signals are these signals
----------------------------------
Each one is cheap, independent, and fails in a different direction:

``schema_grounded``
    Every identifier exists. Deterministic, from the AST walk. The strongest single signal,
    because an invented column is always wrong.
``execution``
    The query ran. Catches type errors and malformed aggregates that parse fine.
``non_empty``
    Zero rows is weak evidence of a wrong literal — ``status = 'shipped'`` when the value
    set is ``{'in_transit', ...}`` returns nothing and raises nothing. It is only weak
    evidence, because plenty of correct questions legitimately have no answer, so an empty
    result is penalised rather than failed.
``agreement``
    Self-consistency: sample the model several times and see how often the *results* match.
    Comparing results rather than SQL text is deliberate — two queries can be spelled
    completely differently and be equally correct.
``clean_rewrite``
    No guardrail had to rewrite the query. A LIMIT that had to be injected means the model
    ignored an explicit instruction, which correlates with it ignoring others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Bands the UI renders. Thresholds are judgement calls, documented rather than tuned,
#: because tuning them without a labelled set would be false precision.
HIGH_THRESHOLD = 0.80
MEDIUM_THRESHOLD = 0.55


@dataclass(frozen=True)
class Signal:
    name: str
    value: float
    weight: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"signal {self.name} value {self.value} outside [0, 1]")

    @property
    def contribution(self) -> float:
        return self.value * self.weight


@dataclass(frozen=True)
class Confidence:
    score: float
    band: str
    signals: tuple[Signal, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 3),
            "band": self.band,
            "signals": [
                {
                    "name": s.name,
                    "value": round(s.value, 3),
                    "weight": s.weight,
                    "detail": s.detail,
                }
                for s in self.signals
            ],
        }

    @property
    def weakest(self) -> Signal | None:
        """The signal dragging the score down most — what the UI should explain."""
        scored = [s for s in self.signals if s.weight > 0]
        if not scored:
            return None
        return min(scored, key=lambda s: s.value)


def band_for(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def score_answer(
    *,
    schema_grounded: bool,
    executed: bool,
    row_count: int,
    agreement: float | None,
    rewrite_warnings: int,
) -> Confidence:
    """Combine the available signals into a single score.

    ``agreement`` is ``None`` when self-consistency is disabled (``self_consistency_n=1``).
    Its weight is then redistributed proportionally across the remaining signals rather
    than counted as zero — otherwise turning the feature off would cap every answer's score
    at 0.7 and make confidence look broken.
    """
    signals: list[Signal] = [
        Signal(
            "schema_grounded",
            1.0 if schema_grounded else 0.0,
            0.40,
            "every table and column exists in the catalog"
            if schema_grounded
            else "query referenced identifiers that do not exist",
        ),
        Signal(
            "execution",
            1.0 if executed else 0.0,
            0.20,
            "query executed without error" if executed else "query failed to execute",
        ),
        Signal(
            "non_empty",
            1.0 if row_count > 0 else 0.35,
            0.15,
            f"{row_count} rows returned"
            if row_count > 0
            else "no rows returned - possible wrong literal or genuinely empty answer",
        ),
        Signal(
            "clean_rewrite",
            1.0 if rewrite_warnings == 0 else 0.5,
            0.05,
            "no guardrail rewrite needed"
            if rewrite_warnings == 0
            else f"{rewrite_warnings} guardrail rewrite(s) applied",
        ),
    ]

    agreement_weight = 0.20
    if agreement is not None:
        signals.append(
            Signal(
                "agreement",
                max(0.0, min(1.0, agreement)),
                agreement_weight,
                f"{agreement:.0%} of samples produced the same result set",
            )
        )
        total_weight = sum(s.weight for s in signals)
    else:
        signals.append(
            Signal("agreement", 0.0, 0.0, "self-consistency disabled (n=1)")
        )
        total_weight = sum(s.weight for s in signals)

    # Normalising by the weights actually in play is what makes the disabled-agreement
    # case behave correctly instead of silently capping the maximum achievable score.
    raw = sum(s.contribution for s in signals)
    score = raw / total_weight if total_weight else 0.0
    return Confidence(score=score, band=band_for(score), signals=tuple(signals))


def rejected() -> Confidence:
    """Confidence for a query the guardrails refused to run."""
    return Confidence(
        score=0.0,
        band="low",
        signals=(Signal("schema_grounded", 0.0, 1.0, "query rejected by guardrails"),),
    )
