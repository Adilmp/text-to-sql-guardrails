"""Evaluation: multilingual and held-out suites, execution-accuracy metrics, durable runner."""

from __future__ import annotations

from .harness import load_completed, run_suite
from .metrics import CaseOutcome, Match, SuiteSummary, results_match, score_prediction, summarise
from .suite import (
    INJECTION_CASES,
    SUITES,
    EvalCase,
    build_holdout,
    build_suite,
    get_suite,
    injection_suite,
)

__all__ = [
    "INJECTION_CASES",
    "SUITES",
    "CaseOutcome",
    "EvalCase",
    "Match",
    "SuiteSummary",
    "build_holdout",
    "build_suite",
    "get_suite",
    "injection_suite",
    "load_completed",
    "results_match",
    "run_suite",
    "score_prediction",
    "summarise",
]
