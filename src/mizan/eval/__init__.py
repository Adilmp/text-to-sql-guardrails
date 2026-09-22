"""Evaluation: bilingual suite, execution-accuracy metrics, durable runner."""

from __future__ import annotations

from .harness import injection_suite, load_completed, run_suite
from .metrics import CaseOutcome, SuiteSummary, results_match, summarise
from .suite import INJECTION_CASES, EvalCase, build_suite

__all__ = [
    "INJECTION_CASES",
    "CaseOutcome",
    "EvalCase",
    "SuiteSummary",
    "build_suite",
    "injection_suite",
    "load_completed",
    "results_match",
    "run_suite",
    "summarise",
]
