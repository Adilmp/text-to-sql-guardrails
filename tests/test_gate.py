"""The regression gate (`mizan.eval.gate`), its fingerprint, and the script's exit codes.

Everything here is offline. The model-level behaviour is exercised with the mock provider,
and with an Ollama provider pointed at a closed port, which replays a real incident: the
first attempt at measuring run-to-run noise lost its model server mid-run, and a naive
comparison read that as accuracy falling from 19/24 to 2/24.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mizan.config import Settings
from mizan.eval import build_suite, run_suite
from mizan.eval.fingerprint import behaviour_fingerprint
from mizan.eval.gate import (
    GateInputError,
    Policy,
    Run,
    Verdict,
    check_fresh,
    compare,
    is_infra_error,
    load_run,
    mcnemar_exact,
    render_markdown,
    worst,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def outcome(case_id: str, correct: bool = True, **extra: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "case_id": case_id,
        "question": f"question {case_id}",
        "gloss": "",
        "predicted_sql": f"SELECT {case_id}",
        "correct": correct,
        "blocked": False,
        "blocked_rules": [],
        "executed": True,
        "error": None,
        "error_code": None,
    }
    record.update(extra)
    return record


def make_run(*records: dict[str, Any], suite: str = "bilingual", label: str = "run") -> Run:
    return Run(label, suite, "qwen2.5:7b", None, {r["case_id"]: r for r in records})


BASE = make_run(outcome("a"), outcome("b"), outcome("c", correct=False), label="baseline")


class TestVerdicts:
    def test_identical_runs_pass(self) -> None:
        report = compare(BASE, BASE)
        assert report.verdict is Verdict.PASS
        assert (report.baseline_correct, report.candidate_correct) == (2, 2)

    def test_one_case_going_wrong_fails_and_is_named(self) -> None:
        candidate = make_run(outcome("a"), outcome("b", correct=False), outcome("c", False))
        report = compare(BASE, candidate)
        assert report.verdict is Verdict.FAIL
        assert [ch.case_id for ch in report.regressed] == ["b"]
        assert "`b`" in render_markdown(report)

    def test_same_accuracy_can_hide_a_regression(self) -> None:
        """79% before and 79% after, but a different question broke: the gate must see it."""
        candidate = make_run(outcome("a"), outcome("b", correct=False), outcome("c"))
        report = compare(BASE, candidate)
        assert report.baseline_correct == report.candidate_correct
        assert report.verdict is Verdict.FAIL
        assert [ch.case_id for ch in report.fixed] == ["c"]

    def test_policy_tolerance_and_accepted_regressions(self) -> None:
        candidate = make_run(
            outcome("a", correct=False), outcome("b", correct=False), outcome("c", False)
        )
        assert compare(BASE, candidate, Policy(max_regressions=2)).verdict is Verdict.PASS
        policy = Policy(accepted={"a": "gold answer was wrong", "zzz": "stale"})
        report = compare(BASE, candidate, policy)
        assert report.verdict is Verdict.FAIL  # "b" is still unaccepted
        assert [ch.case_id for ch, _ in report.accepted] == ["a"]
        assert report.unused_acceptances == ["zzz"]

    def test_a_leaked_injection_fails_even_when_accuracy_improves(self) -> None:
        base = make_run(outcome("inj1"), outcome("inj2", correct=False), suite="injection")
        candidate = make_run(
            outcome("inj1", correct=False, blocked_rules=["write_operation"]),
            outcome("inj2"),
            suite="injection",
        )
        report = compare(base, candidate, Policy(max_regressions=5))
        assert report.verdict is Verdict.FAIL
        assert [case for case, _ in report.safety] == ["inj1"]

    def test_executed_dangerous_statement_fails_in_any_suite(self) -> None:
        candidate = make_run(
            outcome("a", blocked_rules=["stacked_statements"]), outcome("b"), outcome("c", False)
        )
        assert compare(BASE, candidate).verdict is Verdict.FAIL

    def test_missing_cases_and_mismatched_suites_are_inconclusive(self) -> None:
        assert compare(BASE, make_run(outcome("a"), outcome("b"))).verdict is Verdict.INCONCLUSIVE
        other = make_run(outcome("a"), outcome("b"), outcome("c", False), suite="injection")
        assert compare(BASE, other).verdict is Verdict.INCONCLUSIVE

    def test_new_cases_are_reported_not_inconclusive(self) -> None:
        """Adding questions to the suite must be able to pass CI. It used to be impossible:
        a case with no baseline counted as missing evidence."""
        candidate = make_run(outcome("a"), outcome("b"), outcome("c", False), outcome("d"))
        report = compare(BASE, candidate)
        assert report.verdict is Verdict.PASS
        assert report.added == [("d", True)]
        assert "New cases (no baseline yet): 1, of which 1 correct" in render_markdown(report)

    def test_a_new_case_is_still_safety_checked(self) -> None:
        leaked = outcome("d", blocked_rules=["write_operation"], executed=True)
        report = compare(BASE, make_run(outcome("a"), outcome("b"), outcome("c", False), leaked))
        assert report.verdict is Verdict.FAIL
        assert [case for case, _ in report.safety] == ["d"]

    def test_a_declared_suite_rename_is_comparable(self) -> None:
        renamed = make_run(outcome("a"), outcome("b"), outcome("c", False), suite="multilingual")
        assert compare(BASE, renamed).verdict is Verdict.INCONCLUSIVE  # undeclared
        policy = Policy(renamed_suites={"multilingual": "bilingual"})
        assert compare(BASE, renamed, policy).verdict is Verdict.PASS

    def test_worst_verdict_wins(self) -> None:
        assert worst([Verdict.PASS, Verdict.INCONCLUSIVE]) is Verdict.INCONCLUSIVE
        assert worst([Verdict.INCONCLUSIVE, Verdict.FAIL]) is Verdict.FAIL
        assert worst([]) is Verdict.PASS
        assert [v.exit_code for v in Verdict] == [0, 1, 2]


class TestInfrastructureErrors:
    DOWN: ClassVar[dict[str, Any]] = {
        "correct": False,
        "predicted_sql": None,
        "executed": False,
        "error": "cannot reach ollama at http://127.0.0.1:11434: [Errno 111] Connection refused",
    }

    def test_a_dead_server_is_inconclusive_not_a_regression(self) -> None:
        candidate = make_run(
            outcome("a", **self.DOWN, error_code="provider_unavailable"),
            outcome("b", **self.DOWN, error_code="provider_timeout"),
            outcome("c", correct=False),
        )
        report = compare(BASE, candidate)
        assert report.verdict is Verdict.INCONCLUSIVE
        assert report.regressed == []
        assert report.compared == 1

    def test_a_broken_run_can_hide_a_regression_but_not_excuse_one(self) -> None:
        candidate = make_run(
            outcome("a", **self.DOWN, error_code="provider_unavailable"),
            outcome("b", correct=False),
            outcome("c", correct=False),
        )
        assert compare(BASE, candidate).verdict is Verdict.FAIL

    def test_old_records_without_a_code_are_recognised_by_message(self) -> None:
        assert is_infra_error(self.DOWN)
        assert is_infra_error({"error": "ollama timed out after 120.0s", "predicted_sql": None})

    def test_the_models_own_slow_query_is_not_infrastructure(self) -> None:
        slow = {"error": "query exceeded 5.0s (timeout_s=5.0)", "predicted_sql": None}
        assert not is_infra_error(slow)
        assert not is_infra_error({**self.DOWN, "predicted_sql": "SELECT 'cannot reach'"})
        assert not is_infra_error({"error": "x", "error_code": "unknown_column"})


class TestMcNemar:
    @pytest.mark.parametrize(
        ("regressed", "fixed", "p"),
        [(0, 0, 1.0), (5, 0, 0.0625), (3, 1, 0.625), (1, 1, 1.0), (10, 0, 2 / 1024)],
    )
    def test_exact_values(self, regressed: int, fixed: int, p: float) -> None:
        assert mcnemar_exact(regressed, fixed) == pytest.approx(p)


class TestPolicy:
    def write(self, tmp_path: Path, data: dict[str, Any]) -> Path:
        path = tmp_path / "policy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_renames_load(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        path.write_text(
            json.dumps({"renamed": {"runs": {"new-x": "old-x"}, "suites": {"new": "old"}}}),
            encoding="utf-8",
        )
        policy = Policy.load(path)
        assert policy.renamed_runs == {"new-x": "old-x"}
        assert policy.same_suite("old", "new") and not policy.same_suite("new", "old")

    def test_repo_policy_loads(self) -> None:
        policy = Policy.load(REPO_ROOT / "regression-gate.json")
        assert "multilingual-qwen2.5-7b" in policy.gated_runs
        assert policy.renamed_runs["multilingual-qwen2.5-7b"] == "bilingual-qwen2.5-7b"

    def test_an_acceptance_needs_a_reason(self, tmp_path: Path) -> None:
        with pytest.raises(GateInputError):
            Policy.load(self.write(tmp_path, {"accepted": [{"case_id": "a", "reason": " "}]}))

    def test_tolerance_must_be_a_non_negative_integer(self, tmp_path: Path) -> None:
        for bad in (-1, 1.5, "2", True):
            with pytest.raises(GateInputError):
                Policy.load(self.write(tmp_path, {"max_regressions": bad}))


class TestFingerprint:
    def make_tree(self, root: Path) -> list[str]:
        (root / "pkg").mkdir()
        (root / "pkg" / "prompt.py").write_bytes(b"PROMPT = 'x'\n")
        (root / "pkg" / "__pycache__").mkdir()
        (root / "pkg" / "__pycache__" / "prompt.pyc").write_bytes(b"junk")
        (root / "glossary.json").write_bytes(b"{}\n")
        (root / "api.py").write_bytes(b"# not behaviour\n")
        return ["pkg", "glossary.json"]

    def test_changes_with_behaviour_and_nothing_else(self, tmp_path: Path) -> None:
        paths = self.make_tree(tmp_path)
        first = behaviour_fingerprint(tmp_path, paths)
        (tmp_path / "api.py").write_bytes(b"# edited\n")
        (tmp_path / "pkg" / "__pycache__" / "prompt.pyc").write_bytes(b"other junk")
        assert behaviour_fingerprint(tmp_path, paths) == first
        (tmp_path / "pkg" / "prompt.py").write_bytes(b"PROMPT = 'y'\n")
        assert behaviour_fingerprint(tmp_path, paths) != first

    def test_line_endings_do_not_matter(self, tmp_path: Path) -> None:
        paths = self.make_tree(tmp_path)
        first = behaviour_fingerprint(tmp_path, paths)
        (tmp_path / "pkg" / "prompt.py").write_bytes(b"PROMPT = 'x'\r\n")
        assert behaviour_fingerprint(tmp_path, paths) == first

    def test_a_missing_path_is_an_error_not_a_silent_skip(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            behaviour_fingerprint(tmp_path, ["src/renamed_module"])

    def test_repo_fingerprint_is_well_formed(self) -> None:
        fp = behaviour_fingerprint()
        assert fp.startswith("sha256:") and len(fp) == len("sha256:") + 16

    def test_check_fresh(self) -> None:
        run = make_run(outcome("a"))
        assert "no behaviour fingerprint" in (check_fresh(run, "sha256:1") or "")
        stamped = Run("r", "bilingual", "m", "sha256:1", run.outcomes)
        assert check_fresh(stamped, "sha256:1") is None
        assert "re-run the eval" in (check_fresh(stamped, "sha256:2") or "")


class TestLoadingRuns:
    def write_run(
        self, run_dir: Path, records: list[dict[str, Any]], suite: str = "bilingual"
    ) -> None:
        run_dir.mkdir(parents=True)
        config = {"settings": {"provider": "ollama", "ollama_model": "m"}, "suite": suite}
        (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        lines = "".join(json.dumps(r) + "\n" for r in records)
        (run_dir / "outcomes.jsonl").write_text(lines, encoding="utf-8")

    def test_duplicate_records_are_refused(self, tmp_path: Path) -> None:
        self.write_run(tmp_path / "r", [outcome("a"), outcome("a", correct=False)])
        with pytest.raises(GateInputError, match="twice"):
            load_run(tmp_path / "r")

    def test_baseline_can_be_read_from_a_git_ref(self, tmp_path: Path) -> None:
        git = shutil.which("git")
        if git is None:
            pytest.skip("git not installed")
        repo = tmp_path / "repo"
        self.write_run(repo / "runs" / "r", [outcome("a"), outcome("b")])

        def run_git(*args: str) -> None:
            subprocess.run(  # noqa: S603 - test helper, fixed arguments
                [git, "-C", str(repo), *args], check=True, capture_output=True
            )

        run_git("init", "-q")
        run_git("add", ".")
        run_git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "baseline")
        (repo / "runs" / "r" / "outcomes.jsonl").write_text(
            json.dumps(outcome("a")) + "\n" + json.dumps(outcome("b", correct=False)) + "\n",
            encoding="utf-8",
        )
        report = compare(
            load_run(repo / "runs" / "r", ref="HEAD", repo=repo), load_run(repo / "runs" / "r")
        )
        assert report.verdict is Verdict.FAIL
        assert report.baseline.label.startswith("HEAD:runs/r")
        with pytest.raises(GateInputError):
            load_run(repo / "runs" / "r", ref="--output=/tmp/x", repo=repo)
        with pytest.raises(GateInputError):
            load_run(repo / "runs" / "r", ref="no-such-branch", repo=repo)


class TestWithTheRealHarness:
    def test_mock_runs_record_fingerprint_and_pass_against_themselves(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        cfg = Settings.from_env(provider="mock", db_path=db_path, run_dir=tmp_path)
        cases = build_suite()[:4]
        run_suite(cfg, cases=cases, suite_name="bilingual", run_id="one")
        run_suite(cfg, cases=cases, suite_name="bilingual", run_id="two")
        one, two = load_run(tmp_path / "one"), load_run(tmp_path / "two")
        assert one.fingerprint == behaviour_fingerprint()
        assert compare(one, two).verdict is Verdict.PASS

    def test_rerunning_into_the_same_directory_replaces_instead_of_appending(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        cfg = Settings.from_env(provider="mock", db_path=db_path, run_dir=tmp_path)
        cases = build_suite()[:3]
        run_suite(cfg, cases=cases, suite_name="bilingual", run_id="same")
        run_suite(cfg, cases=cases, suite_name="bilingual", run_id="same")
        assert (tmp_path / "same" / "outcomes.jsonl").read_text().count("\n") == 3

    def test_unreachable_model_server_is_recorded_as_such_and_gated_inconclusive(
        self, db_path: Path, tmp_path: Path
    ) -> None:
        """Replays the incident: the model server disappears between two runs."""
        cases = build_suite()[:3]
        good = Settings.from_env(provider="mock", db_path=db_path, run_dir=tmp_path)
        run_suite(good, cases=cases, suite_name="bilingual", run_id="before")
        down = Settings.from_env(
            provider="ollama",
            ollama_host="http://127.0.0.1:9",  # the discard port: nothing listens there
            db_path=db_path,
            run_dir=tmp_path,
            max_retries=0,
            request_timeout_s=2.0,
        )
        run_suite(down, cases=cases, suite_name="bilingual", run_id="after")
        after = load_run(tmp_path / "after")
        assert {r["error_code"] for r in after.outcomes.values()} == {"provider_unavailable"}
        report = compare(load_run(tmp_path / "before"), after)
        assert report.verdict is Verdict.INCONCLUSIVE
        assert report.regressed == []


class TestScript:
    def gate(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - this interpreter running the repo's own script
            [sys.executable, str(REPO_ROOT / "scripts" / "regression_gate.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_exit_codes(self, tmp_path: Path) -> None:
        loader = TestLoadingRuns()
        loader.write_run(tmp_path / "base", [outcome("a"), outcome("b")])
        loader.write_run(tmp_path / "same", [outcome("a"), outcome("b")])
        loader.write_run(tmp_path / "worse", [outcome("a"), outcome("b", correct=False)])
        report = tmp_path / "report.md"
        passed = self.gate("compare", str(tmp_path / "base"), str(tmp_path / "same"))
        failed = self.gate(
            "compare", str(tmp_path / "base"), str(tmp_path / "worse"), "--report", str(report)
        )
        broken = self.gate("compare", str(tmp_path / "base"), str(tmp_path / "missing"))
        assert (passed.returncode, failed.returncode, broken.returncode) == (0, 1, 2)
        assert "`b`" in report.read_text(encoding="utf-8")


def test_cases_run_grouped_by_language_in_suite_order(db_path: Path, tmp_path: Path) -> None:
    """The suite interleaves en/ar/ur; the runner groups them, keeping suite order within each."""
    cases = build_suite()[:6]
    assert [c.language for c in cases] == ["en", "ar", "ur"] * 2
    cfg = Settings.from_env(provider="mock", db_path=db_path, run_dir=tmp_path)
    summary = run_suite(cfg, cases=cases, suite_name="multilingual", run_id="order")
    lines = (tmp_path / "order" / "outcomes.jsonl").read_text(encoding="utf-8").splitlines()
    ran = [json.loads(line)["case_id"] for line in lines]
    assert [case_id[:2] for case_id in ran] == ["ar"] * 2 + ["en"] * 2 + ["ur"] * 2
    assert ran[2:4] == [c.id for c in cases if c.language == "en"]  # suite order kept
    assert summary.n == 6
