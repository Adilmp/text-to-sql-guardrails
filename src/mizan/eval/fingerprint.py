"""Behaviour fingerprint: a hash of every file that can change an answer or a score.

Why this exists
---------------
The regression gate compares eval runs, and an eval run is only evidence about the code
that produced it. A run committed last week says nothing about a prompt edited today. So the
harness stamps each run's ``config.json`` with this fingerprint, and CI recomputes it from
the code being merged: if they differ, the committed numbers are stale and the build fails
until the eval is re-run.

What counts as behaviour
------------------------
Everything between the question and the verdict: text cleanup, the prompt, the model
providers and their defaults, SQL extraction, the guardrails, execution, confidence, the
eval cases and how they are scored, and the demo database's generator and glossary.
Deliberately **not** included: the API, the CLI, logging and the gate itself. Editing those
cannot change what the model is asked or how an answer is judged, and a fingerprint that
changed on every docstring edit in them would train everyone to ignore it.

Git may check text files out with CRLF line endings on Windows. They are hashed with LF, so
the same commit gives the same fingerprint on every machine.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ..config import PROJECT_ROOT

#: Paths (relative to the repository root) whose contents define behaviour. A directory
#: means every ``.py`` file beneath it.
BEHAVIOUR_PATHS: tuple[str, ...] = (
    "src/mizan/config.py",
    "src/mizan/errors.py",
    "src/mizan/nl",
    "src/mizan/generate",
    "src/mizan/providers",
    "src/mizan/guardrails",
    "src/mizan/schema",
    "src/mizan/validate",
    "src/mizan/db",
    "src/mizan/eval/suite.py",
    "src/mizan/eval/metrics.py",
    "src/mizan/eval/harness.py",
    "data/gulf_logistics.glossary.json",
)

_TEXT_SUFFIXES = frozenset({".py", ".json"})


def behaviour_files(
    root: Path = PROJECT_ROOT, paths: Iterable[str] = BEHAVIOUR_PATHS
) -> list[Path]:
    """Every file the fingerprint covers, in a stable order. Missing paths raise, because a
    fingerprint that silently skipped a renamed module would stop protecting it."""
    files: set[Path] = set()
    for rel in paths:
        target = root / rel
        if target.is_dir():
            files.update(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        elif target.is_file():
            files.add(target)
        else:
            raise FileNotFoundError(f"behaviour path does not exist: {rel}")
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def behaviour_fingerprint(root: Path = PROJECT_ROOT, paths: Iterable[str] = BEHAVIOUR_PATHS) -> str:
    """``sha256:<16 hex>`` over the path and contents of every behaviour file."""
    digest = hashlib.sha256()
    for path in behaviour_files(root, paths):
        data = path.read_bytes()
        if path.suffix in _TEXT_SUFFIXES:
            data = data.replace(b"\r\n", b"\n")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()[:16]
