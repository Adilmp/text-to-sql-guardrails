"""Structured JSON logging with run-scoped correlation IDs.

Why JSON lines rather than printf logging
-----------------------------------------
Every eval run writes hundreds of records across generation, guardrails and execution. To
answer "which guardrail rejected the most queries on the 0.5b model?" you need to *query*
logs, not read them. One JSON object per line makes that a two-line ``jq`` pipeline.

Why ``contextvars`` rather than passing a run_id parameter everywhere
---------------------------------------------------------------------
The run ID has to appear on log records emitted deep inside the guardrail and provider
layers. Threading a ``run_id`` argument through every function signature would pollute the
API of modules that have no business knowing about runs. A context variable binds it once
per run and is correctly isolated across threads *and* asyncio tasks, which matters because
the FastAPI layer is async.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("mizan_run_id", default=None)

# Attributes present on every LogRecord. Anything *not* in this set was attached by our
# own `extra=` calls and is therefore worth emitting as structured data.
_STANDARD_RECORD_KEYS = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
        "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
        "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
        "message", "asctime",
    }
)


def current_run_id() -> str | None:
    return _run_id.get()


def new_run_id() -> str:
    """Short, sortable, collision-resistant enough for local runs."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:6]}"


@contextmanager
def run_context(run_id: str | None = None) -> Iterator[str]:
    """Bind a run ID for the duration of the block."""
    rid = run_id or new_run_id()
    token = _run_id.set(rid)
    try:
        yield rid
    finally:
        _run_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if (rid := _run_id.get()) is not None:
            payload["run_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = _jsonable(value)

        # default=str keeps a stray Path or datetime from taking down the logger, which
        # must never be the thing that crashes a long eval run.
        return json.dumps(payload, ensure_ascii=False, default=str)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


class HumanFormatter(logging.Formatter):
    """Compact console format. Logs are for machines; the console is for the user."""

    def format(self, record: logging.LogRecord) -> str:
        rid = _run_id.get()
        prefix = f"[{rid}] " if rid else ""
        return f"{record.levelname:<7} {prefix}{record.name}: {record.getMessage()}"


_configured = False


def configure(
    level: str = "INFO",
    log_dir: Path | None = None,
    *,
    console: bool = True,
    force: bool = False,
) -> None:
    """Install handlers on the ``mizan`` logger.

    Idempotent: calling twice will not duplicate handlers (a classic source of every log
    line appearing N times once a CLI and a library both configure logging).
    """
    global _configured
    root = logging.getLogger("mizan")
    if _configured and not force:
        return
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    root.setLevel(level)
    # Do not let records bubble to the root logger, or anything that calls
    # logging.basicConfig() elsewhere would print every line a second time.
    root.propagate = False

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(HumanFormatter())
        stream.setLevel(level)
        root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "mizan.jsonl", encoding="utf-8")
        file_handler.setFormatter(JsonFormatter())
        file_handler.setLevel(logging.DEBUG)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``mizan`` logger, e.g. ``get_logger("guardrails")``."""
    return logging.getLogger(f"mizan.{name}")
