"""Provider abstraction over LLM backends.

Why an adapter layer for what looks like one HTTP call
------------------------------------------------------
Three concrete reasons, all of which showed up during this build:

1. **The test suite must run with no network and no model.** ``MockProvider`` makes every
   test except the ones marked ``live`` deterministic and instant. Without it, CI would
   depend on a 4.7 GB model being present.
2. **The eval harness compares backends.** "qwen2.5:0.5b vs 7b vs Claude on the same
   suite" is only expressible if the backend is a parameter, not a hardcoded call.
3. **Failure modes differ per backend and must be normalised.** Ollama returns a 404 with
   an HTML body when the model is not pulled; the Anthropic SDK raises its own exception
   types. Callers should see :class:`ProviderUnavailable` either way, because the retry and
   reporting logic is identical.

Retry policy
------------
Only *transient* failures are retried: timeouts and connection errors. A 400 from a bad
request will fail identically on the third attempt, so retrying it wastes wall-clock time
in an eval run and muddies latency measurements. Backoff is exponential with jitter —
without jitter, a batch of parallel eval requests that fail together retry together, which
is the thundering-herd pattern that turns a blip into an outage.
"""

from __future__ import annotations

import abc
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..errors import ProviderError, ProviderTimeout, ProviderUnavailable
from ..logging import get_logger

logger = get_logger("provider")

#: One earlier turn of a conversation: ``("user" | "assistant", text)``. Used by the repair
#: loop, which shows the model its own query and what was wrong with it.
Turn = tuple[str, str]


def build_messages(history: Sequence[Turn], user: str) -> list[dict[str, str]]:
    """Chat messages for ``history`` followed by the new ``user`` turn."""
    for role, _ in history:
        if role not in ("user", "assistant"):
            raise ValueError(f"history roles must be 'user' or 'assistant', got {role!r}")
    return [{"role": role, "content": text} for role, text in history] + [
        {"role": "user", "content": user}
    ]


@dataclass(frozen=True)
class Completion:
    """One model response, plus everything needed to audit it later."""

    text: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    #: Backend-specific extras, written verbatim into the run log. Never parsed by callers.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "meta": self.meta,
        }


class Provider(abc.ABC):
    """Uniform interface every backend implements."""

    #: Stable short name used in eval reports and run filenames.
    name: str = "base"

    def __init__(self, model: str, *, timeout_s: float = 120.0, max_retries: int = 2) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @abc.abstractmethod
    def _generate_once(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Single attempt. Must raise a :class:`ProviderError` subclass on failure."""

    def generate(
        self,
        system: str,
        user: str,
        *,
        history: Sequence[Turn] = (),
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Completion:
        """Generate with retry on transient failures.

        ``history`` holds earlier turns of the same conversation, oldest first; ``user`` is
        the new message. The system prompt comes first and never changes between turns, so
        a backend with a prompt cache only processes what is new.
        """
        messages = build_messages(history, user)
        last: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = self._generate_once(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
                if attempt:
                    logger.info("recovered after retry", extra={"attempts": attempt + 1})
                return completion
            except (ProviderTimeout, ProviderUnavailable) as exc:
                last = exc
                if attempt == self.max_retries:
                    break
                # Full jitter: sleep uniformly in [0, 2^attempt). Decorrelates retries
                # across concurrent eval workers instead of re-synchronising them.
                delay = random.uniform(0, 2**attempt)
                logger.warning(
                    "transient provider failure, retrying",
                    extra={"attempt": attempt + 1, "delay_s": round(delay, 2), "error": str(exc)},
                )
                time.sleep(delay)
            except ProviderError:
                # Non-transient (bad request, unparseable response): fail immediately.
                raise
        # Not an `assert`: those are stripped under `python -O`, and the retry loop must
        # always terminate in a raised error rather than falling through to `None`.
        if last is None:  # pragma: no cover - loop always sets `last` before breaking
            raise ProviderError("generation failed with no recorded error")
        raise last

    def health(self) -> bool:
        """Whether the backend is reachable and the model is available."""
        try:
            self.generate("You are a health check.", "Reply with OK.", max_tokens=5)
            return True
        except ProviderError as exc:
            logger.warning("health check failed", extra={"provider": self.name, "error": str(exc)})
            return False

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"
