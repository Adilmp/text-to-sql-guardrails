"""Anthropic backend.

The SDK is imported lazily inside ``__init__`` rather than at module scope so that
``pip install mizan`` without the ``[anthropic]`` extra still imports cleanly, and the test
suite never needs the package. The cost of the deferred import is one dictionary lookup per
process; the benefit is that an optional dependency stays genuinely optional.

The API key is never read into an attribute. The SDK picks ``ANTHROPIC_API_KEY`` up from
the environment itself, which means the key cannot leak through a ``repr``, a pickled
config, or a log line that dumps ``self.__dict__``.
"""

from __future__ import annotations

import time
from typing import Any

from ..errors import ProviderResponseError, ProviderTimeout, ProviderUnavailable
from ..logging import get_logger
from .base import Completion, Provider

logger = get_logger("provider.anthropic")


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, timeout_s=timeout_s, max_retries=max_retries)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ProviderUnavailable(
                "the anthropic extra is not installed (pip install 'mizan[anthropic]')"
            ) from exc

        self._sdk = anthropic
        # max_retries=0: retry policy lives in Provider.generate so that every backend
        # retries identically and the eval harness sees one consistent latency story.
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)

    def _generate_once(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> Completion:
        started = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except self._sdk.APITimeoutError as exc:
            raise ProviderTimeout(
                f"anthropic timed out after {self.timeout_s}s", model=self.model
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderUnavailable(f"cannot reach the Anthropic API: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            # 429 and 5xx are transient and worth a retry; 4xx will fail identically
            # forever, so it is raised as a non-retryable ProviderResponseError.
            status = getattr(exc, "status_code", 0)
            if status == 429 or status >= 500:
                raise ProviderUnavailable(
                    f"anthropic returned {status}", status=status
                ) from exc
            raise ProviderResponseError(
                f"anthropic returned {status}", status=status, body=str(exc)[:300]
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        text = _first_text_block(response)
        usage: Any = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "input_tokens", None),
            completion_tokens=getattr(usage, "output_tokens", None),
            meta={"stop_reason": getattr(response, "stop_reason", None)},
        )


def _first_text_block(response: Any) -> str:
    """Extract text from a Messages response.

    ``content`` is a list of typed blocks, not a string. Indexing ``[0].text`` blindly
    breaks the moment a response leads with a non-text block, so this scans for the first
    block that actually carries text.
    """
    blocks = getattr(response, "content", None) or []
    for block in blocks:
        if getattr(block, "type", None) == "text":
            return str(block.text)
    raise ProviderResponseError(
        "anthropic response contained no text block",
        block_types=[getattr(b, "type", "?") for b in blocks],
    )
