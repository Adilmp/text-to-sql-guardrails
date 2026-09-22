"""Ollama backend, spoken to over its HTTP API directly.

No SDK, deliberately: the surface we need is one POST to ``/api/chat``, and a hand-rolled
client makes the error taxonomy explicit rather than hidden behind someone else's
exception hierarchy. It also keeps ``httpx`` as the only transport dependency.
"""

from __future__ import annotations

import time

import httpx

from ..errors import ProviderResponseError, ProviderTimeout, ProviderUnavailable
from ..logging import get_logger
from .base import Completion, Provider

logger = get_logger("provider.ollama")


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        *,
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model, timeout_s=timeout_s, max_retries=max_retries)
        self.host = host.rstrip("/")

    def _generate_once(
        self, system: str, user: str, *, temperature: float, max_tokens: int
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # Ollama's default context is small enough to silently truncate a schema
                # card plus few-shot examples, which shows up as the model inventing
                # tables it was actually told about. Raising it removes a whole class of
                # phantom "hallucination".
                "num_ctx": 8192,
            },
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                f"ollama timed out after {self.timeout_s}s", model=self.model
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"cannot reach ollama at {self.host}: {exc}", host=self.host
            ) from exc

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code == 404:
            # Ollama returns 404 both for a missing route and a missing model. The body
            # distinguishes them, and telling the user which one it is saves real time.
            raise ProviderUnavailable(
                f"model {self.model!r} is not pulled (run: ollama pull {self.model})",
                model=self.model,
                body=response.text[:200],
            )
        if response.status_code >= 400:
            raise ProviderResponseError(
                f"ollama returned HTTP {response.status_code}",
                status=response.status_code,
                body=response.text[:300],
            )

        try:
            data = response.json()
            text = data["message"]["content"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ProviderResponseError(
                f"unexpected ollama response shape: {exc}", body=response.text[:300]
            ) from exc

        return Completion(
            text=text,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            meta={
                "done_reason": data.get("done_reason"),
                "total_duration_ns": data.get("total_duration"),
            },
        )

    def health(self) -> bool:
        """Check the model is present without paying for a generation."""
        try:
            response = httpx.get(f"{self.host}/api/tags", timeout=10.0)
            response.raise_for_status()
            models = {m["name"] for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("ollama unreachable", extra={"host": self.host, "error": str(exc)})
            return False
        if self.model not in models:
            logger.warning(
                "model not pulled", extra={"model": self.model, "available": sorted(models)}
            )
            return False
        return True
