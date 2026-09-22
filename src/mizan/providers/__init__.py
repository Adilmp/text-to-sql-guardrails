"""LLM backends behind one interface."""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import Completion, Provider
from .mock import MockProvider
from .ollama import OllamaProvider

__all__ = [
    "Completion",
    "MockProvider",
    "OllamaProvider",
    "Provider",
    "build_provider",
]


def build_provider(settings: Settings) -> Provider:
    """Construct the provider named by ``settings.provider``.

    ``AnthropicProvider`` is imported here rather than at module scope so that importing
    this package does not require the optional ``anthropic`` dependency.
    """
    if settings.provider == "mock":
        return MockProvider()
    if settings.provider == "ollama":
        return OllamaProvider(
            settings.ollama_model,
            host=settings.ollama_host,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
    if settings.provider == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            settings.anthropic_model,
            timeout_s=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
    raise ConfigError(f"unknown provider {settings.provider!r}")
