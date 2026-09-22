"""Typed configuration, loaded from environment with validated defaults.

Design notes
------------
* Secrets are **never** stored on the config object. ``anthropic_api_key`` is deliberately
  absent: the Anthropic SDK reads ``ANTHROPIC_API_KEY`` from the environment itself, so the
  key never lands in a repr, a log line, or a pickled run artifact. This is the single
  cheapest way to make accidental credential logging structurally impossible.
* Every limit that protects the database (row cap, timeout) lives here rather than being
  hardcoded at the call site, so the guardrail posture of a run is fully described by the
  serialised config written into each run directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .errors import ConfigError

ProviderName = Literal["ollama", "anthropic", "mock"]

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer", value=raw) from exc


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number", value=raw) from exc


class Settings(BaseModel):
    """Runtime configuration for a single Mizan process."""

    model_config = {"frozen": True, "extra": "forbid"}

    # -- provider ----------------------------------------------------------------
    provider: ProviderName = "ollama"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    anthropic_model: str = "claude-sonnet-5"
    request_timeout_s: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)

    # -- generation --------------------------------------------------------------
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=800, gt=0)
    # Number of independent samples used for self-consistency scoring. 1 disables it.
    self_consistency_n: int = Field(default=1, ge=1, le=9)
    # Temperature used for the *extra* self-consistency samples. The first sample always
    # uses `temperature` so that n=1 and n>1 agree on the primary candidate.
    self_consistency_temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    # -- safety limits -----------------------------------------------------------
    max_rows: int = Field(default=200, gt=0, le=10_000)
    query_timeout_s: float = Field(default=5.0, gt=0)
    # Hard ceiling on generated SQL length; a defence against prompt-injection payloads
    # that try to smuggle a very large script through.
    max_sql_chars: int = Field(default=8_000, gt=0)

    # -- paths -------------------------------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "gulf_logistics.sqlite"
    log_dir: Path = PROJECT_ROOT / "logs"
    run_dir: Path = PROJECT_ROOT / "runs"
    data_dir: Path = PROJECT_ROOT / "data"

    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return v

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        """Build settings from ``MIZAN_*`` environment variables, then apply overrides.

        Explicit overrides (i.e. CLI flags) always win over the environment, which in turn
        wins over the defaults declared above.
        """
        provider = _env_str("MIZAN_PROVIDER", "ollama")
        if provider not in ("ollama", "anthropic", "mock"):
            raise ConfigError(
                "MIZAN_PROVIDER must be one of: ollama, anthropic, mock", value=provider
            )

        base: dict[str, Any] = {
            "provider": provider,
            "ollama_host": _env_str("MIZAN_OLLAMA_HOST", "http://127.0.0.1:11434"),
            "ollama_model": _env_str("MIZAN_OLLAMA_MODEL", "qwen2.5:7b"),
            "anthropic_model": _env_str("MIZAN_ANTHROPIC_MODEL", "claude-sonnet-5"),
            "request_timeout_s": _env_float("MIZAN_REQUEST_TIMEOUT_S", 120.0),
            "max_retries": _env_int("MIZAN_MAX_RETRIES", 2),
            "temperature": _env_float("MIZAN_TEMPERATURE", 0.0),
            "max_output_tokens": _env_int("MIZAN_MAX_OUTPUT_TOKENS", 800),
            "self_consistency_n": _env_int("MIZAN_SELF_CONSISTENCY_N", 1),
            "max_rows": _env_int("MIZAN_MAX_ROWS", 200),
            "query_timeout_s": _env_float("MIZAN_QUERY_TIMEOUT_S", 5.0),
            "log_level": _env_str("MIZAN_LOG_LEVEL", "INFO"),
        }
        if db := os.environ.get("MIZAN_DB_PATH"):
            base["db_path"] = Path(db)

        base.update(overrides)
        try:
            return cls(**base)
        except Exception as exc:  # pydantic ValidationError -> our own error type
            raise ConfigError(f"invalid configuration: {exc}") from exc

    def redacted_dict(self) -> dict[str, Any]:
        """Config as plain JSON-able data, safe to write into a run artifact."""
        out: dict[str, Any] = {}
        for key, value in self.model_dump().items():
            out[key] = str(value) if isinstance(value, Path) else value
        return out

    def ensure_dirs(self) -> None:
        for directory in (self.log_dir, self.run_dir, self.data_dir):
            directory.mkdir(parents=True, exist_ok=True)
