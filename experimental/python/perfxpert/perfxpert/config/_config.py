"""perfxpert global configuration.

Precedence (highest first):
    1. Environment variables (PERFXPERT_*)
    2. YAML at ~/.config/perfxpert/config.yaml
    3. Built-in defaults

The config model is Pydantic v2, frozen (immutable after creation).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["anthropic", "openai", "ollama", "private", "opencode"]
FenceProfile = Literal["minimal", "standard", "full"]


class PerfXpertConfig(BaseModel):
    """Immutable global settings object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName = Field("anthropic", description="Default LLM provider")
    model: Optional[str] = Field(None, description="Default model id (None = provider default)")
    max_tokens: int = Field(2048, ge=1, le=32768)
    fence_profile: FenceProfile = Field("standard")
    airgap: bool = Field(False, description="When true, skip all LLM calls (tools-only)")
    regression_threshold_pct: float = Field(3.0, ge=0.0, le=100.0)
    hot_kernel_coverage_pct: float = Field(80.0, ge=0.0, le=100.0)


def _coerce(value: str, hint: Any) -> Any:
    if hint is bool or hint == "bool":
        return value.lower() in ("1", "true", "yes", "on")
    if hint is int or hint == "int":
        return int(value)
    if hint is float or hint == "float":
        return float(value)
    return value


_ENV_MAP = {
    "PERFXPERT_PROVIDER": ("provider", str),
    "PERFXPERT_MODEL": ("model", str),
    "PERFXPERT_MAX_TOKENS": ("max_tokens", int),
    "PERFXPERT_FENCE_PROFILE": ("fence_profile", str),
    "PERFXPERT_AIRGAP": ("airgap", bool),
    "PERFXPERT_REGRESSION_THRESHOLD_PCT": ("regression_threshold_pct", float),
    "PERFXPERT_HOT_KERNEL_COVERAGE_PCT": ("hot_kernel_coverage_pct", float),
}


def _config_path() -> Optional[Path]:
    home = os.environ.get("HOME")
    if home:
        return Path(home) / ".config" / "perfxpert" / "config.yaml"
    try:
        return Path.home() / ".config" / "perfxpert" / "config.yaml"
    except RuntimeError:
        # If the home directory cannot be resolved, fall back to built-in defaults.
        return None


def load_config() -> PerfXpertConfig:
    """Build a PerfXpertConfig from env + YAML file + defaults."""
    data: Dict[str, Any] = {}

    # Layer 1: YAML file
    path = _config_path()
    if path is not None and path.exists():
        try:
            parsed = yaml.safe_load(path.read_text()) or {}
            if isinstance(parsed, dict):
                data.update(parsed)
        except yaml.YAMLError:
            pass  # ignore malformed file; defaults prevail

    # Layer 2: env vars (override YAML)
    for env_key, (field, hint) in _ENV_MAP.items():
        raw = os.environ.get(env_key)
        if raw is not None and raw != "":
            data[field] = _coerce(raw, hint)

    return PerfXpertConfig(**data)


__all__ = ["PerfXpertConfig", "load_config"]
