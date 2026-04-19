"""OpenAI provider -- GPT-series via official `openai` SDK (spec N28)."""
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Union
from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    AuthError, DryRunResponse, ProviderError, RateLimitError, TimeoutError,
)
from perfxpert.providers.registry import register
from perfxpert.tools._tooldep import require_tool
try:
    import openai as _openai_sdk
    _SDK = _openai_sdk
except ImportError:
    _SDK = None  # type: ignore[assignment]
# Cycle-4 B2: parameterize default model via env var. See
# anthropic_provider for the rationale; the OpenAI default moves less
# frequently but should stay consistent across both layers.
_BUILTIN_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS = 2048


def _resolve_default_model() -> str:
    """Return the default model id for OpenAIProvider at call time.

    Order:
      1. PERFXPERT_OPENAI_MODEL
      2. PERFXPERT_LLM_MODEL
      3. Built-in default.
    """
    for var in ("PERFXPERT_OPENAI_MODEL", "PERFXPERT_LLM_MODEL"):
        val = os.environ.get(var)
        if val:
            return val
    return _BUILTIN_DEFAULT_MODEL


def __getattr__(name: str) -> Any:  # pragma: no cover - shim for legacy imports
    if name == "_DEFAULT_MODEL":
        return _resolve_default_model()
    raise AttributeError(name)


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for var in ("PERFXPERT_LLM_OPENAI_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    # Pre-rename API-key env var alias. Each fallthrough emits a
    # DeprecationWarning via `_legacy_env_warn`.
    for legacy, canonical in (
        ("ROCPD_LLM_OPENAI_KEY", "PERFXPERT_LLM_OPENAI_KEY"),
    ):
        val = os.environ.get(legacy)
        if val:
            from perfxpert.providers._exceptions import _legacy_env_warn
            _legacy_env_warn(legacy, canonical)
            return val
    raise AuthError("openai", "no API key (set PERFXPERT_LLM_OPENAI_KEY or OPENAI_API_KEY)")
class OpenAIProvider(Provider):
    """OpenAI GPT via the official SDK."""
    def __init__(self, api_key: Optional[str] = None, **_: Any) -> None:
        require_tool("openai", allow_install=False)
        key = _resolve_api_key(api_key)
        self._client = _SDK.OpenAI(api_key=key)  # type: ignore[union-attr]
    def _call(self, *, model, system, messages, budget):
        full = [{"role": "system", "content": system}] if system else []
        full.extend(messages)
        try:
            return self._client.chat.completions.create(model=model, messages=full, max_completion_tokens=budget)
        except _SDK.BadRequestError as e:  # type: ignore[union-attr]
            if "max_completion_tokens" not in str(e) and "unknown" not in str(e).lower():
                raise
            return self._client.chat.completions.create(model=model, messages=full, max_tokens=budget)
    def complete(self, messages, *, system="", model=None, max_tokens=None, dry_run=False):
        if dry_run:
            return DryRunResponse
        model_id = model or _resolve_default_model()
        budget = max_tokens or _DEFAULT_MAX_TOKENS
        try:
            resp = self._call(model=model_id, system=system, messages=messages, budget=budget)
        except _SDK.AuthenticationError as e:  # type: ignore[union-attr]
            raise AuthError("openai", str(e)) from e
        except _SDK.RateLimitError as e:  # type: ignore[union-attr]
            retry = getattr(e, "retry_after", 0.0) or 0.0
            raise RateLimitError("openai", retry_after=retry, message=str(e)) from e
        except _SDK.APITimeoutError as e:  # type: ignore[union-attr]
            raise TimeoutError("openai", 0.0, message=str(e)) from e
        except _SDK.APIError as e:  # type: ignore[union-attr]
            raise ProviderError(f"[openai] {e}") from e
        msg = resp.choices[0].message
        return ProviderResponse(
            content=msg.content or "", provider="openai",
            model=getattr(resp, "model", model_id),
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )
register("openai", OpenAIProvider, "OpenAI GPT series via official SDK")
__all__ = ["OpenAIProvider"]
