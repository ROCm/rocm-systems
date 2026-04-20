"""Anthropic provider -- Claude via official `anthropic` SDK (spec N28)."""
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
    import anthropic as _anthropic_sdk
    _SDK = _anthropic_sdk
except ImportError:
    _SDK = None  # type: ignore[assignment]
# Cycle-4 B2: parameterize default model via env. The previous hardcode
# (claude-3-5-sonnet-20241022) is stale; callers can still pin any model
# via the explicit `model=` argument to `.complete()`, via
# PERFXPERT_ANTHROPIC_MODEL (provider-specific), or via PERFXPERT_LLM_MODEL
# (global). Built-in default is the latest widely-available Claude Sonnet.
_BUILTIN_DEFAULT_MODEL = "claude-sonnet-4-5"
_DEFAULT_MAX_TOKENS = 2048


def _resolve_default_model() -> str:
    """Return the default model id for AnthropicProvider at call time.

    Order (matches framework._resolve_model for cross-layer consistency):
      1. PERFXPERT_ANTHROPIC_MODEL
      2. PERFXPERT_LLM_MODEL
      3. Built-in default.
    """
    for var in ("PERFXPERT_ANTHROPIC_MODEL", "PERFXPERT_LLM_MODEL"):
        val = os.environ.get(var)
        if val:
            return val
    return _BUILTIN_DEFAULT_MODEL


# Backwards-compat name — some tests import `_DEFAULT_MODEL` directly.
# Resolve lazily so env changes in-process are picked up.
def __getattr__(name: str) -> Any:  # pragma: no cover - shim
    if name == "_DEFAULT_MODEL":
        return _resolve_default_model()
    raise AttributeError(name)


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for var in ("PERFXPERT_LLM_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    # Pre-rename API-key env var alias. Each fallthrough emits a
    # DeprecationWarning via `_legacy_env_warn`.
    for legacy, canonical in (
        ("ROCPD_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY"),
    ):
        val = os.environ.get(legacy)
        if val:
            from perfxpert.providers._exceptions import _legacy_env_warn
            _legacy_env_warn(legacy, canonical)
            return val
    raise AuthError("anthropic", "no API key (set PERFXPERT_LLM_ANTHROPIC_KEY or ANTHROPIC_API_KEY)")
class AnthropicProvider(Provider):
    """Claude via anthropic SDK."""
    def __init__(self, api_key: Optional[str] = None, **_: Any) -> None:
        require_tool("anthropic", allow_install=False)
        key = _resolve_api_key(api_key)
        self._client = _SDK.Anthropic(api_key=key)  # type: ignore[union-attr]
    def complete(self, messages, *, system="", model=None, max_tokens=None, dry_run=False):
        if dry_run:
            return DryRunResponse
        model_id = model or _resolve_default_model()
        budget = max_tokens or _DEFAULT_MAX_TOKENS
        try:
            resp = self._client.messages.create(
                model=model_id, max_tokens=budget,
                system=system or "You are a helpful assistant.", messages=messages,
            )
        except _SDK.AuthenticationError as e:  # type: ignore[union-attr]  # pragma: no cover
            raise AuthError("anthropic", str(e)) from e
        except _SDK.RateLimitError as e:  # type: ignore[union-attr]
            retry = getattr(e, "retry_after", 0.0) or 0.0
            raise RateLimitError("anthropic", retry_after=retry, message=str(e)) from e
        except _SDK.APITimeoutError as e:  # type: ignore[union-attr]
            raise TimeoutError("anthropic", timeout_seconds=0.0, message=str(e)) from e
        except _SDK.APIError as e:  # type: ignore[union-attr]
            raise ProviderError(f"[anthropic] {e}") from e
        text = resp.content[0].text if resp.content else ""
        return ProviderResponse(
            content=text, provider="anthropic",
            model=getattr(resp, "model", model_id),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
register("anthropic", AnthropicProvider, "Anthropic Claude via official SDK")
# Phase 8: ``claude-code`` is a credential-alias of ``anthropic``. The
# CLI choice is advertised by analyze.py but the actual auth path
# currently uses ANTHROPIC_API_KEY (claude-agent-sdk does not expose
# credential-lookup for the ``claude`` CLI's stored OAuth token). Users
# with a key in ~/.claude/.credentials.json must still set
# ANTHROPIC_API_KEY; see docs/known-issues.md.
register(
    "claude-code",
    AnthropicProvider,
    "Anthropic Claude via claude-code CLI credentials (requires ANTHROPIC_API_KEY)",
)
__all__ = ["AnthropicProvider"]
