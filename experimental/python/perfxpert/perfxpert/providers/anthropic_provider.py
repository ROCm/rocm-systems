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
_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_MAX_TOKENS = 2048
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
        model_id = model or _DEFAULT_MODEL
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
__all__ = ["AnthropicProvider"]
