"""Anthropic provider — Claude via official `anthropic` SDK."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import anthropic

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    AuthError,
    DryRunResponse,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from perfxpert.providers.registry import register

_DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
_DEFAULT_MAX_TOKENS = 2048


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for var in ("PERFXPERT_LLM_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    for legacy, canonical in (
        ("ROCINSIGHT_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY"),
        ("ROCPD_LLM_ANTHROPIC_KEY", "PERFXPERT_LLM_ANTHROPIC_KEY"),
    ):
        val = os.environ.get(legacy)
        if val:
            from perfxpert.providers._exceptions import _legacy_env_warn
            _legacy_env_warn(legacy, canonical)
            return val
    raise AuthError(
        "anthropic",
        "no API key (set PERFXPERT_LLM_ANTHROPIC_KEY or ANTHROPIC_API_KEY)",
    )


class AnthropicProvider(Provider):
    """Claude via anthropic SDK."""

    def __init__(self, api_key: Optional[str] = None, **_: Any) -> None:
        key = _resolve_api_key(api_key)
        self._client = anthropic.Anthropic(api_key=key)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        system: str = "",
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        dry_run: bool = False,
    ) -> Union[ProviderResponse, object]:
        if dry_run:
            return DryRunResponse

        model_id = model or _DEFAULT_MODEL
        budget = max_tokens or _DEFAULT_MAX_TOKENS

        try:
            resp = self._client.messages.create(
                model=model_id,
                max_tokens=budget,
                system=system or "You are a helpful assistant.",
                messages=messages,
            )
        except anthropic.AuthenticationError as e:  # pragma: no cover - SDK shape
            raise AuthError("anthropic", str(e)) from e
        except anthropic.RateLimitError as e:
            retry = getattr(e, "retry_after", 0.0) or 0.0
            raise RateLimitError("anthropic", retry_after=retry, message=str(e)) from e
        except anthropic.APITimeoutError as e:
            raise TimeoutError("anthropic", timeout_seconds=0.0, message=str(e)) from e
        except anthropic.APIError as e:
            raise ProviderError(f"[anthropic] {e}") from e

        text = resp.content[0].text if resp.content else ""
        return ProviderResponse(
            content=text,
            provider="anthropic",
            model=getattr(resp, "model", model_id),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )


register(
    "anthropic",
    AnthropicProvider,
    "Anthropic Claude via official SDK (requires PERFXPERT_LLM_ANTHROPIC_KEY or ANTHROPIC_API_KEY)",
)


__all__ = ["AnthropicProvider"]
