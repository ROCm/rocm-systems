"""OpenAI provider — GPT-series via official `openai` SDK.

Auto-falls-back from `max_completion_tokens` to `max_tokens` if the
deployed endpoint rejects the newer parameter (self-hosted / Azure
OpenAI on older API versions).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import openai

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    AuthError,
    DryRunResponse,
    ProviderError,
    RateLimitError,
    TimeoutError,
)
from perfxpert.providers.registry import register

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_MAX_TOKENS = 2048


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    for var in ("PERFXPERT_LLM_OPENAI_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    for legacy, canonical in (
        ("ROCINSIGHT_LLM_OPENAI_KEY", "PERFXPERT_LLM_OPENAI_KEY"),
        ("ROCPD_LLM_OPENAI_KEY", "PERFXPERT_LLM_OPENAI_KEY"),
    ):
        val = os.environ.get(legacy)
        if val:
            from perfxpert.providers._exceptions import _legacy_env_warn
            _legacy_env_warn(legacy, canonical)
            return val
    raise AuthError(
        "openai",
        "no API key (set PERFXPERT_LLM_OPENAI_KEY or OPENAI_API_KEY)",
    )


class OpenAIProvider(Provider):
    """OpenAI GPT via the official SDK."""

    def __init__(self, api_key: Optional[str] = None, **_: Any) -> None:
        key = _resolve_api_key(api_key)
        self._client = openai.OpenAI(api_key=key)

    def _call(self, *, model: str, system: str, messages: List[Dict[str, Any]], budget: int) -> Any:
        full = [{"role": "system", "content": system}] if system else []
        full.extend(messages)

        # Try new token param first; fall back for older deployments.
        try:
            return self._client.chat.completions.create(
                model=model,
                messages=full,
                max_completion_tokens=budget,
            )
        except openai.BadRequestError as e:
            if "max_completion_tokens" not in str(e) and "unknown" not in str(e).lower():
                raise
            return self._client.chat.completions.create(
                model=model,
                messages=full,
                max_tokens=budget,
            )

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
            resp = self._call(model=model_id, system=system, messages=messages, budget=budget)
        except openai.AuthenticationError as e:
            raise AuthError("openai", str(e)) from e
        except openai.RateLimitError as e:
            retry = getattr(e, "retry_after", 0.0) or 0.0
            raise RateLimitError("openai", retry_after=retry, message=str(e)) from e
        except openai.APITimeoutError as e:
            raise TimeoutError("openai", 0.0, message=str(e)) from e
        except openai.APIError as e:
            raise ProviderError(f"[openai] {e}") from e

        msg = resp.choices[0].message
        return ProviderResponse(
            content=msg.content or "",
            provider="openai",
            model=getattr(resp, "model", model_id),
            input_tokens=getattr(resp.usage, "prompt_tokens", 0),
            output_tokens=getattr(resp.usage, "completion_tokens", 0),
        )


register(
    "openai",
    OpenAIProvider,
    "OpenAI GPT series via official SDK (requires PERFXPERT_LLM_OPENAI_KEY or OPENAI_API_KEY)",
)


__all__ = ["OpenAIProvider"]
