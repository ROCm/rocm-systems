"""Ollama provider — local LLMs served by ollama daemon (default localhost:11434)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import httpx

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    DryRunResponse,
    ProviderError,
    TimeoutError,
)
from perfxpert.providers.registry import register

_DEFAULT_URL = "http://localhost:11434"
_DEFAULT_MODEL = "llama3"
_DEFAULT_TIMEOUT = 120.0


def _resolve_url(explicit: Optional[str]) -> str:
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("PERFXPERT_LLM_LOCAL_URL")
    return (env or _DEFAULT_URL).rstrip("/")


class OllamaProvider(Provider):
    """HTTP chat against a local ollama daemon."""

    def __init__(self, url: Optional[str] = None, timeout: float = _DEFAULT_TIMEOUT, **_: Any) -> None:
        self._url = _resolve_url(url)
        self._timeout = timeout

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
        full = [{"role": "system", "content": system}] if system else []
        full.extend(messages)

        payload: Dict[str, Any] = {
            "model": model_id,
            "messages": full,
            "stream": False,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        endpoint = f"{self._url}/api/chat"

        try:
            resp = httpx.post(endpoint, json=payload, timeout=self._timeout)
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError("ollama", self._timeout, str(e)) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"[ollama] HTTP failure: {e}") from e

        data = resp.json()
        text = (data.get("message") or {}).get("content", "")
        return ProviderResponse(
            content=text,
            provider="ollama",
            model=data.get("model", model_id),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )


register(
    "ollama",
    OllamaProvider,
    "Local ollama daemon (default http://localhost:11434; override with PERFXPERT_LLM_LOCAL_URL)",
)


__all__ = ["OllamaProvider"]
