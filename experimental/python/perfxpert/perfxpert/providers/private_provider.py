"""Private provider — OpenAI-compatible HTTP endpoint with custom auth.

Designed for corporate internal gateways exposing an OpenAI-style
`/chat/completions` API (vLLM, TGI, LiteLLM, Azure-compatible shims).
Optional TLS bypass for self-signed CAs (opt-in via env).
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional, Union

import httpx

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    AuthError,
    DryRunResponse,
    ProviderError,
    TimeoutError,
)
from perfxpert.providers._sanitization import redact_paths, sanitize_messages
from perfxpert.providers.registry import register

_DEFAULT_TIMEOUT = 120.0


def _private_url_from_env() -> str:
    return (
        os.environ.get("PERFXPERT_LLM_PRIVATE_URL")
        or os.environ.get("PRIVATE_LLM_ENDPOINT")
        or ""
    ).rstrip("/")


def _parse_headers(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        try:
            obj = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as literal_error:
            raise ValueError(
                "PERFXPERT_LLM_PRIVATE_HEADERS contains invalid JSON "
                f"or Python literal dict: {e}. Value redacted because it may contain secrets."
            ) from literal_error
    if not isinstance(obj, dict):
        raise ValueError(f"PERFXPERT_LLM_PRIVATE_HEADERS must be a JSON object, got {type(obj).__name__}")
    return {str(k): str(v) for k, v in obj.items()}


def _verify_ssl_from_env() -> bool:
    env_flag = os.environ.get("PERFXPERT_LLM_PRIVATE_VERIFY_SSL", "1")
    return env_flag.lower() not in ("0", "false", "no", "off")


class PrivateProvider(Provider):
    """Generic OpenAI-compatible private endpoint."""

    def __init__(
        self,
        url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        verify_ssl: Optional[bool] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> None:
        self._url = (url.rstrip("/") if url else _private_url_from_env())
        if not self._url:
            raise AuthError(
                "private",
                "no endpoint configured (set PERFXPERT_LLM_PRIVATE_URL or PRIVATE_LLM_ENDPOINT)",
            )
        self._model_default = model or os.environ.get("PERFXPERT_LLM_PRIVATE_MODEL") or "default"
        self._api_key = api_key or os.environ.get("PERFXPERT_LLM_PRIVATE_API_KEY") or "dummy"

        merged: Dict[str, str] = {}
        merged.update(_parse_headers(os.environ.get("PERFXPERT_LLM_PRIVATE_HEADERS", "")))
        if extra_headers:
            merged.update(extra_headers)
        self._extra_headers = merged

        if verify_ssl is None:
            verify_ssl = _verify_ssl_from_env()
        self._verify_ssl = bool(verify_ssl)
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

        model_id = model or self._model_default
        messages = sanitize_messages(messages)
        system = redact_paths(system)
        full = [{"role": "system", "content": system}] if system else []
        full.extend(messages)

        payload: Dict[str, Any] = {"model": model_id, "messages": full}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        headers.update(self._extra_headers)

        endpoint = f"{self._url}/chat/completions"

        try:
            resp = httpx.post(
                endpoint,
                json=payload,
                headers=headers,
                verify=self._verify_ssl,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except httpx.TimeoutException as e:
            raise TimeoutError("private", self._timeout, str(e)) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"[private] HTTP failure: {e}") from e

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        usage = data.get("usage") or {}
        return ProviderResponse(
            content=text,
            provider="private",
            model=data.get("model", model_id),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )


register(
    "private",
    PrivateProvider,
    "Private OpenAI-compatible endpoint (PERFXPERT_LLM_PRIVATE_URL or PRIVATE_LLM_ENDPOINT required)",
)


__all__ = ["PrivateProvider", "_parse_headers", "_private_url_from_env", "_verify_ssl_from_env"]
