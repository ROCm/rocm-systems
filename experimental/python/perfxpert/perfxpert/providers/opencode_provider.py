"""opencode provider — subprocess wrapper around the `opencode` CLI.

Recursion guard: if env var PERFXPERT_IN_OPENCODE_SESSION=="1" (set by the
perfxpert-code launcher), complete() raises ProviderError to prevent an
opencode-launched session from spawning another opencode.

Binary resolution order:
    1. constructor kwarg `opencode_path`
    2. env var PERFXPERT_OPENCODE_PATH
    3. bundled patched perfxpert opencode binary
    4. shutil.which("opencode") on PATH
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Union

from perfxpert.providers._base import Provider, ProviderResponse
from perfxpert.providers._exceptions import (
    DryRunResponse,
    ProviderError,
    TimeoutError,
)
from perfxpert.providers.registry import register

_DEFAULT_TIMEOUT = 180.0


def _find_binary(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("PERFXPERT_OPENCODE_PATH")
    if env:
        return env
    try:
        from perfxpert.cli.opencode_launcher import resolve_opencode_binary

        return str(resolve_opencode_binary())
    except FileNotFoundError:
        pass
    found = shutil.which("opencode")
    if found:
        return found
    raise ProviderError(
        "[opencode] binary not found (reinstall perfxpert so the bundled binary is built, "
        "set PERFXPERT_OPENCODE_PATH, or install opencode on PATH)"
    )


def _flatten_messages(messages: List[Dict[str, Any]], system: str) -> str:
    parts: List[str] = []
    if system:
        parts.append(f"[system]\n{system}")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


class OpencodeProvider(Provider):
    """Run a completion via the opencode subprocess CLI."""

    def __init__(
        self,
        opencode_path: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        **_: Any,
    ) -> None:
        self._binary = _find_binary(opencode_path)
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

        if os.environ.get("PERFXPERT_IN_OPENCODE_SESSION") == "1":
            raise ProviderError(
                "[opencode] recursion guard tripped — "
                "cannot invoke opencode provider inside an opencode session"
            )

        prompt = _flatten_messages(messages, system)
        cmd = [self._binary, "run", "--no-color"]
        if model:
            cmd += ["--model", model]

        try:
            completed = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError("opencode", self._timeout, str(e)) from e
        except OSError as e:
            raise ProviderError(f"[opencode] exec failure: {e}") from e

        if completed.returncode != 0:
            raise ProviderError(
                f"[opencode] exit={completed.returncode}: {completed.stderr.strip()[:200]}"
            )

        return ProviderResponse(
            content=completed.stdout,
            provider="opencode",
            model=model or "opencode-default",
            input_tokens=0,
            output_tokens=0,
        )


register(
    "opencode",
    OpencodeProvider,
    "opencode CLI (subprocess; recursion-guarded; PERFXPERT_OPENCODE_PATH or PATH lookup)",
)


__all__ = ["OpencodeProvider"]
