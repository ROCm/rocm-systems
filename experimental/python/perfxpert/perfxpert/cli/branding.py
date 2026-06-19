"""AMD-themed branding + opencode launcher for the perfxpert CLI."""

from __future__ import annotations

import os
from typing import List, Optional

from perfxpert.cli.opencode_launcher import resolve_opencode_binary
from perfxpert.runtime import recursion_guard


_AMD_RED = "#ED1C24"


_ASCII_BANNER = r"""
  ____            __ __  __                _
 |  _ \ ___ _ __ / _|  \/  | ___ _ __   __| |
 | |_) / _ \ '__| |_| |\/| |/ _ \ '__| / _` |
 |  __/  __/ |  |  _| |  | |  __/ |   | (_| |
 |_|   \___|_|  |_| |_|  |_|\___|_|    \__,_|

      PerfXpert — AMD ROCm GPU Performance Expert
      AMD red #ED1C24 · built on rocprofiler-sdk
"""


def get_amd_banner() -> str:
    """Return the ASCII banner for CLI launch."""
    return _ASCII_BANNER


def _provider_configured(name: str) -> bool:
    checks = {
        "anthropic": ("PERFXPERT_LLM_ANTHROPIC_KEY", "ANTHROPIC_API_KEY"),
        "openai": ("PERFXPERT_LLM_OPENAI_KEY", "OPENAI_API_KEY"),
        "ollama": ("PERFXPERT_LLM_LOCAL_URL", "OLLAMA_HOST"),
        "opencode": (),
    }
    if name == "private":
        endpoint = os.environ.get("PERFXPERT_LLM_PRIVATE_URL") or os.environ.get(
            "PRIVATE_LLM_ENDPOINT"
        )
        return bool(endpoint and os.environ.get("PERFXPERT_LLM_PRIVATE_API_KEY"))
    for env in checks.get(name, ()):
        if os.environ.get(env):
            return True
    if name == "ollama":
        return True  # defaults to localhost
    if name == "opencode":
        try:
            resolve_opencode_binary()
            return True
        except FileNotFoundError:
            return False
    return False


def _provider_status(name: str) -> str:
    if name == "opencode":
        return "available" if _provider_configured(name) else "missing"
    return "configured" if _provider_configured(name) else "missing"


def get_provider_status_table() -> str:
    """Return a human-readable provider status table."""
    rows = []
    header = f"{'Provider':<12} {'Status':<12} {'Env hint'}"
    rows.append(header)
    rows.append("-" * len(header))
    hints = {
        "anthropic": "PERFXPERT_LLM_ANTHROPIC_KEY | ANTHROPIC_API_KEY",
        "openai": "PERFXPERT_LLM_OPENAI_KEY | OPENAI_API_KEY",
        "ollama": "PERFXPERT_LLM_LOCAL_URL (defaults to localhost:11434)",
        "private": "PERFXPERT_LLM_PRIVATE_URL/PRIVATE_LLM_ENDPOINT + _API_KEY",
        "opencode": "bundled binary; opencode handles provider auth",
    }
    for name in ("anthropic", "openai", "ollama", "private", "opencode"):
        status = _provider_status(name)
        rows.append(f"{name:<12} {status:<12} {hints[name]}")
    return "\n".join(rows)


def launch_opencode(
    *,
    dry_run: bool = False,
    opencode_path: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
) -> List[str]:
    """Launch opencode with AMD branding + recursion guard env var.

    dry_run=True returns the prospective command without exec'ing.
    """
    binary = opencode_path or str(resolve_opencode_binary())

    argv: List[str] = [binary]
    if extra_args:
        argv.extend(extra_args)

    if dry_run:
        return argv

    env = recursion_guard.subprocess_env(os.environ)
    os.execvpe(binary, argv, env)
    return argv  # pragma: no cover (execvpe does not return)


__all__ = ["get_amd_banner", "get_provider_status_table", "launch_opencode"]
