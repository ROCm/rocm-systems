"""Opencode-in-opencode recursion guard (spec §5.8, review N8).

When the opencode binary is bundled as a perfxpert provider (see Phase 2
providers/opencode_model.py), a naively-configured session could recurse:
  perfxpert-code (opencode TUI) → MCP → agent with provider=opencode →
  subprocess opencode → MCP → … forever.

Mitigation: whenever we enter an opencode-launched session we set an env
var. Providers check the env var and refuse to subprocess opencode again.
"""

from __future__ import annotations

import contextlib
import os
from contextvars import ContextVar
from typing import Dict, Iterator, Mapping, Optional

_ENV_VAR = "PERFXPERT_IN_OPENCODE_SESSION"
_LOCAL_SESSION_OVERRIDE: ContextVar[bool | None] = ContextVar(
    "perfxpert_in_opencode_session", default=None
)


class RecursionGuardViolation(RuntimeError):
    """Raised when a provider attempts to recursively launch opencode."""


def ensure_not_recursive(provider: str) -> None:
    """Raise if provider would recurse into an already-running opencode session.

    Called from agents/runtime.py at session-construction time, and also
    from the provider adapter for opencode as a second line of defense.
    """
    if provider == "opencode" and in_opencode_session():
        raise RecursionGuardViolation(
            "Cannot use provider='opencode' from within an opencode session "
            "(recursion detected via PERFXPERT_IN_OPENCODE_SESSION). "
            "Choose a different provider (anthropic / openai / ollama / private)."
        )


def in_opencode_session() -> bool:
    """Return whether this context is already running inside opencode."""
    override = _LOCAL_SESSION_OVERRIDE.get()
    if override is not None:
        return override
    return os.environ.get(_ENV_VAR) == "1"


def mark_entry() -> None:
    """Mark the current in-process context as running inside opencode.

    For subprocess or exec inheritance, use subprocess_env() to materialize
    PERFXPERT_IN_OPENCODE_SESSION in the child environment without mutating
    process-global os.environ for sibling threads.
    """
    _LOCAL_SESSION_OVERRIDE.set(True)


def clear() -> None:
    """Remove the breadcrumb (for tests / clean shutdown)."""
    _LOCAL_SESSION_OVERRIDE.set(None)


def subprocess_env(base_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return an environment mapping that marks child processes as in-session."""
    env = dict(base_env or os.environ)
    env[_ENV_VAR] = "1"
    return env


@contextlib.contextmanager
def opencode_session() -> Iterator[None]:
    """Context manager convenience wrapper — sets flag on enter, clears on exit."""
    token = _LOCAL_SESSION_OVERRIDE.set(True)
    try:
        yield
    finally:
        _LOCAL_SESSION_OVERRIDE.reset(token)


__all__ = [
    "RecursionGuardViolation",
    "ensure_not_recursive",
    "in_opencode_session",
    "mark_entry",
    "clear",
    "subprocess_env",
    "opencode_session",
]
