"""Opencode-in-opencode recursion guard (spec §5.8, review N8).

When the opencode binary is bundled as a perfxpert provider (see
providers/opencode_model.py), a naively-configured session could recurse:
  perfxpert-code (opencode TUI) → MCP → agent with provider=opencode →
  subprocess opencode → MCP → … forever.

Mitigation: whenever we enter an opencode-launched session we set an env
var. Providers check the env var and refuse to subprocess opencode again.
"""

from __future__ import annotations

import contextlib
import os
from typing import Iterator

_ENV_VAR = "PERFXPERT_IN_OPENCODE_SESSION"


class RecursionGuardViolation(RuntimeError):
    """Raised when a provider attempts to recursively launch opencode."""


def ensure_not_recursive(provider: str) -> None:
    """Raise if provider would recurse into an already-running opencode session.

    Called from agents/runtime.py at session-construction time, and also
    from the provider adapter for opencode as a second line of defense.
    """
    if provider == "opencode" and os.environ.get(_ENV_VAR) == "1":
        raise RecursionGuardViolation(
            "Cannot use provider='opencode' from within an opencode session "
            "(recursion detected via PERFXPERT_IN_OPENCODE_SESSION). "
            "Choose a different provider (anthropic / openai / ollama / private)."
        )


def mark_entry() -> None:
    """Mark the current process as running inside an opencode session.

    Called by cli/opencode_launcher.py immediately before exec'ing
    the bundled opencode binary.
    """
    os.environ[_ENV_VAR] = "1"


def clear() -> None:
    """Remove the breadcrumb (for tests / clean shutdown)."""
    os.environ.pop(_ENV_VAR, None)


@contextlib.contextmanager
def opencode_session() -> Iterator[None]:
    """Context manager convenience wrapper — sets flag on enter, clears on exit."""
    previous = os.environ.get(_ENV_VAR)
    os.environ[_ENV_VAR] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = previous


__all__ = [
    "RecursionGuardViolation",
    "ensure_not_recursive",
    "mark_entry",
    "clear",
    "opencode_session",
]
