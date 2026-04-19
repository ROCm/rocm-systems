"""Backend-subcommand dispatcher for `perfxpert-code {claude|codex|gemini}`.

Task 2 delivers a stub registry and help-passthrough / recursion-guard
scaffolding. Tasks 4b/5/10 register real adapters; Task 6 wires the
flag parser. Until then, every backend returns a "not-yet-implemented"
rc=42 so the user sees a clear signal rather than a traceback.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Dict


__all__ = [
    "_exec_backend",
    "is_help_request",
    "RECURSION_GUARD_ENV",
]


# Environment variable that the dispatcher sets in the child env before
# handing control to the backend. Used by the recursion guard (a new
# `perfxpert-code claude` from within an already-running agent session
# refuses to launch).
RECURSION_GUARD_ENV = "PERFXPERT_IN_AGENT_SESSION"


def is_help_request(remaining_argv: list[str]) -> bool:
    """Return True iff the user's argv begins with a help flag.

    Cycle-2 invariant: `perfxpert-code claude --help` MUST skip the
    installer and pass `--help` through to the backend binary (so the
    user discovers native flags without us writing files).
    """
    return bool(remaining_argv) and remaining_argv[0] in ("--help", "-h")


def _stub_backend(name: str) -> Callable[[list[str]], int]:
    """Return a stub handler for a backend whose adapter has not landed yet.

    Task 2 scaffolding: PR 1's final commit (Task 6) wires the real
    adapters into this registry via a monkey-patched dict. Until then
    (or if a user invokes `codex` in PR 1, where Codex ships in PR 2),
    the stub prints an actionable message and returns rc=42 so the
    surrounding tests can assert the stub was reached.
    """

    def _run(remaining_argv: list[str]) -> int:
        sys.stderr.write(
            f"perfxpert-code {name}: adapter not yet implemented in this build.\n"
            f"  The dispatcher routes correctly; the {name}-adapter lands in a\n"
            f"  later task (see docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-plan.md).\n"
        )
        return 42

    return _run


# Registry of backend-name → handler. Tasks 4b / 5 / 10 replace the stub
# entries. Keeping the mapping module-level so tests can monkeypatch the
# registry directly.
BACKEND_REGISTRY: Dict[str, Callable[[list[str]], int]] = {
    "claude": _stub_backend("claude"),
    "codex": _stub_backend("codex"),
    "gemini": _stub_backend("gemini"),
}


def _exec_backend(name: str, remaining_argv: list[str]) -> int:
    """Dispatch to the named backend.

    Responsibilities (Task 2 scope):

    * Recursion guard (R5): refuse if `PERFXPERT_IN_AGENT_SESSION` is
      already set in env — unless `--force` is present in argv.
    * Help passthrough (practical §1.2): `--help` / `-h` short-circuits
      the installer and forwards to the backend binary. In Task 2 this
      means the stub adapter just prints its message; once the real
      adapter lands, it will exec `<backend> --help`.

    The returned int is the exit code surfaced to the caller. On
    `execvpe` strategy (Task 4b+), control normally never returns.
    """
    # Recursion guard — refuse if we are already inside an agent session.
    already = os.environ.get(RECURSION_GUARD_ENV, "").strip()
    if already and "--force" not in remaining_argv:
        sys.stderr.write(
            f"perfxpert-code: already inside a perfxpert-{already} session "
            f"(via {RECURSION_GUARD_ENV}={already!r}). "
            "Pass --force to override.\n"
        )
        return 3

    handler = BACKEND_REGISTRY.get(name)
    if handler is None:
        # Shouldn't happen — route_subcommand gates the name set — but be
        # defensive so a refactor doesn't silently swallow the mismatch.
        sys.stderr.write(
            f"perfxpert-code: no handler registered for backend {name!r}.\n"
        )
        return 2

    return handler(remaining_argv)
