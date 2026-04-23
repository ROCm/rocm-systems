"""Backend-subcommand dispatcher for `perfxpert-code {claude|codex|gemini}`.

Task 2 delivered the stub registry + help-passthrough / recursion-guard.
Task 6 wires the real adapters into the registry, parses dispatcher-
owned flags (`--dry-run`, `--quiet`, `--force`,
`--allow-agents-md-append`), runs the install-then-spawn flow, and
sets the recursion-guard env for child processes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List


__all__ = [
    "_exec_backend",
    "is_help_request",
    "RECURSION_GUARD_ENV",
    "parse_dispatcher_flags",
    "BACKEND_REGISTRY",
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


# Registry of backend-name → handler. Tests may monkeypatch the dict.
# Codex adapter was wired in PR 2 (Task 10).
def _claude_runner(argv: list[str]) -> int:
    from perfxpert.cli._backend.claude import ClaudeCodeAdapter

    return _run_adapter(ClaudeCodeAdapter(), argv)


def _gemini_runner(argv: list[str]) -> int:
    from perfxpert.cli._backend.gemini import GeminiAdapter

    return _run_adapter(GeminiAdapter(), argv)


def _codex_runner(argv: list[str]) -> int:
    from perfxpert.cli._backend.codex import CodexAdapter

    return _run_adapter(CodexAdapter(), argv)


BACKEND_REGISTRY: Dict[str, Callable[[list[str]], int]] = {
    "claude": _claude_runner,
    "gemini": _gemini_runner,
    "codex": _codex_runner,
}


class DispatcherFlags:
    """Mutable container for dispatcher-owned flags (Task 6)."""

    __slots__ = (
        "dry_run",
        "quiet",
        "force",
        "allow_agents_md_append",
        "remaining",
    )

    def __init__(self) -> None:
        self.dry_run: bool = False
        self.quiet: bool = False
        self.force: bool = False
        self.allow_agents_md_append: bool = False
        self.remaining: List[str] = []


def parse_dispatcher_flags(argv: list[str]) -> DispatcherFlags:
    """Consume dispatcher-owned leading flags; return the remainder.

    Flags:

    * `--dry-run` — run `plan()`, skip writes, skip `spawn()`.
    * `--quiet` — suppress per-step progress + banner.
    * `--force` — bypass recursion guard + clobber checks.
    * `--allow-agents-md-append` — opt-in for appending to tracked
      CLAUDE.md / AGENTS.md files.

    Flags are consumed greedily from the FRONT of argv; once a
    non-dispatcher token appears, the remainder is left intact for
    the backend binary. This means users can write
    `perfxpert-code claude --dry-run hello` and the `hello` prompt
    reaches the backend, while `perfxpert-code claude hello --dry-run`
    treats `--dry-run` as a backend flag (unambiguous).
    """
    flags = DispatcherFlags()
    idx = 0
    while idx < len(argv):
        a = argv[idx]
        if a == "--dry-run":
            flags.dry_run = True
        elif a == "--quiet":
            flags.quiet = True
        elif a == "--force":
            flags.force = True
        elif a == "--allow-agents-md-append":
            flags.allow_agents_md_append = True
        else:
            break
        idx += 1
    flags.remaining = list(argv[idx:])
    return flags


def _run_adapter(adapter, remaining_argv: list[str]) -> int:
    """End-to-end flow: parse flags → install → spawn (or dry-run).

    `adapter` must satisfy `BackendAdapter` (Task 1 Protocol). The
    backend name is read from `adapter.name`.
    """
    # Help passthrough — skip the installer entirely.
    if is_help_request(remaining_argv):
        # Just exec the backend with --help; adapter.spawn returns
        # the exit code if spawn_strategy == "subprocess".
        env = dict(os.environ)
        env[RECURSION_GUARD_ENV] = adapter.name
        return adapter.spawn(remaining_argv, env, Path.cwd())

    flags = parse_dispatcher_flags(remaining_argv)

    # Banner (unless quiet).
    if not flags.quiet:
        from perfxpert.cli.opencode_launcher import print_banner

        try:
            print_banner()
        except Exception:
            pass

    try:
        adapter.install(
            Path.cwd(),
            allow_agents_md_append=flags.allow_agents_md_append,
            dry_run=flags.dry_run,
            quiet=flags.quiet,
        )
    except Exception as exc:
        sys.stderr.write(
            f"perfxpert-code {adapter.name}: install failed: {exc}\n"
        )
        return 1

    if flags.dry_run:
        sys.stderr.write(
            f"[DRY-RUN] Would exec: {adapter.binary_name} {' '.join(flags.remaining)}\n"
        )
        return 0

    env = dict(os.environ)
    env[RECURSION_GUARD_ENV] = adapter.name
    if not flags.quiet:
        sys.stderr.write(
            f"perfxpert-code {adapter.name}: MCP verified; launching "
            f"{adapter.binary_name} ...\n"
        )
        sys.stderr.flush()
    return adapter.spawn(flags.remaining, env, Path.cwd())


def _exec_backend(name: str, remaining_argv: list[str]) -> int:
    """Dispatch to the named backend.

    Responsibilities:

    * Recursion guard (R5): refuse if `PERFXPERT_IN_AGENT_SESSION` is
      already set in env — unless `--force` is present in argv.
    * Help passthrough (practical §1.2): `--help` / `-h` short-circuits
      the installer and forwards to the backend binary directly.
    """
    # Recursion guard — refuse if we are already inside an agent session.
    already = os.environ.get(RECURSION_GUARD_ENV, "").strip()
    force_requested = parse_dispatcher_flags(remaining_argv).force
    if already and not force_requested:
        sys.stderr.write(
            f"perfxpert-code: already inside a perfxpert-{already} session "
            f"(via {RECURSION_GUARD_ENV}={already!r}). "
            "Pass --force to override.\n"
        )
        return 3

    handler = BACKEND_REGISTRY.get(name)
    if handler is None:
        sys.stderr.write(
            f"perfxpert-code: no handler registered for backend {name!r}.\n"
        )
        return 2

    return handler(remaining_argv)
