"""Codex gate hook — prompt-layer-only (Task 4.6 Codex-portion, PR 2).

**Surface decision.** Codex's `PreToolUse` hook exists (gated behind
`[features] codex_hooks = true` in `~/.codex/config.toml`) but as of
April 2026 **only intercepts Bash tool calls** — it cannot block MCP
tool calls, Write, WebSearch, etc. (Source: developers.openai.com/
codex/hooks, retrieved 2026-04-18, explicit docs note: "Currently
`PreToolUse` only supports Bash tool interception... this doesn't
intercept MCP, Write, WebSearch, or other non-shell tool calls.")

The perfxpert gate MUST intercept EVERY non-`perfxpert_*` tool call
until `intent_classify` returns. A Bash-only hook cannot satisfy
that requirement.

Per plan guardrail "If Codex's hook surface probe returns ambiguous,
default to prompt-layer-only": we default to prompt-layer-only here.
The full research + rationale is captured in the local Codex hook
surface decision record.

This module exposes `install(cwd)` / `uninstall(cwd)` as a uniform
gate-hook interface for the adapter. `install()` raises
`GateHookUnsupported` — which the `CodexAdapter` catches BEFORE MCP
registration runs (I-N1 partial-state protection) and records as a
documented-known-limit in `LiveCheckReport.gate_hook_installed=False`.

`evaluate_gate_state()` exists for symmetry with the Claude/Gemini
hooks; unit tests exercise it to document what the gate WOULD have
decided if the surface supported it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, NoReturn

from perfxpert.cli._backend.protocol import GateHookUnsupported
from perfxpert.cli._gate_hooks import (
    GATE_HOOK_DISABLED_ENV,
    GATE_REJECTION_REASON_TEMPLATE,
)


__all__ = [
    "install",
    "uninstall",
    "evaluate_gate_state",
]


def install(cwd: Path, *, env: dict | None = None) -> NoReturn:
    """Install the Codex gate hook.

    **Always raises `GateHookUnsupported`** on current Codex (April
    2026) because the native PreToolUse hook cannot intercept MCP
    tool calls. The `CodexAdapter` catches this and records
    `gate_hook_installed=False` as a documented-known-limit.

    Honors `PERFXPERT_GATE_HOOK=0` as a separate "user disabled"
    path so disabling the gate doesn't look like a surface failure.
    """
    if os.environ.get(GATE_HOOK_DISABLED_ENV, "").strip() == "0":
        raise GateHookUnsupported(
            f"{GATE_HOOK_DISABLED_ENV}=0 — Codex gate hook install "
            "skipped by user request (already prompt-layer-only; this "
            "further disables the prompt-layer warning)."
        )
    # Primary outcome: Codex surface cannot satisfy the event-based
    # gate requirement. The adapter translates this into a
    # prompt-layer-only install via its exception handler.
    raise GateHookUnsupported(
        "Codex native PreToolUse hook intercepts only Bash (not MCP, "
        "Write, or other tool types) as of April 2026 — cannot "
        "enforce the perfxpert gate mechanically. Degrading to "
        "prompt-layer-only enforcement (rejection-language stanza "
        "in the staged AGENTS.md). Rationale + re-visit conditions "
        "are captured in the local Codex hook-surface decision record."
    )


def uninstall(cwd: Path) -> None:
    """Uninstall the Codex gate hook.

    No-op: `install()` never writes files, so `uninstall()` has
    nothing to clean up. Kept for API uniformity with the Claude /
    Gemini hooks.
    """
    _ = cwd
    return None


def evaluate_gate_state(
    tool_name: str,
    *,
    intent_classify_observed: bool,
    classify_tool: str = "mcp__perfxpert__intent_classify",
) -> dict[str, Any]:
    """Mirror of the Claude / Gemini evaluators for unit test parity.

    Documents what the gate WOULD decide if Codex's hook surface
    supported MCP interception. Not wired into any live code path
    today; purely documentation-via-test.
    """
    if tool_name.startswith("mcp__perfxpert__"):
        return {"allowed": True, "enforced_by": "prompt-layer"}
    if intent_classify_observed:
        return {"allowed": True, "enforced_by": "prompt-layer"}
    return {
        "allowed": False,
        "enforced_by": "prompt-layer",
        "reason": GATE_REJECTION_REASON_TEMPLATE.format(
            classify_tool=classify_tool
        ),
    }
