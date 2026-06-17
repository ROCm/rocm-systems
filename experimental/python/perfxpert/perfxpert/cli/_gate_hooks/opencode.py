"""Opencode gate hook — fork-only patch 0020 extension (Task 4.6, B-N2).

**Important: this hook depends on the bundled patched opencode.**

The `{ block: true, retryWith: <message> }` return shape from
`plugin.trigger("tool.execute.before", ...)` is NOT part of upstream
opencode's public plugin API. It is provided by
`perfxpert/.patches/0020-perfxpert-tool-gate.patch`. When a user
swaps `PERFXPERT_OPENCODE_PATH` to point at an upstream (unpatched)
opencode binary, the plugin trigger returns a truthy object but
opencode ignores the `block` field and lets the tool run. In that
scenario the hook degrades to prompt-layer enforcement only (the
rejection-language stanza from Task 4a `_prompt_adapter`).

This is documented behavior, not a bug. `verify_mcp_live` records
`gate_hook_installed=False` + warning-level log when the user has
swapped out the bundled binary.

Session-state: held in memory by the patched opencode process
(in-session object). Invalidation = process exit (session end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpencodeGateInstallResult:
    fork_only_notice: str = (
        "opencode gate hook requires the bundled patched opencode "
        "(perfxpert/.patches/0020-perfxpert-tool-gate.patch). Upstream "
        "opencode silently ignores the {block, retryWith} return shape."
    )
    # No files to write on the opencode side: the patch bakes the hook
    # directly into the opencode binary's plugin invocation path.
    installed: bool = True


def install(cwd: Any, *, env: dict | None = None) -> OpencodeGateInstallResult:
    """Install the opencode gate hook.

    For the bundled patched opencode this is a no-op install: the
    gate logic is compiled into the binary via patch 0020. The
    function exists so the per-backend install flow is uniform and
    so future extensions (per-session config files) have a single
    entry point.

    `cwd` and `env` are accepted for signature parity with the
    claude / gemini hooks; they are unused today.
    """
    return OpencodeGateInstallResult()


def evaluate(
    tool_name: str,
    *,
    intent_classify_observed: bool,
    classify_tool: str = "perfxpert_intent_classify",
) -> dict[str, Any]:
    """Event-based gate decision (B-N3).

    Returns either `{}` (allow) or the fork-only
    `{"block": True, "retryWith": <message>}` payload that the
    patched opencode binary interprets.

    Rule:

    * Any `perfxpert_*` tool is ALWAYS allowed (otherwise the user
      cannot call `intent_classify` to lift the gate).
    * Any other tool is allowed iff `intent_classify_observed` is
      True (the caller tracks the in-memory session state).
    """
    if tool_name.startswith("perfxpert_") or tool_name.startswith("mcp__perfxpert__"):
        return {}
    if intent_classify_observed:
        return {}

    from perfxpert.cli._gate_hooks import GATE_REJECTION_REASON_TEMPLATE

    return {
        "block": True,
        "retryWith": GATE_REJECTION_REASON_TEMPLATE.format(
            classify_tool=classify_tool
        ),
    }
