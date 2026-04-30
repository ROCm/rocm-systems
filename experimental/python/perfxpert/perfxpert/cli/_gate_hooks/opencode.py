"""Opencode gate hook — prompt-layer patch 0020 extension (Task 4.6, B-N2).

**Important: this hook is prompt-layer only for opencode today.**

`perfxpert/.patches/0020-perfxpert-tool-gate.patch` strengthens the primary
opencode prompts with tool-priority rejection language. It does not patch
opencode's TypeScript runtime to mechanically block or rewrite a bad first
tool call. When a user intentionally uses the upstream escape hatch
(`perfxpert-code opencode ...` with `PERFXPERT_OPENCODE_PATH`), the patched
prompt is absent and only any user-supplied upstream guidance remains.

Session-state: held in memory by the patched opencode process
(in-session object). Invalidation = process exit (session end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpencodeGateInstallResult:
    fork_only_notice: str = (
        "opencode gate discipline is prompt-layer only; patched opencode "
        "adds PerfXpert prompt/tool-priority guidance, not a mechanical "
        "runtime block hook."
    )
    # No files to write on the opencode side: the patch bakes the prompt
    # guidance directly into the opencode prompt files.
    installed: bool = True


def install(cwd: Any, *, env: dict | None = None) -> OpencodeGateInstallResult:
    """Install the opencode gate hook.

    For the patched opencode binary this is a no-op install: prompt-layer
    gate guidance is compiled into the prompt files via patch 0020. The
    function exists so the per-backend install flow is uniform and so future
    runtime-hook extensions have a single entry point.

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

    Returns either `{}` (allow) or the
    `{"block": True, "retryWith": <message>}` payload reserved for a future
    mechanical runtime hook. Current patched opencode does not consume it.

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
