"""Server-side tool-gate hook implementations per backend (Task 4.6, F1).

Event-based only (B-N3): the gate rejects any non-`perfxpert_*` tool
call UNTIL `perfxpert_intent_classify` has been observed returning in
the current session. There is no turn counter; a legitimate `bash`
on turn 2 after `intent_classify` on turn 1 passes through.

Each backend module exposes an adapter-shaped `install(cwd, ...)`
helper; `BackendAdapter.install()` calls it BEFORE MCP registration
(I-N1) so `GateHookUnsupported` surfaces cleanly without leaving
partial state.

Shared constants:

* `GATE_REJECTION_REASON_TEMPLATE` — the one-line "call
  intent_classify first" message. Same text in both the prompt-layer
  rejection stanza (Task 4a) and every backend's gate hook; avoids
  drift between the two enforcement paths.
* `GATE_STATE_LIFTED_SENTINEL` — magic token that every hook checks
  in its session-state file to decide "lifted this session".
* `GATE_HOOK_DISABLED_ENV` — escape hatch for reviewer debugging.

Explicitly NO `_FIRST_N_TURNS` constant — cycle-2 B-N3 removed the
turn-based rule.
"""

from __future__ import annotations

GATE_REJECTION_REASON_TEMPLATE = (
    "Tool call blocked by perfxpert gate: you must call "
    "{classify_tool} FIRST in this session. After it returns, any "
    "tool is permitted for the rest of the session."
)


GATE_STATE_LIFTED_SENTINEL = "perfxpert-gate-lifted-v1"


# Users can set `PERFXPERT_GATE_HOOK=0` to disable the hook entirely
# (for reviewer debugging or local-only workflows that don't want the
# gate at all).
GATE_HOOK_DISABLED_ENV = "PERFXPERT_GATE_HOOK"


__all__ = [
    "GATE_REJECTION_REASON_TEMPLATE",
    "GATE_STATE_LIFTED_SENTINEL",
    "GATE_HOOK_DISABLED_ENV",
]
