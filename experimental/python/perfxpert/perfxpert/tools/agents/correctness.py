"""agent_correctness — MCP tool wrapping the Layer-1 Correctness agent.

Correctness consumes a :class:`GateVerdictModel` from the 5-gate cascade
(compile / sol / bitwise / regression / anchors) and decides whether an
applied optimization patch should be accepted, reverted, or flagged.
The output is a frozen :class:`CorrectnessOutput` with a verdict and a
concrete ``action`` (``accept`` / ``revert`` / ``reject_and_log``).

Call this tool after the backend has run a gate-cascade probe (e.g.
compile + sol check) and needs a structured decision on the patch.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the shared
provider-selection semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_correctness(
    input: Dict[str, Any],
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the Correctness agent on a gate-cascade verdict.

    Args:
        input: :class:`CorrectnessInput` fields as a dict. Required:
            ``gate_verdict`` (a :class:`GateVerdictModel`-shaped dict
            with at least ``status`` and — when ``status != 'pass'`` —
            ``failing_gate``). Optional: ``kernel_name``,
            ``last_technique``, ``edit_history``.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), skip every
            provider call.
        session_id: Re-use an existing session id. Generated when unset.
        progress_callback: Optional ``Callable[[str], None]`` fired with
            phase-transition messages. ``None`` for zero overhead.

    Returns:
        A dict with :class:`CorrectnessOutput` keys:
        ``{"verdict", "action", "narrative", "alternative_technique",
        "follow_up_task_id"}``.
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
        progress_callback=progress_callback,
    )
    payload = schemas.CorrectnessInput(**input)
    output = session.run_correctness(payload, progress_callback=progress_callback)

    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "verdict": getattr(output, "verdict", "reject"),
        "action": getattr(output, "action", "reject_and_log"),
        "narrative": getattr(output, "narrative", ""),
        "alternative_technique": getattr(output, "alternative_technique", None),
        "follow_up_task_id": getattr(output, "follow_up_task_id", None),
    }


agent_correctness.__tool_name__ = "agent_correctness"

__all__ = ["agent_correctness"]
