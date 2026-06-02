"""agent_diff_specialist — MCP tool wrapping the Layer-2 Trace-Diff specialist.

Trace-Diff Specialist compares a baseline rocprofiler-sdk database
against a new run and returns a structured verdict (improved /
regressed / neutral) plus ranked per-kernel deltas and a narrative.

Call this tool conversationally from any TUI backend (opencode /
claude / codex / gemini) when the user says things like "diff this
run against baseline.db" or "what got slower since yesterday's
trace?" — the backend LLM picks this tool based on the
``AGENTS.md`` inventory rather than running ``analyze`` twice.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the shared
provider-selection semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_diff_specialist(
    baseline_db: str,
    new_db: str,
    top_kernels: int = 20,
    user_intent: str = "summarize the diff",
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the Trace-Diff specialist.

    Args:
        baseline_db: Path to the baseline rocprofiler-sdk ``.db``.
        new_db: Path to the new run's rocprofiler-sdk ``.db``.
        top_kernels: How many kernels to retain in the per-kernel diff
            list (default 20, capped at 100).
        user_intent: Free-text hint from the user ("summarize",
            "find the worst regression", etc.). Shapes the narrative
            tone only; the arithmetic is always deterministic.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), fall back to
            the deterministic narrative template.
        session_id: Re-use an existing session id. Generated when unset.
        progress_callback: Optional ``Callable[[str], None]`` fired with
            phase-transition messages. ``None`` for zero overhead.

    Returns:
        A dict with the flattened :class:`DiffSpecialistOutput` shape::

            {
              "wall_delta_pct": float,
              "regressions":   [{name, baseline_ns, new_ns, delta_pct, ...}],
              "improvements":  [{name, baseline_ns, new_ns, delta_pct, ...}],
              "verdict":       "improved" | "regressed" | "neutral",
              "narrative":     str,
              "confidence":    float  (0..1),
            }

        ``regressions`` and ``improvements`` are flattened from the
        Pydantic output's ``kernel_deltas`` dict so callers interact
        with the shape documented in the agent hierarchy reference
        (not the internal 5-field-capped schema).
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
        progress_callback=progress_callback,
    )
    payload = schemas.DiffSpecialistInput(
        baseline_db=baseline_db,
        new_db=new_db,
        top_kernels=top_kernels,
        user_intent=user_intent,
    )
    output = session.run_diff_specialist(
        payload, progress_callback=progress_callback
    )

    if hasattr(output, "model_dump"):
        raw = output.model_dump()
    else:  # pragma: no cover — defensive
        raw = dict(output)

    # Flatten ``kernel_deltas`` → top-level regressions / improvements
    # so the public tool signature matches the agent-hierarchy docs.
    deltas = raw.pop("kernel_deltas", {}) or {}
    raw["regressions"] = list(deltas.get("regressions", []) or [])
    raw["improvements"] = list(deltas.get("improvements", []) or [])
    return raw


agent_diff_specialist.__tool_name__ = "agent_diff_specialist"

__all__ = ["agent_diff_specialist"]
