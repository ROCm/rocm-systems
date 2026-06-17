"""agent_analysis — MCP tool wrapping the Layer-1 Analysis agent.

Analysis classifies the primary bottleneck of a GPU trace: it reads
the trace database, computes time breakdown and hot kernels, and
returns a frozen :class:`AnalysisOutput` with the bottleneck type,
confidence, hot kernels, and counter-availability flag.

Call this tool when the backend wants ONLY the bottleneck verdict —
without the surrounding Root narrative or recommendation list.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the shared
provider-selection semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_analysis(
    input: Dict[str, Any],
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the Analysis agent on a trace database.

    Args:
        input: :class:`AnalysisInput` fields as a dict. Required key:
            ``database_path``. Optional: ``top_kernels`` (default 10),
            ``att_dir``.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), skip every
            provider call.
        session_id: Re-use an existing session id. Generated when unset.
        progress_callback: Optional ``Callable[[str], None]`` fired with
            phase-transition messages. ``None`` for zero overhead.

    Returns:
        A dict with :class:`AnalysisOutput` keys:
        ``{"primary_bottleneck", "confidence", "time_breakdown",
        "hot_kernels", "counter_data_available"}``.
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
        progress_callback=progress_callback,
    )
    payload = schemas.AnalysisInput(**input)
    output = session.run_analysis(payload, progress_callback=progress_callback)

    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "primary_bottleneck": getattr(output, "primary_bottleneck", "mixed"),
        "confidence": getattr(output, "confidence", 0.0),
        "time_breakdown": dict(getattr(output, "time_breakdown", {}) or {}),
        "hot_kernels": list(getattr(output, "hot_kernels", []) or []),
        "counter_data_available": getattr(
            output, "counter_data_available", False
        ),
    }


agent_analysis.__tool_name__ = "agent_analysis"

__all__ = ["agent_analysis"]
