"""agent_recommendation — MCP tool wrapping the Layer-1 Recommendation agent.

Recommendation turns an :class:`AnalysisOutput` verdict into a ranked,
deduplicated list of optimization techniques. It picks the right Layer-2
specialist (compute / memory / latency / none) based on the primary
bottleneck and returns the specialist's techniques plus a plateau flag.

Call this tool when the backend already has an analysis verdict and
wants the next-step technique list without re-running Analysis.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the shared
provider-selection semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_recommendation(
    input: Dict[str, Any],
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the Recommendation agent for an analysis verdict.

    Args:
        input: :class:`RecommendationInput` fields as a dict. Required:
            ``findings`` (an :class:`AnalysisOutput`-shaped dict).
            Optional: ``kernel_filter``, ``edit_history``,
            ``seen_recommendation_hashes``.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), skip every
            provider call.
        session_id: Re-use an existing session id. Generated when unset.
        progress_callback: Optional ``Callable[[str], None]`` fired with
            phase-transition messages. ``None`` for zero overhead.

    Returns:
        A dict with :class:`RecommendationOutput` keys:
        ``{"recommendations", "specialist_used", "plateau_detected"}``.
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
        progress_callback=progress_callback,
    )
    payload = schemas.RecommendationInput(**input)
    output = session.run_recommendation(
        payload, progress_callback=progress_callback
    )

    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "recommendations": list(getattr(output, "recommendations", []) or []),
        "specialist_used": getattr(output, "specialist_used", "none"),
        "plateau_detected": getattr(output, "plateau_detected", False),
    }


agent_recommendation.__tool_name__ = "agent_recommendation"

__all__ = ["agent_recommendation"]
