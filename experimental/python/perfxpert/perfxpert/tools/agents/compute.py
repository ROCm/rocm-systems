"""agent_compute_specialist — MCP tool wrapping the Layer-2 Compute specialist.

Compute-Techniques Specialist filters and ranks optimization techniques
for compute-bound kernels (VGPR pressure, FMA usage, wave occupancy,
roofline position). It returns a ranked list of techniques with impact
+ effort + risk metadata, a confidence score, and catalog citations.

Call this tool directly when the backend already knows the kernel is
compute-bound and wants technique recommendations without running the
full Root → Analysis → Recommendation chain.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the shared
provider-selection semantics from ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_compute_specialist(
    input: Dict[str, Any],
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run the Compute-Techniques specialist for a compute-bound kernel.

    Args:
        input: :class:`ComputeSpecialistInput` fields as a dict. Required:
            ``gfx_id`` (e.g. ``"gfx942"``), ``hot_kernels`` (list of
            kernel dicts). Optional: ``counter_data``, ``source_hints``.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), fall back to
            the deterministic catalog ranking (impact / effort).
        session_id: Re-use an existing session id. Generated when unset.
        progress_callback: Optional ``Callable[[str], None]`` fired with
            phase-transition messages. ``None`` for zero overhead.

    Returns:
        A dict with :class:`ComputeSpecialistOutput` keys:
        ``{"techniques", "confidence", "citations"}``.
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
        progress_callback=progress_callback,
    )
    payload = schemas.ComputeSpecialistInput(**input)
    output = session.run_compute_specialist(
        payload, progress_callback=progress_callback
    )

    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "techniques": list(getattr(output, "techniques", []) or []),
        "confidence": getattr(output, "confidence", 0.0),
        "citations": list(getattr(output, "citations", []) or []),
    }


agent_compute_specialist.__tool_name__ = "agent_compute_specialist"

__all__ = ["agent_compute_specialist"]
