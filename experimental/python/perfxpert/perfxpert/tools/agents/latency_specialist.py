"""agent_latency_specialist — MCP tool wrapping the Layer-2 Latency specialist.

Latency-Techniques Specialist ranks optimization techniques for
latency-bound or API-overhead-bound workloads (short kernels, HIP-API
overhead, kernel-launch batching, graph capture). It returns a ranked
list of techniques with impact + effort + risk metadata, a confidence
score, and citations.

Call this tool directly when the backend already knows the bottleneck
is launch-overhead / latency (e.g. many short kernels or high
``api_overhead_pct``) and wants techniques without the full chain.

Tool class: READ_ONLY. Honors ``PERFXPERT_AIRGAP=1`` + the full
provider / fallback-chain ladder via ``agents.runtime.build_session``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def agent_latency_specialist(
    input: Dict[str, Any],
    provider: Optional[str] = None,
    airgap: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the Latency-Techniques specialist for a latency-bound workload.

    Args:
        input: :class:`LatencySpecialistInput` fields as a dict. Required:
            ``gfx_id``, ``hot_kernels``. Optional: ``api_overhead_pct``,
            ``avg_kernel_duration_us``.
        provider: Explicit LLM provider name. Ignored under airgap.
        airgap: When ``True`` (or ``PERFXPERT_AIRGAP=1``), fall back to
            the deterministic catalog ranking.
        session_id: Re-use an existing session id. Generated when unset.

    Returns:
        A dict with :class:`LatencySpecialistOutput` keys:
        ``{"techniques", "confidence", "citations"}``.
    """
    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(
        provider=provider,
        session_id=session_id,
        airgap=airgap if airgap else None,
    )
    payload = schemas.LatencySpecialistInput(**input)
    output = session.run_latency_specialist(payload)

    if hasattr(output, "model_dump"):
        return output.model_dump()
    return {
        "techniques": list(getattr(output, "techniques", []) or []),
        "confidence": getattr(output, "confidence", 0.0),
        "citations": list(getattr(output, "citations", []) or []),
    }


__all__ = ["agent_latency_specialist"]
