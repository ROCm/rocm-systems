"""predict_impact — change-impact prediction helper (Phase 10 stub).

This module is a minimal-surface placeholder that keeps the Memory
Specialist importable in isolation. A richer implementation lives in
untracked parallel work; we stub the three public functions the MCP
discovery layer + specialist expect so the Phase-10 advanced-specialist
bindings compile + run end-to-end.

Tool class: READ_ONLY (pure lookup; no side effects).
"""

from __future__ import annotations

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class


# Small deterministic catalog of supported changes. Kept intentionally
# conservative — real values live in the un-stubbed implementation.
_SUPPORTED_CHANGES = (
    "vgpr_reduction",
    "launch_bounds",
    "coalesce_access",
    "lds_tiling",
    "kernel_fusion",
)


@tool_class(ToolClass.READ_ONLY)
def predict_change_impact(
    technique: str = "",
    gfx_id: str = "",
    hot_kernels: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a conservative speedup bracket for a technique.

    Args:
        technique: Canonical technique id (e.g. ``"vgpr_reduction"``).
        gfx_id: Architecture id — currently unused by the stub.
        hot_kernels: List of hot-kernel dicts — currently unused.

    Returns:
        ``{"lo": float, "hi": float, "confidence": float}`` — conservative
        default of 1.00x-1.15x @ confidence 0.3 when the technique is not
        recognised; 1.05x-1.25x @ 0.5 for recognised techniques.
    """
    if technique in _SUPPORTED_CHANGES:
        return {"lo": 1.05, "hi": 1.25, "confidence": 0.5}
    return {"lo": 1.00, "hi": 1.15, "confidence": 0.3}


@tool_class(ToolClass.READ_ONLY)
def list_supported_changes() -> List[str]:
    """Enumerate the techniques the prediction tool knows about."""
    return list(_SUPPORTED_CHANGES)


@tool_class(ToolClass.READ_ONLY)
def explain_prediction(technique: str) -> Dict[str, Any]:
    """Return a short rationale dict for a prediction."""
    if technique in _SUPPORTED_CHANGES:
        return {
            "technique": technique,
            "rationale": f"Conservative bracket derived from historical "
                         f"deltas observed for '{technique}' across the "
                         "regression corpus.",
            "source": "stub",
        }
    return {
        "technique": technique,
        "rationale": "No historical data; returning conservative default.",
        "source": "stub",
    }


__all__ = [
    "predict_change_impact",
    "list_supported_changes",
    "explain_prediction",
]
