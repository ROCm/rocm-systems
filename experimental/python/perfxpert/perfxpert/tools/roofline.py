"""roofline — arithmetic intensity + regime classification.

Pure arithmetic. Replaces the LLM's roofline-model reasoning with a single
rule: if AI > ridge_point → compute; else memory.

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools.arch import lookup_peaks


_BALANCED_TOLERANCE = 0.05  # ±5% around ridge = "balanced"


@tool_class(ToolClass.READ_ONLY)
def classify(flops: float, bytes: float, gfx_id: str) -> Dict[str, Any]:
    """Classify a kernel's regime using the roofline model.

    Args:
        flops: Total floating-point operations for the kernel.
        bytes: Total memory bytes read+written (HBM traffic).
        gfx_id: Architecture identifier (e.g., "gfx942").

    Returns:
        {
            "arithmetic_intensity": FLOPS/Byte,
            "ridge_point": ridge for this arch,
            "regime": "compute" | "memory" | "balanced",
            "distance_to_roof": 0.0-1.0 (how far below peak)
        }

    Raises:
        ValueError: if bytes == 0.
        KeyError: if gfx_id is not recognized.

    Example:
        >>> classify(flops=1e12, bytes=1e11, gfx_id="gfx942")
        {"arithmetic_intensity": 10.0, "ridge_point": 15.4, "regime": "memory", ...}
    """
    if bytes == 0:
        raise ValueError("bytes must be > 0 — divide-by-zero in arithmetic intensity")

    specs = lookup_peaks(gfx_id)
    ridge = specs["ridge_point"]
    ai = flops / bytes

    # Classify regime
    if abs(ai - ridge) / ridge <= _BALANCED_TOLERANCE:
        regime = "balanced"
    elif ai > ridge:
        regime = "compute"
    else:
        regime = "memory"

    # Distance to roof: how much performance is "on the table"
    # Crude: normalized gap between actual AI and the higher of (ridge, AI)
    if regime == "compute":
        # Compute-bound → measured FLOPS vs peak FLOPS (caller would supply separately)
        # Phase 1 stub: just return 0.5 placeholder; refined in Phase 3 when we have measured perf
        distance = 0.5
    else:
        distance = min(ai / ridge, 1.0)

    return {
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
        "regime": regime,
        "distance_to_roof": distance,
    }
