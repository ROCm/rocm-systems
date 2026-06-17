"""sol — Speed-of-Light sanity + utilization classification.

Critical anti-Sakana reward-hacking defense (spec §5 Gate 2).
An LLM-produced optimization claiming performance that exceeds hardware
peak is a sandbox exploit, not a real speedup — REJECT.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, Literal

from perfxpert.tools._class import ToolClass, tool_class
# Imported as a private alias so the tool registry (which walks
# perfxpert.tools.* and collects public READ_ONLY callables) does not
# re-register this as `sol.lookup_peaks`. Canonical name is
# `arch.lookup_peaks`; this module re-uses it internally only.
from perfxpert.tools.arch import lookup_peaks as _lookup_peaks


KernelType = Literal["fp64", "fp32", "bf16"]


@tool_class(ToolClass.READ_ONLY)
def sanity_check(
    achieved_flops_per_sec: float,
    kernel_type: KernelType,
    gfx_id: str,
) -> Dict[str, Any]:
    """Verify a claimed achieved FLOPS is physically plausible.

    Blocks Sakana-style reward hacking where an LLM "optimization" claims
    speedups that exceed theoretical hardware peak — which is only
    achievable by gaming the sandbox (skipped work, broken timers, etc.).

    Args:
        achieved_flops_per_sec: claimed throughput in FLOPS/s.
        kernel_type: "fp64", "fp32", or "bf16".
        gfx_id: architecture identifier.

    Returns:
        {"plausible": bool, "reason": str, "sol_peak": float (FLOPS/s)}

    Raises:
        ValueError: if kernel_type is not supported.

    Example:
        >>> sanity_check(500e12, "fp64", "gfx942")
        {"plausible": False, "reason": "500 TFLOPS fp64 exceeds peak 81.7 TFLOPS", ...}
    """
    specs = _lookup_peaks(gfx_id)
    peak_key = f"peak_{kernel_type}_tflops"
    if peak_key not in specs:
        raise ValueError(
            f"Unknown kernel_type {kernel_type!r} for {gfx_id}; "
            f"known: {[k for k in specs if k.startswith('peak_')]}"
        )

    sol_peak = specs[peak_key] * 1e12  # TFLOPS → FLOPS

    if achieved_flops_per_sec > sol_peak:
        return {
            "plausible": False,
            "reason": (
                f"Claimed {achieved_flops_per_sec/1e12:.1f} TFLOPS {kernel_type} "
                f"exceeds hardware peak {sol_peak/1e12:.1f} TFLOPS — "
                f"likely sandbox exploit"
            ),
            "sol_peak": sol_peak,
        }

    return {
        "plausible": True,
        "reason": f"Achieved {achieved_flops_per_sec/1e12:.1f} TFLOPS is within peak {sol_peak/1e12:.1f}",
        "sol_peak": sol_peak,
    }


@tool_class(ToolClass.READ_ONLY)
def classify_utilization(pct: float) -> Dict[str, Any]:
    """Classify an SOL utilization fraction into high/medium/low.

    Args:
        pct: utilization fraction in 0.0-1.0 (not percentage).

    Returns:
        {"category": "high"|"medium"|"low", "headroom_pct": 0.0-1.0}

    Raises:
        ValueError: if pct is outside [0, 1].
    """
    if not (0.0 <= pct <= 1.0):
        raise ValueError(f"utilization fraction must be in [0,1]; got {pct}")

    if pct >= 0.80:
        cat = "high"
    elif pct >= 0.50:
        cat = "medium"
    else:
        cat = "low"

    return {"category": cat, "headroom_pct": 1.0 - pct}
