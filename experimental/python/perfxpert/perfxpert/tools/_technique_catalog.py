"""Shared loader for specialist technique catalogs.

Builds lightweight tool-facing catalogs from ``knowledge/proven_optimizations``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from perfxpert.knowledge import load_yaml


_ENTRY_SPECS: Dict[str, Dict[str, Any]] = {
    "vgpr_reduction_compute_bound": {
        "prediction_id": "vgpr_reduction",
        "effort_factor": 1.0,
        "risk": "low",
    },
    "mfma_enablement": {
        "prediction_id": "mfma_enablement",
        "effort_factor": 3.0,
        "risk": "medium",
    },
    "fast_math_compiler_flag": {
        "prediction_id": "fast_math_flag",
        "effort_factor": 0.5,
        "risk": "medium",
    },
    "memory_coalescing_stride_fix": {
        "prediction_id": "memory_coalescing_stride_fix",
        "effort_factor": 2.0,
        "risk": "medium",
    },
    "lds_tiling_matmul": {
        "prediction_id": "lds_tiling",
        "effort_factor": 3.0,
        "risk": "low",
    },
    "hip_stream_overlap": {
        "prediction_id": "hip_stream_overlap",
        "effort_factor": 2.0,
        "risk": "low",
    },
    "kernel_fusion_small_launches": {
        "prediction_id": "kernel_fusion_small_launches",
        "effort_factor": 3.0,
        "risk": "medium",
    },
    "device_sync_removal": {
        "prediction_id": "device_sync_removal",
        "effort_factor": 1.0,
        "risk": "low",
    },
    "warp_primitives_reduction": {
        "prediction_id": "warp_primitives_reduction",
        "effort_factor": 2.0,
        "risk": "medium",
    },
    "cache_blocking_kernel": {
        "prediction_id": "cache_blocking_kernel",
        "effort_factor": 2.5,
        "risk": "low",
    },
}


def _knowledge_index() -> Dict[str, Dict[str, Any]]:
    return {entry["id"]: entry for entry in load_yaml("proven_optimizations")}


def _expected_impact(entry: Dict[str, Any]) -> float:
    speedup_range = entry.get("measured_speedup_range") or [1.0, 1.0]
    hi = float(speedup_range[1] if len(speedup_range) > 1 else speedup_range[0])
    return max(0.0, hi - 1.0)


def catalog_for(
    public_names: Iterable[str],
    gfx_id: str,
    *,
    category: str,
) -> List[Dict[str, Any]]:
    """Return tool-facing technique dicts for the requested public names."""
    index = _knowledge_index()
    out: List[Dict[str, Any]] = []
    for public_name in public_names:
        entry = index.get(public_name)
        spec = _ENTRY_SPECS.get(public_name)
        if entry is None or spec is None:
            continue
        applies_to_gfx = set(entry.get("applies_to_gfx") or [])
        if applies_to_gfx and gfx_id not in applies_to_gfx:
            continue
        out.append(
            {
                "id": spec["prediction_id"],
                "name": public_name,
                "title": entry.get("technique", public_name.replace("_", " ")),
                "description": str(entry.get("description", "")).strip(),
                "category": category,
                "expected_impact": _expected_impact(entry),
                "effort_factor": spec["effort_factor"],
                "risk": spec["risk"],
                "source_citation": entry.get("source_citation", ""),
            }
        )
    return out


__all__ = ["catalog_for"]
