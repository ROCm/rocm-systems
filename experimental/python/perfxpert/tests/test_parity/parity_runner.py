"""Orchestrator that runs BOTH the legacy and new analysis paths on a fixture
and returns a `DualResult` for comparison.

Legacy path: the pre-refactor call graph via `PERFXPERT_LEGACY=1`.

New path: the agent pipeline via the feature-flagged `analyze_database()` call
with `PERFXPERT_LEGACY` unset.

Both paths return an `AnalysisResult` dataclass (same public type) but populate
it through different call graphs. Parity tests compare structured fields:
  * summary.primary_bottleneck
  * recommendations.high_priority[0].category  → "primary rec type"
  * recommendations.high_priority[0].id → extract technique from legacy id pattern
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .fixtures_inventory import ParityFixture


@dataclass(frozen=True)
class SinglePathResult:
    path_label: str               # "old" | "new"
    primary_bottleneck: Optional[str]
    primary_rec_type: Optional[str]
    primary_rec_technique: Optional[str]
    raw_result: object             # the full AnalysisResult for debugging
    duration_s: float


@dataclass(frozen=True)
class DualResult:
    fixture_id: str
    old: SinglePathResult
    new: SinglePathResult

    def agree_bottleneck(self) -> bool:
        return self.old.primary_bottleneck == self.new.primary_bottleneck

    def agree_rec_type(self) -> bool:
        return self.old.primary_rec_type == self.new.primary_rec_type

    def agree_rec_technique(self) -> bool:
        # Null-safe; treat both-None as agreement (e.g. mixed/INFO fixtures)
        return self.old.primary_rec_technique == self.new.primary_rec_technique

    def agreements(self) -> dict:
        return {
            "bottleneck": self.agree_bottleneck(),
            "rec_type": self.agree_rec_type(),
            "rec_technique": self.agree_rec_technique(),
        }


_SOURCE_BACKGROUND_CATEGORIES = {"Initial Profiling", "Instrumentation"}
_SOURCE_CATEGORY_TO_BOTTLENECK = {
    "Memory Transfer": "memory_transfer",
    "Managed Memory": "memory_transfer",
    "Synchronization": "latency",
    "No Streams": "latency",
    "ROCm Libraries": "compute",
}
_SOURCE_CATEGORY_TO_REC_TYPE = {
    "Memory Transfer": "memory",
    "Managed Memory": "memory",
    "Synchronization": "latency",
    "No Streams": "latency",
    "ROCm Libraries": "compute",
    "Initial Profiling": "info",
    "Instrumentation": "info",
}
_SOURCE_CATEGORY_TO_TECHNIQUE = {
    "Memory Transfer": "hip_stream_overlap",
    "Managed Memory": "hip_stream_overlap",
    "Synchronization": "device_sync_removal",
    "No Streams": "hip_stream_overlap",
}


@contextmanager
def _force_path(new_path: bool):
    """Context manager: flip PERFXPERT_LEGACY for one run, restore on exit."""
    prev = os.environ.get("PERFXPERT_LEGACY")
    if new_path:
        os.environ.pop("PERFXPERT_LEGACY", None)
    else:
        os.environ["PERFXPERT_LEGACY"] = "1"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PERFXPERT_LEGACY", None)
        else:
            os.environ["PERFXPERT_LEGACY"] = prev


class ParityRunner:
    """Runs both paths on a ParityFixture and returns a DualResult."""

    def run_both_paths(self, fx: ParityFixture) -> DualResult:
        old = self._run_single(fx, new_path=False)
        new = self._run_single(fx, new_path=True)
        return DualResult(fixture_id=fx.id, old=old, new=new)

    def _run_single(self, fx: ParityFixture, *, new_path: bool) -> SinglePathResult:
        import time
        from perfxpert.ai_analysis.api import analyze_database, analyze_source

        with _force_path(new_path=new_path):
            start = time.time()
            if fx.source_dir:
                result = analyze_source(source_dir=str(fx.source_dir))
            else:
                result = analyze_database(database_path=str(fx.db_path), enable_llm=False)
            duration = time.time() - start

        return SinglePathResult(
            path_label="new" if new_path else "old",
            primary_bottleneck=_extract_bottleneck(result),
            primary_rec_type=_extract_rec_type(result),
            primary_rec_technique=_extract_rec_technique(result),
            raw_result=result,
            duration_s=duration,
        )


def _extract_bottleneck(result) -> Optional[str]:
    summary = getattr(result, "summary", None)
    if summary is not None:
        return getattr(summary, "primary_bottleneck", None)
    source_rec = _primary_source_recommendation(result)
    if source_rec is None:
        return None
    return _SOURCE_CATEGORY_TO_BOTTLENECK.get(source_rec.get("category"))


def _extract_rec_type(result) -> Optional[str]:
    source_rec = _primary_source_recommendation(result)
    if source_rec is not None:
        return _SOURCE_CATEGORY_TO_REC_TYPE.get(source_rec.get("category"))
    bottleneck = _extract_bottleneck(result)
    kernel_pct, memcpy_pct, api_pct = _percent_triplet(result)
    total_calls = _total_kernel_calls(result)
    avg_kernel_duration_us = _avg_kernel_duration_us(result)
    category = _primary_recommendation_category(result)

    if total_calls > 1000 and avg_kernel_duration_us is not None and avg_kernel_duration_us < 10 and api_pct > 0.15:
        return "latency"
    if memcpy_pct > 0.20 or bottleneck == "memory_transfer":
        return "memory"
    if category in {"API Overhead", "Launch Overhead", "Launch Efficiency", "GPU Utilization", "Low Occupancy"}:
        return "latency"
    if bottleneck in {"latency", "api_overhead"}:
        return "latency"
    if category in {"Compute-Bound Kernel", "Mixed Bottleneck Kernel", "Kernel Hotspot"}:
        return "compute"
    if bottleneck == "compute":
        return "compute"
    return None


def _extract_rec_technique(result) -> Optional[str]:
    source_rec = _primary_source_recommendation(result)
    if source_rec is not None:
        return _SOURCE_CATEGORY_TO_TECHNIQUE.get(source_rec.get("category"))
    candidate = _primary_recommendation(result)
    if candidate is not None:
        technique = getattr(candidate, "technique_id", None)
        if technique:
            return technique
        rid = getattr(candidate, "id", None)
        if rid and "-" in rid:
            return _legacy_id_to_technique(rid)

    bottleneck = _extract_bottleneck(result)
    kernel_pct, memcpy_pct, api_pct = _percent_triplet(result)
    total_calls = _total_kernel_calls(result)
    avg_kernel_duration_us = _avg_kernel_duration_us(result)
    top_kernel_name = _top_kernel_name(result)

    if total_calls > 1000 and avg_kernel_duration_us is not None and avg_kernel_duration_us < 10 and api_pct > 0.15:
        return "kernel_fusion_small_launches"
    if memcpy_pct > 0.20 or bottleneck == "memory_transfer":
        return "hip_stream_overlap"
    if bottleneck == "compute":
        if top_kernel_name and any(token in top_kernel_name.lower() for token in ("gemm", "matmul")):
            return "mfma_enablement"
        if kernel_pct > 0.70:
            return "vgpr_reduction_compute_bound"
    return None


def _primary_source_recommendation(result) -> Optional[dict]:
    """Return the first source-only recommendation with non-boilerplate signal."""
    recs = getattr(result, "recommendations", None)
    if not isinstance(recs, list):
        return None
    actionable = [
        rec for rec in recs
        if isinstance(rec, dict)
        and rec.get("category") not in _SOURCE_BACKGROUND_CATEGORIES
    ]
    if actionable:
        return actionable[0]
    if recs and isinstance(recs[0], dict):
        return recs[0]
    return None


def _legacy_id_to_technique(rid: str) -> Optional[str]:
    """Map legacy rec ids to technique ids from proven_optimizations.yaml."""
    tail = rid.split("-", 2)[1] if "-" in rid else ""
    mapping = {
        "OCCUPANCY": "vgpr_reduction_compute_bound",
        "MEMCPY": "hip_stream_overlap",
        "COALESCING": "memory_coalescing_stride_fix",
        "MFMA": "mfma_enablement",
        "FASTMATH": "fast_math_compiler_flag",
        "LDS": "lds_tiling_matmul",
        "FUSION": "kernel_fusion_small_launches",
        "DEVSYNC": "device_sync_removal",
        "WARP": "warp_primitives_reduction",
        "BLOCKING": "cache_blocking_kernel",
    }
    return mapping.get(tail)


def _primary_recommendation(result) -> Optional[Any]:
    recs = getattr(result, "recommendations", None)
    if recs is None:
        return None
    high = getattr(recs, "high_priority", None) or []
    if high:
        return high[0]
    med = getattr(recs, "medium_priority", None) or []
    if med:
        return med[0]
    return None


def _primary_recommendation_category(result) -> Optional[str]:
    rec = _primary_recommendation(result)
    if rec is None:
        return None
    return getattr(rec, "category", None)


def _result_hotspots(result) -> list[dict]:
    raw = getattr(result, "_raw", {}) or {}
    hotspots = raw.get("hotspots", [])
    return hotspots if isinstance(hotspots, list) else []


def _top_kernel_name(result) -> Optional[str]:
    hotspots = _result_hotspots(result)
    if hotspots and isinstance(hotspots[0], dict):
        return hotspots[0].get("name")
    return None


def _total_kernel_calls(result) -> int:
    total = 0
    for kernel in _result_hotspots(result):
        if isinstance(kernel, dict):
            total += int(kernel.get("calls", 0) or 0)
    return total


def _avg_kernel_duration_us(result) -> Optional[float]:
    total_calls = _total_kernel_calls(result)
    if total_calls <= 0:
        return None
    breakdown = getattr(result, "execution_breakdown", None)
    total_kernel_time_ns = getattr(breakdown, "kernel_time_ns", 0) or 0
    if total_kernel_time_ns:
        return float(total_kernel_time_ns) / float(total_calls) / 1000.0
    hotspots = _result_hotspots(result)
    total_duration = sum(int(kernel.get("duration_ns", kernel.get("total_duration", 0)) or 0) for kernel in hotspots if isinstance(kernel, dict))
    if total_duration <= 0:
        return None
    return float(total_duration) / float(total_calls) / 1000.0


def _fraction(value: Any) -> float:
    if value is None:
        return 0.0
    numeric = float(value)
    return numeric / 100.0 if numeric > 1.0 else numeric


def _percent_triplet(result) -> tuple[float, float, float]:
    breakdown = getattr(result, "execution_breakdown", None)
    if breakdown is None:
        return (0.0, 0.0, 0.0)
    return (
        _fraction(getattr(breakdown, "kernel_time_pct", 0.0)),
        _fraction(getattr(breakdown, "memcpy_time_pct", 0.0)),
        _fraction(getattr(breakdown, "api_overhead_pct", 0.0)),
    )
