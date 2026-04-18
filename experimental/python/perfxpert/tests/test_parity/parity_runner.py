"""Orchestrator that runs BOTH the legacy and new analysis paths on a fixture
and returns a `DualResult` for comparison.

Legacy path: the pre-refactor call graph via `PERFXPERT_USE_AGENTS=0`.

New path: the agent pipeline via the feature-flagged `analyze_database()` call
with `PERFXPERT_USE_AGENTS=1`.

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
from typing import Optional

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


@contextmanager
def _force_path(new_path: bool):
    """Context manager: flip PERFXPERT_USE_AGENTS for one run, restore on exit."""
    prev = os.environ.get("PERFXPERT_USE_AGENTS")
    os.environ["PERFXPERT_USE_AGENTS"] = "1" if new_path else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("PERFXPERT_USE_AGENTS", None)
        else:
            os.environ["PERFXPERT_USE_AGENTS"] = prev


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
    if summary is None:
        return None
    return getattr(summary, "primary_bottleneck", None)


def _extract_rec_type(result) -> Optional[str]:
    recs = getattr(result, "recommendations", None)
    if recs is None:
        return None
    high = getattr(recs, "high_priority", None) or []
    if not high:
        med = getattr(recs, "medium_priority", None) or []
        if med:
            return getattr(med[0], "category", None)
        return None
    return getattr(high[0], "category", None)


def _extract_rec_technique(result) -> Optional[str]:
    recs = getattr(result, "recommendations", None)
    if recs is None:
        return None
    high = getattr(recs, "high_priority", None) or []
    if not high:
        return None
    # Technique lives in rec.technique_id (new path) OR rec.id first tail token (old path).
    r0 = high[0]
    technique = getattr(r0, "technique_id", None)
    if technique:
        return technique
    rid = getattr(r0, "id", None)
    if rid and "-" in rid:
        return _legacy_id_to_technique(rid)
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
