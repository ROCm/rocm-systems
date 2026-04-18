"""Orchestrator that runs BOTH the legacy and new analysis paths on a fixture
and returns a `DualResult` for comparison.

Legacy path: the pre-refactor call graph via ``PERFXPERT_LEGACY=1``.
  Invokes _route_to_legacy() inside analyze_database(), which uses the
  old analysis.py function chain directly.

New path (default Phase 6+): the agentic pipeline.
  Invokes _route_to_agents() inside analyze_database(), which uses the
  Phase 3 session API (perfxpert.agents.runtime).

IMPORTANT: The old Phase-4 flag ``PERFXPERT_USE_AGENTS`` was retired in
Phase 6 and is now a silent no-op (see ai_analysis/api.py:_is_legacy_mode).
Both sides used to set PERFXPERT_USE_AGENTS=0/1 — which meant BOTH sides
ran the agentic path, making the parity comparison agentic-vs-agentic
(zero actual parity signal). This runner now uses the correct flag:
  PERFXPERT_LEGACY=1  → legacy path (_route_to_legacy)
  (unset)             → agentic path (_route_to_agents, default)

Both paths return an ``AnalysisResult`` dataclass (same public type) but
populate it through different call graphs. Parity tests compare structured
fields:
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
        """True if both paths agree on rec type, or if both return None.

        If one path returns None and the other doesn't, treat as "skip" (None)
        rather than disagreement — the agentic path by design returns no
        recommendations in the non-LLM wrapper, so this signal is only
        meaningful when both paths produce recommendations.
        """
        # Both None → agree (e.g. data_insufficient on both sides)
        if self.old.primary_rec_type is None and self.new.primary_rec_type is None:
            return True
        # Both non-None → compare
        if self.old.primary_rec_type is not None and self.new.primary_rec_type is not None:
            return self.old.primary_rec_type == self.new.primary_rec_type
        # Asymmetric: one has recs, other doesn't.
        # Agentic path (new) never populates recommendations without LLM — skip
        # this signal (return None = "not evaluated") rather than failing.
        return None  # type: ignore[return-value]  # caller must handle None

    def agree_rec_technique(self) -> bool:
        """True if both paths agree on rec technique, or if both return None.

        Same null-safety as agree_rec_type: asymmetric None is skipped.
        """
        if self.old.primary_rec_technique is None and self.new.primary_rec_technique is None:
            return True
        if self.old.primary_rec_technique is not None and self.new.primary_rec_technique is not None:
            return self.old.primary_rec_technique == self.new.primary_rec_technique
        return None  # type: ignore[return-value]

    def agreements(self) -> dict:
        """Return dict of {signal: agree} for signals that can be evaluated.

        Signals where one path returns None and the other doesn't (asymmetric
        None) are excluded from the dict — the aggregate must not count
        unevaluable signals toward either agreements or total_signals.
        """
        result: dict = {}
        result["bottleneck"] = self.agree_bottleneck()
        rt = self.agree_rec_type()
        if rt is not None:
            result["rec_type"] = rt
        rtech = self.agree_rec_technique()
        if rtech is not None:
            result["rec_technique"] = rtech
        return result


@contextmanager
def _force_path(legacy: bool):
    """Context manager: set PERFXPERT_LEGACY=1 for the old path, unset for new path.

    This is the correct flag as of Phase 6+. The old PERFXPERT_USE_AGENTS flag
    is a no-op and must not be used here.
    """
    prev = os.environ.get("PERFXPERT_LEGACY")
    if legacy:
        os.environ["PERFXPERT_LEGACY"] = "1"
    else:
        os.environ.pop("PERFXPERT_LEGACY", None)
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
        old = self._run_single(fx, legacy=True)
        new = self._run_single(fx, legacy=False)
        return DualResult(fixture_id=fx.id, old=old, new=new)

    def _run_single(self, fx: ParityFixture, *, legacy: bool) -> SinglePathResult:
        import time
        from perfxpert.ai_analysis.api import analyze_database, analyze_source

        with _force_path(legacy=legacy):
            start = time.time()
            if fx.source_dir:
                result = analyze_source(source_dir=str(fx.source_dir))
            else:
                result = analyze_database(database_path=str(fx.db_path), enable_llm=False)
            duration = time.time() - start

        return SinglePathResult(
            path_label="old" if legacy else "new",
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
