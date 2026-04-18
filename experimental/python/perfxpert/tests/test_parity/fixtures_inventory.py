"""Canonical inventory of parity fixtures.

Fixture names reflect what is **actually on disk** inside tests/fixtures/.
The old inventory referenced names that never existed (compute_bound_gemm.db,
memory_bound_stride.db, etc.) — available_fixtures() was returning 0 and every
parity test was unconditionally skipped.

Disk layout (as of Phase 7 refactor):
  compute_bound.db       — Tier 2: has pmc_events (SQ_INSTS_VALU, GRBM_*,
                            FETCH_SIZE, WRITE_SIZE); 31 kernels
  memory_bound.db        — Tier 1: trace only (no pmc rows); 100 kernels;
                            high memcpy workload expected
  latency_bound.db       — Tier 1: trace only; 5002 short kernels / high
                            launch count — api_overhead or latency expected
  baseline.db            — Tier 1: neutral trace (2351 kernels); no pmc rows
  regressed.db           — Tier 1: regression trace (60 kernels); no pmc rows
  trace_only_elementwise.db — Tier 1: elementwise kernel trace; no pmc rows

Phase 5 adds no new fixtures to this list.  If a parity gap requires a new
fixture, the fixture and its documentation land in a targeted follow-up PR.

Parity gating policy (Phase 7):
  has_pmc_counters=True  → "PMC subset" — ship-gated; must be 100% consistent.
  has_pmc_counters=False → "Tier-1 trace-only" — excluded from the PMC-subset
                           gate.  These fixtures expose a known policy divergence:
                           the legacy path uses heuristics to classify trace-only
                           data; the agentic path correctly returns data_insufficient.
                           Re-profiling with --pmc (Part C) will flip these to True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"


@dataclass(frozen=True)
class ParityFixture:
    id: str                              # stable id for reporting
    db_path: Path                        # absolute path to the DB
    scenario: str                        # human description
    expected_bottleneck: str             # canonical label from bottleneck_types.yaml
    expected_rec_type: Optional[str]     # "compute" | "memory" | "latency" | "info" | None
    expected_rec_technique: Optional[str]  # e.g. "launch_bounds", "lds_tiling"
    tier: Literal[1, 2, 3]              # 1 = trace only; 2 = with counters; 3 = with ATT
    has_pmc_counters: bool = False       # True iff DB contains pmc_events rows.
                                         # PMC fixtures form the ship-gated parity subset.
                                         # Trace-only (False) fixtures are xfail until
                                         # re-profiled with --pmc (Phase 7 Part C).
    source_dir: Optional[Path] = None    # optional Tier 0 source path
    notes: str = ""


PARITY_FIXTURES: List[ParityFixture] = [
    ParityFixture(
        id="compute_bound",
        db_path=FIXTURE_ROOT / "compute_bound.db",
        scenario="Compute-bound trace: pure VALU kernel (heavy_valu_kernel) with high valu_util_pct and AI above ridge",
        expected_bottleneck="compute",
        expected_rec_type="compute",
        expected_rec_technique=None,   # any compute technique acceptable
        tier=2,
        has_pmc_counters=True,
        notes=(
            "Re-profiled with --pmc (Phase 7 Part C). "
            "Workload: pure VALU FP32 kernel (4M elements, 512 inner FMA iterations, 30 repeats). "
            "PMC verdict: compute (confidence 0.667, 2/3 rules: valu_util_pct=1.0, AI=23.7 > ridge). "
            "Part D evidence-weighting fix compatible: mfma_util_pct=None (skipped), "
            "2/3 evaluated rules pass → score=0.667 > 0.5 → compute. "
            "Counters: FETCH_SIZE, GRBM_COUNT, GRBM_GUI_ACTIVE, SQ_INSTS_VALU, SQ_WAVES, WRITE_SIZE."
        ),
    ),
    ParityFixture(
        id="memory_bound",
        db_path=FIXTURE_ROOT / "memory_bound.db",
        scenario="Memory-transfer-heavy workload (H2D/D2H copies + low-AI kernel)",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique=None,
        tier=2,
        has_pmc_counters=True,
        notes=(
            "Re-profiled with --pmc (Phase 7 Part C). "
            "Workload: 128MB arrays, 20 H2D+kernel+D2H repetitions, low-AI kernel. "
            "PMC verdict: memory_transfer (memcpy_pct=90%, hbm_bw_utilization≈0). "
            "Agentic and legacy both agree: memory_transfer."
        ),
    ),
    ParityFixture(
        id="latency_bound",
        db_path=FIXTURE_ROOT / "latency_bound.db",
        scenario="API-overhead-heavy workload (6000 tiny kernel launches + hipDeviceSynchronize)",
        expected_bottleneck="api_overhead",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=2,
        has_pmc_counters=True,
        notes=(
            "Re-profiled with --pmc (Phase 7 Part C). "
            "Workload: 6000 tiny kernel launches on 1024-element array, each sync'd. "
            "PMC verdict: api_overhead (api_pct=97%, avg_kernel_us=1.9, calls=6002). "
            "LEGACY HEURISTIC BUG: old path classified as 'latency' via overhead_pct>25% "
            "heuristic. PMC verdict is 'api_overhead' which is the correct classification. "
            "Legacy expected_bottleneck updated to match PMC truth."
        ),
    ),
    ParityFixture(
        id="baseline",
        db_path=FIXTURE_ROOT / "baseline.db",
        scenario="Memory-transfer-heavy baseline workload (same as memory_bound profile)",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique=None,
        tier=2,
        has_pmc_counters=True,
        notes=(
            "Re-profiled with --pmc (Phase 7 Part C). "
            "Same workload as memory_bound fixture. "
            "PMC verdict: memory_transfer. Agentic and legacy both agree."
        ),
    ),
    ParityFixture(
        id="regressed",
        db_path=FIXTURE_ROOT / "regressed.db",
        scenario="API-overhead regression fixture (same as latency_bound profile)",
        expected_bottleneck="api_overhead",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=2,
        has_pmc_counters=True,
        notes=(
            "Re-profiled with --pmc (Phase 7 Part C). "
            "Same workload as latency_bound fixture. "
            "PMC verdict: api_overhead. "
            "LEGACY HEURISTIC BUG: old path classified as 'latency'. "
            "PMC verdict is 'api_overhead' (correct). "
            "Fixture name 'regressed' retained for regression-gate test compatibility."
        ),
    ),
]


def available_fixtures() -> List[ParityFixture]:
    """Return the subset of fixtures whose db_path exists on disk."""
    return [
        fx for fx in PARITY_FIXTURES
        if fx.db_path.exists() or (fx.source_dir and fx.source_dir.exists())
    ]


def pmc_fixtures() -> List[ParityFixture]:
    """Return only the PMC-enriched fixtures (ship-gated parity subset).

    These are the fixtures where has_pmc_counters=True: both the legacy and
    agentic paths have the same counter evidence, so they must agree 100%.
    """
    return [fx for fx in available_fixtures() if fx.has_pmc_counters]


def trace_only_fixtures() -> List[ParityFixture]:
    """Return Tier-1 trace-only fixtures (excluded from ship-gated parity gate).

    These are xfail pending re-profile with --pmc (Phase 7 Part C).
    """
    return [fx for fx in available_fixtures() if not fx.has_pmc_counters]
