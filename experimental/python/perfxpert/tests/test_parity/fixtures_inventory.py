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
        scenario="Compute-bound trace with pmc_events (SQ_INSTS_VALU, GRBM counters)",
        expected_bottleneck="compute",
        expected_rec_type="compute",
        expected_rec_technique=None,   # any compute technique acceptable
        tier=2,
        has_pmc_counters=True,
        notes="Only fixture on disk with pmc_events rows (2542 rows, 31 kernels).",
    ),
    ParityFixture(
        id="memory_bound",
        db_path=FIXTURE_ROOT / "memory_bound.db",
        scenario="Trace-only, 100 kernels — expected high memcpy or data_insufficient",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique=None,
        tier=1,
        has_pmc_counters=False,
        notes=(
            "No pmc_events rows. Legacy=memory_transfer; agentic=data_insufficient. "
            "xfail until re-profiled with --pmc (Phase 7 Part C)."
        ),
    ),
    ParityFixture(
        id="latency_bound",
        db_path=FIXTURE_ROOT / "latency_bound.db",
        scenario="Trace-only, 5002 kernels — high launch count; no pmc rows",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique=None,
        tier=1,
        has_pmc_counters=False,
        notes=(
            "No pmc_events rows. Legacy=latency; agentic=data_insufficient. "
            "xfail until re-profiled with --pmc (Phase 7 Part C)."
        ),
    ),
    ParityFixture(
        id="baseline",
        db_path=FIXTURE_ROOT / "baseline.db",
        scenario="Neutral trace (2351 kernels); no pmc rows — memory_transfer expected",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique=None,
        tier=1,
        has_pmc_counters=False,
        notes=(
            "Trace-only baseline fixture. Legacy=memory_transfer; agentic=data_insufficient. "
            "xfail until re-profiled with --pmc (Phase 7 Part C)."
        ),
    ),
    ParityFixture(
        id="regressed",
        db_path=FIXTURE_ROOT / "regressed.db",
        scenario="Regression trace (60 kernels); no pmc rows",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique=None,
        tier=1,
        has_pmc_counters=False,
        notes=(
            "Regression trace. Legacy=latency; agentic=data_insufficient. "
            "xfail until re-profiled with --pmc (Phase 7 Part C)."
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
