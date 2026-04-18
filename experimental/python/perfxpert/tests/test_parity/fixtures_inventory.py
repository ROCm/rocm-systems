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
"""

from __future__ import annotations

from dataclasses import dataclass
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
        notes="Only fixture on disk with pmc_events rows (2542 rows, 31 kernels).",
    ),
    ParityFixture(
        id="memory_bound",
        db_path=FIXTURE_ROOT / "memory_bound.db",
        scenario="Trace-only, 100 kernels — expected high memcpy or data_insufficient",
        expected_bottleneck="data_insufficient",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=1,
        notes="No pmc_events rows. Classifier will return data_insufficient (no counter evidence).",
    ),
    ParityFixture(
        id="latency_bound",
        db_path=FIXTURE_ROOT / "latency_bound.db",
        scenario="Trace-only, 5002 kernels — high launch count; no pmc rows",
        expected_bottleneck="data_insufficient",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=1,
        notes="No pmc_events rows. Classifier will return data_insufficient.",
    ),
    ParityFixture(
        id="baseline",
        db_path=FIXTURE_ROOT / "baseline.db",
        scenario="Neutral trace (2351 kernels); no pmc rows — data_insufficient expected",
        expected_bottleneck="data_insufficient",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=1,
        notes="Trace-only baseline fixture. Same schema as trace_only_elementwise.",
    ),
    ParityFixture(
        id="regressed",
        db_path=FIXTURE_ROOT / "regressed.db",
        scenario="Regression trace (60 kernels); no pmc rows",
        expected_bottleneck="data_insufficient",
        expected_rec_type=None,
        expected_rec_technique=None,
        tier=1,
        notes="Used in regression gate tests; included here to verify parity runner handles it.",
    ),
]


def available_fixtures() -> List[ParityFixture]:
    """Return the subset of fixtures whose db_path exists on disk."""
    return [
        fx for fx in PARITY_FIXTURES
        if fx.db_path.exists() or (fx.source_dir and fx.source_dir.exists())
    ]
