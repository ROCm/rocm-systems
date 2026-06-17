"""Canonical inventory of parity fixtures.

Every fixture here MUST have been created in Phases 1-4 (or be an inherited
real-trace test DB). When the full real-trace inventory is unavailable in CI,
the parity suite still needs a small floor of hermetic SQLite fixtures so it
cannot pass by skipping everything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures"


@dataclass(frozen=True)
class ParityFixture:
    id: str                              # stable id for reporting
    db_path: Path                        # relative to fixture root
    scenario: str                        # human description
    expected_bottleneck: str             # canonical label from bottleneck_types.yaml
    expected_rec_type: Optional[str]     # "compute" | "memory" | "latency" | "info" | None
    expected_rec_technique: Optional[str]  # e.g. "launch_bounds", "lds_tiling"
    tier: Literal[1, 2, 3]               # 1 = trace only; 2 = with counters; 3 = with ATT
    gfx_id: str = "gfx942"               # hardware context for recommendation routing
    source_only: bool = False            # Tier 0 source-only contract, excluded from parity gate
    source_dir: Optional[Path] = None    # optional Tier 0 source path
    notes: str = ""


PARITY_FIXTURES: List[ParityFixture] = [
    ParityFixture(
        id="mi300x_compute_bound_gemm",
        db_path=FIXTURE_ROOT / "compute_bound_gemm.db",
        scenario="Compute-bound MFMA-heavy GEMM on MI300X, high VALU util",
        expected_bottleneck="compute",
        expected_rec_type="compute",
        expected_rec_technique="mfma_enablement",
        tier=2,
    ),
    ParityFixture(
        id="mi300x_memory_bound_stride",
        db_path=FIXTURE_ROOT / "memory_bound_stride.db",
        scenario="Strided access pattern, low arithmetic intensity",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="memory_coalescing_stride_fix",
        tier=2,
    ),
    ParityFixture(
        id="mi300x_high_memcpy_pct",
        db_path=FIXTURE_ROOT / "high_memcpy_pct.db",
        scenario="> 30% memcpy_pct; H2D-dominated workload",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="hip_stream_overlap",
        tier=1,
    ),
    ParityFixture(
        id="mi300x_latency_low_occupancy",
        db_path=FIXTURE_ROOT / "latency_low_occupancy.db",
        scenario="avg_waves_per_cu = 6; GPU-util 40%",
        expected_bottleneck="latency",
        expected_rec_type="compute",
        expected_rec_technique="vgpr_reduction_compute_bound",
        tier=2,
    ),
    ParityFixture(
        id="mi300x_api_overhead_small_kernels",
        db_path=FIXTURE_ROOT / "api_overhead_small_kernels.db",
        scenario=">1000 kernel launches, avg < 10 µs",
        expected_bottleneck="api_overhead",
        expected_rec_type="latency",
        expected_rec_technique="kernel_fusion_small_launches",
        tier=1,
    ),
    ParityFixture(
        id="mi300x_device_sync_in_loop",
        db_path=FIXTURE_ROOT / "device_sync_in_loop.db",
        scenario="hipDeviceSynchronize called 500+ times",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique="device_sync_removal",
        tier=1,
    ),
    ParityFixture(
        id="mi300x_mixed_no_dominant",
        db_path=FIXTURE_ROOT / "mixed_balanced.db",
        scenario="No dominant bottleneck; kernel 60%, memcpy 15%, overhead 10%",
        expected_bottleneck="mixed",
        expected_rec_type=None,   # both paths expected to return INFO / triage rec
        expected_rec_technique=None,
        tier=1,
    ),
    ParityFixture(
        id="mi300x_att_stall_heavy",
        db_path=FIXTURE_ROOT / "att_heavy_vmem_stall.db",
        scenario="ATT run showing VMEM stall ratio > 0.5 on hot kernel",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="lds_tiling_matmul",
        tier=3,
    ),
    ParityFixture(
        id="mi300x_real_trace_merged_db",
        db_path=Path(
            "/dockerx/ai-analysis-rocpd/rocm-systems-dev/projects/rocprofiler-sdk/"
            "build/tests/rocprofv3/rocpd/rocpd-input-data/merged_db.db"
        ),
        scenario="Real 2000-dispatch rocprofv3 trace from build tree",
        expected_bottleneck="compute",  # validated manually
        expected_rec_type="compute",
        expected_rec_technique=None,    # any compute-class technique acceptable
        tier=2,
        notes="Source-of-truth real trace; absence tolerated (skipped) when not present",
    ),
    ParityFixture(
        id="mi350x_cdna4_lds_tiling",
        db_path=FIXTURE_ROOT / "mi350x_lds_tiling.db",
        scenario="gfx950 kernel with large LDS footprint; CDNA4 160 KB LDS test",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="lds_tiling_matmul",
        tier=2,
        gfx_id="gfx950",
    ),
    ParityFixture(
        id="tier0_blocking_memcpy",
        db_path=FIXTURE_ROOT / "_source_only_marker.db",  # placeholder; source_dir is primary
        scenario="Tier 0 source-only app with blocking hipMemcpy calls and no async overlap",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="hip_stream_overlap",
        tier=1,
        source_only=True,
        source_dir=FIXTURE_ROOT / "tier0_blocking_memcpy",
    ),
    ParityFixture(
        id="tier0_device_sync_loop",
        db_path=FIXTURE_ROOT / "_source_only_marker.db",  # placeholder; source_dir is primary
        scenario="Tier 0 source-only app with repeated hipDeviceSynchronize calls",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique="device_sync_removal",
        tier=1,
        source_only=True,
        source_dir=FIXTURE_ROOT / "tier0_device_sync_loop",
    ),
    ParityFixture(
        id="tier0_default_stream_only",
        db_path=FIXTURE_ROOT / "_source_only_marker.db",  # placeholder; source_dir is primary
        scenario="Tier 0 source-only app launching kernels only on the default stream",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique="hip_stream_overlap",
        tier=1,
        source_only=True,
        source_dir=FIXTURE_ROOT / "tier0_default_stream_only",
    ),
    ParityFixture(
        id="minimal_parity_compute_bound",
        db_path=FIXTURE_ROOT / "parity_compute_bound.db",
        scenario="Hermetic Tier 2 compute-bound matmul with high GPU utilization counters",
        expected_bottleneck="compute",
        expected_rec_type="compute",
        expected_rec_technique="mfma_enablement",
        tier=2,
        notes="Small SQLite fixture that keeps parity meaningful when real traces are absent.",
    ),
    ParityFixture(
        id="minimal_parity_memory_transfer",
        db_path=FIXTURE_ROOT / "parity_memory_transfer.db",
        scenario="Hermetic Tier 1 trace dominated by host-device memcpy time",
        expected_bottleneck="memory_transfer",
        expected_rec_type="memory",
        expected_rec_technique="hip_stream_overlap",
        tier=1,
        notes="Small SQLite fixture that keeps parity meaningful when real traces are absent.",
    ),
    ParityFixture(
        id="minimal_parity_launch_overhead",
        db_path=FIXTURE_ROOT / "parity_launch_overhead.db",
        scenario="Hermetic Tier 1 trace with many tiny kernels and heavy launch overhead",
        expected_bottleneck="latency",
        expected_rec_type="latency",
        expected_rec_technique="kernel_fusion_small_launches",
        tier=1,
        notes="Small SQLite fixture that keeps parity meaningful when real traces are absent.",
    ),
]


def available_fixtures() -> List[ParityFixture]:
    """Return every fixture present on disk (including Tier 0 source-only ones)."""
    return [
        fx for fx in PARITY_FIXTURES
        if fx.db_path.exists() or (fx.source_dir and fx.source_dir.exists())
    ]


def available_parity_fixtures() -> List[ParityFixture]:
    """Return the parity fixtures that should count toward the real parity gate."""
    return [fx for fx in available_fixtures() if not fx.source_only]


def available_source_only_fixtures() -> List[ParityFixture]:
    """Return the Tier 0 source-only fixtures for their dedicated contract."""
    return [fx for fx in available_fixtures() if fx.source_only]
