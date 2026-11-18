# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Shared kernel-replay performance cost model for CI regression tests.
# Uses relative thresholds and the Figure 5 byte budget (P x footprint over host
# link). No GPU product or ISA identifiers — suitable for any AMD GPU in CI.

import os

# Optional[...] rather than the PEP 604 "float | None": these annotations are evaluated when the
# module is imported, and the CI runners include distros whose system Python predates 3.10.
from typing import Optional

# Conservative host-link floor (GB/s). Intentionally low for heterogeneous CI
# including older AMD GPUs. Override locally for tighter checks:
#   ROCPROFILER_KR_MIN_SNAP_GBPS=8
DEFAULT_MIN_GBPS = float(os.environ.get("ROCPROFILER_KR_MIN_SNAP_GBPS", "4.0"))

# Multiplier over pure DMA prediction for drain, counter collection, snap discovery.
DEFAULT_OVERHEAD_MARGIN = float(os.environ.get("ROCPROFILER_KR_OVERHEAD_MARGIN", "10.0"))

# Per-pass kernel re-execution + profiler overhead allowance (seconds).
KERNEL_OVERHEAD_PER_PASS_S = 0.002

# End-to-end fixed cost (snap inventory, client attach, context) not in byte budget.
FIXED_REPLAY_OVERHEAD_MS = float(
    os.environ.get("ROCPROFILER_KR_FIXED_OVERHEAD_MS", "120.0")
)


def model_max_ms(
    ballast_mb: int,
    launches: int,
    passes: int,
    min_gbps: Optional[float] = None,
    margin: Optional[float] = None,
) -> float:
    gbps = DEFAULT_MIN_GBPS if min_gbps is None else min_gbps
    overhead = DEFAULT_OVERHEAD_MARGIN if margin is None else margin
    footprint_bytes = ballast_mb * 1024 * 1024
    per_dispatch_bytes = passes * footprint_bytes
    dma_seconds = per_dispatch_bytes / (gbps * 1e9)
    kernel_seconds = passes * KERNEL_OVERHEAD_PER_PASS_S
    per_dispatch_ms = (dma_seconds + kernel_seconds) * 1000.0
    return launches * per_dispatch_ms * overhead + FIXED_REPLAY_OVERHEAD_MS


def max_pass_scaling_ratio(p_low: int, p_high: int, slack: float = 2.0) -> float:
    """Upper bound on wall-time ratio P_high/P_low assuming linear pass scaling."""
    if p_low <= 0:
        return slack * float(p_high)
    return (float(p_high) / float(p_low)) * slack
