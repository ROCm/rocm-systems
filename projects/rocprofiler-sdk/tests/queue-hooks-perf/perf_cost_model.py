# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Cost model for dispatch-counter queue interposition perf tests.
# Architecture-agnostic ceilings for CI regression detection.

import os

# Per-dispatch counter instrumentation allowance (ms).
PER_DISPATCH_MS = float(os.environ.get("ROCPROFILER_QH_PER_DISPATCH_MS", "3.0"))

# Fixed attach / context setup (ms).
FIXED_OVERHEAD_MS = float(os.environ.get("ROCPROFILER_QH_FIXED_OVERHEAD_MS", "80.0"))

# Slack over linear launch scaling.
OVERHEAD_MARGIN = float(os.environ.get("ROCPROFILER_QH_OVERHEAD_MARGIN", "8.0"))


def model_max_ms(
    launches: int,
    per_dispatch_ms: float | None = None,
    fixed_ms: float | None = None,
    margin: float | None = None,
) -> float:
    per = PER_DISPATCH_MS if per_dispatch_ms is None else per_dispatch_ms
    fixed = FIXED_OVERHEAD_MS if fixed_ms is None else fixed_ms
    m = OVERHEAD_MARGIN if margin is None else margin
    return fixed + launches * per * m


def max_launch_scaling_ratio(l_low: int, l_high: int, slack: float = 2.0) -> float:
    if l_low <= 0:
        return slack * float(l_high)
    return (float(l_high) / float(l_low)) * slack
