"""Weighted geomean computation over benchmark RunResults.

Weighting: each kernel's log-speedup is weighted by its baseline runtime.
This is closer to "business impact" than a flat geomean — a 2× speedup on
a 1ns kernel shouldn't dominate a 1.05× speedup on a 1-second kernel.
"""

import math
from typing import Iterable, List

from tests.benchmarks.tritonbench_runner import RunResult


def filter_recommended(results: Iterable[RunResult]) -> List[RunResult]:
    """Keep only kernels where perfxpert emitted a recommendation AND it was applied."""
    return [r for r in results if r.pr_applied]


def weighted_geomean(results: Iterable[RunResult]) -> float:
    """Compute weighted geomean of speedup over `results` (baseline_ns-weighted)."""
    rs = list(results)
    if not rs:
        return 1.0
    total_weight = float(sum(r.baseline_ns for r in rs))
    if total_weight == 0.0:
        return 1.0
    weighted_log = sum(math.log(r.speedup) * (r.baseline_ns / total_weight) for r in rs)
    return math.exp(weighted_log)


def assert_geomean_above(results: Iterable[RunResult], *, threshold: float = 1.20) -> None:
    """Raise AssertionError if weighted geomean of recommendations is below threshold."""
    recs = filter_recommended(results)
    g = weighted_geomean(recs)
    if g < threshold:
        details = ", ".join(f"{r.kernel_id}={r.speedup:.2f}×" for r in recs)
        raise AssertionError(
            f"weighted geomean {g:.3f} < {threshold:.2f} threshold.\n"
            f"Per-kernel speedups: {details}"
        )
