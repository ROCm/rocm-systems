#!/usr/bin/env python3
###############################################################################
# MIT License
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""
TraceLens-derived analysis algorithms for rocpd.

Ports interval arithmetic, kernel categorization, and short kernel detection
from AMD TraceLens (https://github.com/AMD-AGI/TraceLens).

All functions read from an existing RocpdImportData connection and return
plain dict / list structures. No output formatting. No ai_analysis imports.

Call order dependency:
    timeline = compute_interval_timeline(conn)
    categories = analyze_kernels_by_category(conn, timeline["total_wall_ns"])
    short = analyze_short_kernels(conn)
"""

import re
from typing import Any, Dict, List, Tuple

from .connection import RocinsightConnection as RocpdImportData, execute_statement

__all__ = [
    "compute_interval_timeline",
    "categorize_kernel_name",
    "analyze_kernels_by_category",
    "analyze_short_kernels",
]

# ---------------------------------------------------------------------------
# Kernel category patterns (matching TraceLens kernel_name_parser.py)
# Order matters: first match wins.
# ---------------------------------------------------------------------------
_CATEGORY_PATTERNS: List[Tuple[str, Any]] = [
    ("CONV", re.compile(r"conv|winograd|implicit_gemm_conv", re.IGNORECASE)),
    ("GEMM", re.compile(r"gemm|gemv|xdlops_gemm|Cijk_|rocblas_gemm", re.IGNORECASE)),
    (
        "SDPA",
        re.compile(
            r"flash_attention|fmha|scaled_dot_product|FlashAttention", re.IGNORECASE
        ),
    ),
    (
        "NCCL",
        re.compile(
            r"ncclKernel|rccl|AllReduce|AllGather|ReduceScatter|Broadcast",
            re.IGNORECASE,
        ),
    ),
    (
        "Elementwise",
        re.compile(
            r"vectorized_elementwise|aten_add|aten_mul|relu|gelu|silu", re.IGNORECASE
        ),
    ),
    (
        "Normalization",
        re.compile(r"layer_norm|batch_norm|group_norm|rms_norm", re.IGNORECASE),
    ),
    ("Reduction", re.compile(r"reduce|softmax|sum_|amax", re.IGNORECASE)),
]


def categorize_kernel_name(name: str) -> str:
    """Map a kernel name to a TraceLens op category.

    Returns one of: GEMM, CONV, SDPA, NCCL, Elementwise, Normalization,
    Reduction, Other.
    """
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(name):
            return category
    return "Other"


# ---------------------------------------------------------------------------
# Interval arithmetic helpers (matching TraceLens gpu_event_analyser.py)
# ---------------------------------------------------------------------------


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and merge overlapping (start, end) intervals.

    Returns a list of non-overlapping (start, end) tuples in ascending order.
    """
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _total_ns(intervals: List[Tuple[int, int]]) -> int:
    """Sum the duration of a list of non-overlapping intervals."""
    return sum(end - start for start, end in intervals)


def _subtract_intervals(
    a: List[Tuple[int, int]], b: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return intervals in *a* that do not overlap with any interval in *b*.

    Both inputs must already be merged (non-overlapping, sorted).
    Implements set difference A − B for interval sets.
    """
    result = []
    b_idx = 0
    for a_start, a_end in a:
        cur_start = a_start
        while b_idx < len(b) and b[b_idx][1] <= cur_start:
            b_idx += 1
        j = b_idx
        while j < len(b) and b[j][0] < a_end:
            b_start, b_end = b[j]
            if cur_start < b_start:
                result.append((cur_start, b_start))
            cur_start = max(cur_start, b_end)
            j += 1
        if cur_start < a_end:
            result.append((cur_start, a_end))
    return result


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def compute_interval_timeline(connection: RocpdImportData) -> Dict[str, Any]:
    """Compute accurate GPU timeline using set-theoretic interval arithmetic.

    Unlike compute_time_breakdown() which sums raw durations and double-counts
    overlapping periods, this function uses merged interval sets to compute:
    - true_compute_ns: kernel time with overlaps removed
    - exposed_memcpy_ns: memcpy time NOT overlapping any kernel
    - idle_ns: wall time minus all GPU activity

    total_wall_ns is defined as MAX(end) - MIN(start) across the union of
    kernels and memory_copies — matching compute_time_breakdown()'s definition.

    Edge cases:
    - Empty kernels table → true_compute_ns=0, true_compute_pct=0.0
    - Empty memory_copies → exposed_memcpy_ns=0, exposed_memcpy_pct=0.0
    - total_wall_ns==0  → all _pct fields return 0.0
    """
    # Load kernel intervals
    try:
        kernel_rows = execute_statement(
            connection, "SELECT start, end FROM kernels", ()
        ).fetchall()
        kernel_intervals = [
            (int(r[0]), int(r[1]))
            for r in kernel_rows
            if r[0] is not None and r[1] is not None
        ]
    except Exception:
        kernel_intervals = []

    # Load memcpy intervals
    try:
        memcpy_rows = execute_statement(
            connection, "SELECT start, end FROM memory_copies", ()
        ).fetchall()
        memcpy_intervals = [
            (int(r[0]), int(r[1]))
            for r in memcpy_rows
            if r[0] is not None and r[1] is not None
        ]
    except Exception:
        memcpy_intervals = []

    # Compute wall time across union of both tables
    all_starts = [s for s, _ in kernel_intervals] + [s for s, _ in memcpy_intervals]
    all_ends = [e for _, e in kernel_intervals] + [e for _, e in memcpy_intervals]
    if not all_starts:
        return {
            "total_wall_ns": 0,
            "true_compute_ns": 0,
            "true_compute_pct": 0.0,
            "exposed_memcpy_ns": 0,
            "exposed_memcpy_pct": 0.0,
            "idle_ns": 0,
            "idle_pct": 0.0,
        }

    total_wall_ns = max(all_ends) - min(all_starts)
    if total_wall_ns <= 0:
        return {
            "total_wall_ns": 0,
            "true_compute_ns": 0,
            "true_compute_pct": 0.0,
            "exposed_memcpy_ns": 0,
            "exposed_memcpy_pct": 0.0,
            "idle_ns": 0,
            "idle_pct": 0.0,
        }

    # Merge intervals within each set
    merged_kernels = _merge_intervals(kernel_intervals)
    merged_memcpy = _merge_intervals(memcpy_intervals)

    # Compute metrics
    true_compute_ns = _total_ns(merged_kernels)
    exposed_memcpy = _subtract_intervals(merged_memcpy, merged_kernels)
    exposed_memcpy_ns = _total_ns(exposed_memcpy)

    # Idle = wall minus union of all activity
    all_activity = _merge_intervals(merged_kernels + merged_memcpy)
    active_ns = _total_ns(all_activity)
    idle_ns = max(0, total_wall_ns - active_ns)

    def _pct(v: int) -> float:
        return round(100.0 * v / total_wall_ns, 2)

    return {
        "total_wall_ns": total_wall_ns,
        "true_compute_ns": true_compute_ns,
        "true_compute_pct": _pct(true_compute_ns),
        "exposed_memcpy_ns": exposed_memcpy_ns,
        "exposed_memcpy_pct": _pct(exposed_memcpy_ns),
        "idle_ns": idle_ns,
        "idle_pct": _pct(idle_ns),
    }


def analyze_kernels_by_category(
    connection: RocpdImportData,
    total_wall_ns: int,
) -> List[Dict[str, Any]]:
    """Aggregate kernel execution time by TraceLens op category.

    Call compute_interval_timeline() first and pass its total_wall_ns here
    so pct_of_total_time uses the same wall-time baseline.

    Returns list of dicts sorted by total_ns descending, one entry per category.
    Returns [] if kernels table is empty.

    Edge cases:
    - Empty kernels table → []
    - total_wall_ns==0   → pct_of_total_time=0.0 for all categories
    """
    try:
        rows = execute_statement(
            connection, "SELECT name, duration FROM kernels", ()
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    # Aggregate by category
    cat_totals: Dict[str, Dict[str, Any]] = {}
    total_kernel_ns = 0
    for name, duration in rows:
        if name is None or duration is None:
            continue
        category = categorize_kernel_name(str(name))
        dur = int(duration)
        total_kernel_ns += dur
        if category not in cat_totals:
            cat_totals[category] = {"count": 0, "total_ns": 0}
        cat_totals[category]["count"] += 1
        cat_totals[category]["total_ns"] += dur

    if not cat_totals:
        return []

    result = []
    for category, data in cat_totals.items():
        count = data["count"]
        total_ns = data["total_ns"]
        avg_ns = total_ns // count if count > 0 else 0
        pct_kernel = (
            round(100.0 * total_ns / total_kernel_ns, 2) if total_kernel_ns > 0 else 0.0
        )
        pct_wall = (
            round(100.0 * total_ns / total_wall_ns, 2) if total_wall_ns > 0 else 0.0
        )
        result.append(
            {
                "category": category,
                "count": count,
                "total_ns": total_ns,
                "pct_of_kernel_time": pct_kernel,
                "avg_duration_ns": avg_ns,
                "pct_of_total_time": pct_wall,
            }
        )

    return sorted(result, key=lambda x: x["total_ns"], reverse=True)


def analyze_short_kernels(
    connection: RocpdImportData,
    threshold_us: float = 10.0,
) -> Dict[str, Any]:
    """Identify kernels below threshold_us microseconds (TraceLens short-kernel analysis).

    threshold_us defaults to 10μs and is not configurable via CLI in Phase 1.

    Edge cases:
    - No kernels below threshold → short_kernel_count=0, histogram=[], top_offenders=[]
    - Empty kernels table        → same as above
    - total_kernel_time==0       → wasted_pct_of_kernel_time=0.0
    """
    threshold_ns = int(threshold_us * 1_000)

    try:
        all_rows = execute_statement(
            connection, "SELECT name, duration FROM kernels", ()
        ).fetchall()
    except Exception:
        all_rows = []

    total_kernels = len(all_rows)
    total_kernel_ns = sum(int(r[1]) for r in all_rows if r[1] is not None)

    # Filter short kernels
    short_rows = [
        (str(r[0]), int(r[1]))
        for r in all_rows
        if r[0] is not None and r[1] is not None and int(r[1]) < threshold_ns
    ]

    short_count = len(short_rows)
    wasted_ns = sum(d for _, d in short_rows)
    short_pct = (
        round(100.0 * short_count / total_kernels, 2) if total_kernels > 0 else 0.0
    )
    wasted_pct = (
        round(100.0 * wasted_ns / total_kernel_ns, 2) if total_kernel_ns > 0 else 0.0
    )

    # Histogram buckets (matching TraceLens short kernel histogram)
    buckets = [
        ("0-1μs", 0, 1_000),
        ("1-5μs", 1_000, 5_000),
        (f"5-{int(threshold_us)}μs", 5_000, threshold_ns),
    ]
    histogram = [
        {"bucket_label": label, "count": sum(1 for _, d in short_rows if lo <= d < hi)}
        for label, lo, hi in buckets
        if any(lo <= d < hi for _, d in short_rows)
    ]

    # Top offenders by total wasted time
    offender_map: Dict[str, Dict[str, Any]] = {}
    for name, dur in short_rows:
        if name not in offender_map:
            offender_map[name] = {"count": 0, "total_wasted_ns": 0}
        offender_map[name]["count"] += 1
        offender_map[name]["total_wasted_ns"] += dur

    top_offenders = sorted(
        [
            {
                "name": name,
                "count": data["count"],
                "avg_us": round(data["total_wasted_ns"] / data["count"] / 1_000, 3),
                "total_wasted_ns": data["total_wasted_ns"],
            }
            for name, data in offender_map.items()
        ],
        key=lambda x: x["total_wasted_ns"],
        reverse=True,
    )[:10]

    return {
        "threshold_us": threshold_us,
        "total_kernels": total_kernels,
        "short_kernel_count": short_count,
        "short_kernel_pct": short_pct,
        "wasted_ns": wasted_ns,
        "wasted_pct_of_kernel_time": wasted_pct,
        "histogram": histogram,
        "top_offenders": top_offenders,
    }
