"""regression — Gate 4 verdict tool with 'hot kernel' definition.

See spec §5 Gate 4: hot kernels = top-K covering 80% cumulative
UNION with any kernel ≥ 3% individually. Catches both "dominant
kernel" and "long-tail-of-medium-kernels" regressions.

Tool class: READ_ONLY (pure analysis, no modification).
"""

import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class


@dataclass(frozen=True)
class KernelRuntime:
    """Simple kernel runtime snapshot for testing and analysis."""
    kernel_name: str
    total_runtime_ns: int
    share: float


HOT_COVERAGE_PCT = 0.80
HOT_INDIVIDUAL_PCT = 0.03
REGRESSION_THRESHOLD_PCT = 0.10    # a hot kernel > 10% worse = regression


def _kernel_durations(db_path: str) -> Dict[str, float]:
    """Total duration_ns per kernel name.

    Handles both legacy (duration_ns column) and rocprofv3 (end-start) schemas.
    """
    conn = sqlite3.connect(db_path)

    # Find the kernel_dispatch table (may have UUID suffix)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch%'"
    )
    kt_result = cursor.fetchone()
    if not kt_result:
        conn.close()
        return {}

    kernel_table = kt_result[0]

    # Extract UUID from kernel_table name
    # Format is rocpd_kernel_dispatch_<UUID>
    # UUID is everything after the underscore following "dispatch_"
    uuid = kernel_table.replace("rocpd_kernel_dispatch_", "")

    info_kernel_table = f"rocpd_info_kernel_symbol_{uuid}"

    # Try to query using real rocprofv3 schema first (end-start + info_kernel join)
    try:
        rows = conn.execute(f"""
            SELECT iks.display_name, SUM(kd.end - kd.start) as total_ns
            FROM {kernel_table} kd
            JOIN {info_kernel_table} iks ON kd.kernel_id = iks.id
            GROUP BY kd.kernel_id
            ORDER BY total_ns DESC
        """).fetchall()
        conn.close()
        return {name: float(total) for name, total in rows}
    except Exception as e:
        # Fall back to legacy schema
        try:
            rows = conn.execute(f"""
                SELECT name, SUM(duration_ns) AS total_ns
                FROM {kernel_table}
                GROUP BY name
                ORDER BY total_ns DESC
            """).fetchall()
            conn.close()
            return {name: float(total) for name, total in rows}
        except Exception:
            conn.close()
            return {}


@tool_class(ToolClass.READ_ONLY)
def identify_hot_kernels(db_path: str) -> List[Dict[str, Any]]:
    """Return list of 'hot' kernels per Gate 4 definition.

    hot = top-K covering 80% cumulative runtime, UNION kernels ≥ 3% individually.

    Returns:
        List of {"name": str, "total_ns": float, "pct_total": float} sorted by runtime.
    """
    durations = _kernel_durations(db_path)
    total = sum(durations.values())
    if total == 0:
        return []

    # Annotate + sort
    ranked = [
        {"name": n, "total_ns": d, "pct_total": d / total}
        for n, d in durations.items()
    ]
    ranked.sort(key=lambda x: x["total_ns"], reverse=True)

    # Top-K covering 80%
    covered = 0.0
    top_k = []
    for r in ranked:
        top_k.append(r)
        covered += r["pct_total"]
        if covered >= HOT_COVERAGE_PCT:
            break

    # Union with kernels ≥ 3%
    big_individuals = [r for r in ranked if r["pct_total"] >= HOT_INDIVIDUAL_PCT]

    # Merge by name
    seen = {k["name"] for k in top_k}
    hot = list(top_k)
    for r in big_individuals:
        if r["name"] not in seen:
            hot.append(r)
            seen.add(r["name"])

    hot.sort(key=lambda x: x["total_ns"], reverse=True)
    return hot


@tool_class(ToolClass.READ_ONLY)
def compare_runs(
    db_before: str,
    db_after: str,
    threshold_pct: float = 3.0,
) -> Dict[str, Any]:
    """Compare two rocpd databases; return regression verdict.

    Verdict is rule-based (spec §5 Gate 4):
    - "improved" if total runtime reduced > threshold_pct AND no hot kernel regressed > 10%
    - "regressed" if any hot kernel regressed > 10% OR weighted geomean regressed
    - "neutral" otherwise (within noise)

    Args:
        db_before: baseline DB path.
        db_after: post-optimization DB path.
        threshold_pct: noise threshold in percent (e.g., 3.0 = 3%).

    Returns:
        {
            "verdict": "improved"|"regressed"|"neutral",
            "total_delta_pct": float (negative = improvement),
            "weighted_geomean_delta_pct": float,
            "per_kernel_deltas": [{"kernel": str, "delta_pct": float, "was_hot": bool}, ...],
            "threshold_pct": threshold_pct / 100,
        }
    """
    before = _kernel_durations(db_before)
    after = _kernel_durations(db_after)

    total_before = sum(before.values())
    total_after = sum(after.values())
    if total_before == 0:
        raise ValueError("baseline has zero total runtime")

    total_delta = (total_after - total_before) / total_before
    threshold = threshold_pct / 100.0

    # Determine hot kernels from baseline
    hot = {k["name"] for k in identify_hot_kernels(db_before)}

    # Per-kernel deltas (union of kernel names)
    all_kernels = set(before) | set(after)
    per_kernel = []
    geomean_logs = []
    hot_regression_triggered = False
    for k in sorted(all_kernels):
        b = before.get(k, 0)
        a = after.get(k, 0)
        if b == 0:
            continue  # new kernel — can't compute delta
        delta = (a - b) / b
        per_kernel.append({"kernel": k, "delta_pct": delta, "was_hot": k in hot})

        # Weighted geomean: each kernel weighted by its baseline runtime share
        weight = b / total_before
        if a > 0:
            geomean_logs.append(weight * math.log(a / b))

        if k in hot and delta > REGRESSION_THRESHOLD_PCT:
            hot_regression_triggered = True

    weighted_geomean = math.exp(sum(geomean_logs)) - 1 if geomean_logs else 0

    # Verdict
    if hot_regression_triggered:
        verdict = "regressed"
    elif abs(total_delta) < threshold and abs(weighted_geomean) < threshold:
        verdict = "neutral"
    elif total_delta < -threshold:
        verdict = "improved"
    elif weighted_geomean > threshold:
        # Long-tail regression
        verdict = "regressed"
    else:
        verdict = "neutral"

    return {
        "verdict": verdict,
        "total_delta_pct": total_delta,
        "weighted_geomean_delta_pct": weighted_geomean,
        "per_kernel_deltas": per_kernel,
        "threshold_pct": threshold,
    }


@tool_class(ToolClass.READ_ONLY)
def extract_kernel_runtimes_from_db(db_path: str) -> List[KernelRuntime]:
    """Extract kernel runtimes from a rocpd database.

    Returns:
        List of KernelRuntime ordered by total_runtime_ns (descending).
    """
    durations = _kernel_durations(db_path)
    total = sum(durations.values())
    if total == 0:
        return []

    result = []
    for kernel_name, total_ns in sorted(durations.items(), key=lambda x: x[1], reverse=True):
        result.append(
            KernelRuntime(
                kernel_name=kernel_name,
                total_runtime_ns=int(total_ns),
                share=total_ns / total,
            )
        )
    return result
