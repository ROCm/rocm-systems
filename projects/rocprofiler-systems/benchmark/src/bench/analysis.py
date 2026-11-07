# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Statistics, baseline comparison and decision rule over in-memory runs."""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable, Sequence

from bench.runner import RunResult

NOISE_FACTOR = 2.0
# Standard error of a median is ~1.2533 * sigma / sqrt(n); sigma ~ 1.4826 * MAD (normal data).
MEDIAN_SE_FACTOR = 1.2533 * 1.4826
# Floor for the noise threshold: a baseline MAD near 0 (e.g. constant RSS) must not make
# every tiny difference "real".
MIN_REAL_FRACTION = 0.01
MIN_DRIFT_RUNS = 4
# Time baselines below 1 ms get no percentage (init/fini without the tool are near zero).
MIN_PCT_BASE_NS = 1e6

METRICS: dict[str, Callable[[RunResult], float]] = {
    "wall": lambda r: r.wall_ns,
    "cpu": lambda r: r.user_s + r.sys_s,
    "max_rss_kb": lambda r: r.max_rss_kb,
}


def mad(values: Sequence[float]) -> float:
    med = statistics.median(values)
    return statistics.median(abs(v - med) for v in values)


def p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def stats(values: Sequence[float]) -> dict[str, float]:
    # sample stddev (n - 1): undefined for a single run
    stddev = statistics.stdev(values) if len(values) > 1 else math.nan
    return {
        "median": statistics.median(values),
        "p95": p95(values),
        "min": min(values),
        "max": max(values),
        "mad": mad(values),
        "n": len(values),
        "sum": sum(values),
        "mean": statistics.fmean(values),
        "stddev": stddev,
    }


PHASES = {
    "init": "t0_t1_ns",
    "work": "t1_t2_ns",
    "fini": "t2_t3_ns",
    "cpu_init": "cpu_t0_t1_ns",
    "cpu_work": "cpu_t1_t2_ns",
    "cpu_fini": "cpu_t2_t3_ns",
}
# Built-in summary keys; every other summary key is an app-reported metric.
KNOWN_KEYS = {*METRICS, *PHASES}
# Summary keys compare() reports in detail: (name, unit suffix of the abs keys, % floor).
DELTAS = (
    ("wall", "ns", MIN_PCT_BASE_NS),
    ("cpu", "s", MIN_PCT_BASE_NS * 1e-9),
    ("work", "ns", MIN_PCT_BASE_NS),
    ("cpu_work", "ns", MIN_PCT_BASE_NS),
)


def app_metric_names(runs: Sequence[RunResult]) -> list[str]:
    """Names reported by every run, excluding the built-in metric keys."""
    names = set.intersection(*(set(r.metrics) for r in runs)) if runs else set()
    return sorted(names - KNOWN_KEYS)


def summarize(runs: Sequence[RunResult]) -> dict[str, dict[str, float]]:
    ok = [r for r in runs if r.ok]
    if not ok:
        return {}
    out = {m: stats([get(r) for r in ok]) for m, get in METRICS.items()}
    for name, attr in PHASES.items():
        values = [getattr(r, attr) for r in ok]
        if all(v is not None for v in values):
            out[name] = stats(values)
    for name in app_metric_names(ok):
        out[name] = stats([r.metrics[name] for r in ok])
    return out


def is_real(base: dict[str, float], other: dict[str, float]) -> bool:
    """Difference of medians exceeds max(2 * its standard error, 1% of the base median)."""
    se_diff = MEDIAN_SE_FACTOR * math.hypot(
        base["mad"] / math.sqrt(base["n"]), other["mad"] / math.sqrt(other["n"])
    )
    threshold = max(NOISE_FACTOR * se_diff, MIN_REAL_FRACTION * abs(base["median"]))
    return abs(other["median"] - base["median"]) > threshold


def pct(diff: float, base: float, floor: float = 0.0) -> float:
    """diff as % of base; NaN when |base| <= floor (a near-zero base gives meaningless %)."""
    return diff / base * 100 if abs(base) > floor else math.nan


def compare(
    base: dict[str, dict[str, float]], other: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Overhead of `other` vs `base`; real only per is_real()."""
    out: dict[str, Any] = {}
    for name, unit, floor in DELTAS:
        if name in base and name in other:
            b, o = base[name], other[name]
            diff, diff_p95 = o["median"] - b["median"], o["p95"] - b["p95"]
            diff_mean = o["mean"] - b["mean"]
            # the wall verdict key has no prefix: it is the headline result
            verdict = (
                "resolvable_from_noise"
                if name == "wall"
                else f"{name}_resolvable_from_noise"
            )
            out.update(
                {
                    f"{name}_abs_{unit}": diff,
                    f"{name}_pct": pct(diff, b["median"], floor),
                    f"{name}_p95_abs_{unit}": diff_p95,
                    f"{name}_p95_pct": pct(diff_p95, b["p95"], floor),
                    f"{name}_mean_abs_{unit}": diff_mean,
                    f"{name}_mean_pct": pct(diff_mean, b["mean"], floor),
                    verdict: is_real(b, o),
                }
            )
    out["phases_abs_ns"] = {
        n: other[n]["median"] - base[n]["median"]
        for n in PHASES
        if n in base and n in other
    }
    out["app_metrics"] = {
        name: {
            "abs": other[name]["median"] - base[name]["median"],
            "pct": pct(
                other[name]["median"] - base[name]["median"], base[name]["median"]
            ),
            "p95_abs": other[name]["p95"] - base[name]["p95"],
            "mean_abs": other[name]["mean"] - base[name]["mean"],
            "mean_pct": pct(other[name]["mean"] - base[name]["mean"], base[name]["mean"]),
            "resolvable_from_noise": is_real(base[name], other[name]),
        }
        for name in sorted(set(base) & set(other))
        if name not in KNOWN_KEYS
    }
    return out


def check_checksums(results: dict[str, list[RunResult]]) -> list[str]:
    seen = {r.checksum for runs in results.values() for r in runs if r.ok}
    if len(seen) > 1:
        return [f"checksum mismatch across runs/commands: {sorted(map(str, seen))}"]
    return []


def check_drift(runs: Sequence[RunResult]) -> dict[str, Any]:
    """Compare wall median of the first half of runs with the second half."""
    walls = [r.wall_ns for r in runs if r.ok]
    if len(walls) < MIN_DRIFT_RUNS:
        return {"drift": False, "checked": False}
    half = len(walls) // 2
    first = statistics.median(walls[:half])
    second = statistics.median(walls[half:])
    diff = abs(second - first)
    return {
        "drift": diff > 0 and diff >= NOISE_FACTOR * mad(walls),
        "checked": True,
        "first_half_median": first,
        "second_half_median": second,
    }


def collect_errors(results: dict[str, list[RunResult]]) -> list[dict[str, Any]]:
    return [
        {"command": name, "run": r.index, "exit": r.exit, "error": r.error}
        for name, runs in results.items()
        for r in runs
        if not r.ok
    ]


def analyze(results: dict[str, list[RunResult]]) -> dict[str, Any]:
    """results: command name -> runs in execution order; first = baseline."""
    names = list(results)
    summary = {n: summarize(results[n]) for n in names}
    base = names[0]
    comparison = {
        f"{n}_vs_{base}": compare(summary[base], summary[n])
        for n in names[1:]
        if summary[n] and summary[base]
    }
    warnings = check_checksums(results)
    drift = {n: check_drift(results[n]) for n in names}
    warnings += [f"drift detected in {n}" for n, d in drift.items() if d["drift"]]
    return {
        "summary": summary,
        "comparison": comparison,
        "drift": drift,
        "warnings": warnings,
        "errors": collect_errors(results),
    }
