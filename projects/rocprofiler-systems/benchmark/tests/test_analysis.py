# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math
import statistics

import pytest
from bench import analysis
from bench.runner import RunResult


def run(wall_ms, cpu_s=0.0, checksum="x", exit=0, i=0, error=None):
    return RunResult(
        "c",
        i,
        exit=exit,
        wall_ns=int(wall_ms * 1e6),
        user_s=cpu_s,
        checksum=checksum,
        error=error,
    )


def test_stats_median_p95_mad():
    s = analysis.stats(list(range(1, 101)))
    assert s["median"] == 50.5
    assert s["p95"] == 95
    assert (s["min"], s["max"]) == (1, 100)
    assert s["mad"] == 25


def test_decision_rule_resolvable_and_noise():
    base = analysis.summarize([run(w) for w in (10, 11, 9, 12, 8)])
    big = analysis.summarize([run(w) for w in (20, 21, 19, 22, 18)])
    tiny = analysis.summarize([run(w) for w in (10.5, 11.5, 9.5, 12.5, 8.5)])
    assert analysis.compare(base, big)["resolvable_from_noise"]
    assert not analysis.compare(base, tiny)["resolvable_from_noise"]
    assert analysis.compare(base, big)["wall_pct"] == pytest.approx(100.0)


def test_failed_runs_go_to_errors_not_stats():
    results = {"A": [run(10), run(10, exit=1, error="exit code 1", i=1)]}
    out = analysis.analyze(results)
    assert len(out["errors"]) == 1 and out["errors"][0]["run"] == 1
    assert out["summary"]["A"]["wall"]["max"] == 10e6


def test_checksum_mismatch_warns():
    out = analysis.analyze({"A": [run(1, checksum="a")], "B": [run(1, checksum="b")]})
    assert any("checksum" in w for w in out["warnings"])


def test_drift_detected():
    drift = analysis.check_drift([run(w) for w in (10, 10, 10, 10, 50, 50, 50, 50)])
    assert drift["drift"]
    assert not analysis.check_drift([run(10)] * 3)["checked"]


def test_app_reported_metrics_are_summarized_and_compared():
    base = [run(10, i=i) for i in range(6)]
    tool = [run(10, i=i) for i in range(6)]
    for r in base:
        r.metrics = {"gpu_kernel_us": 10.0}
    for r in tool:
        r.metrics = {"gpu_kernel_us": 15.0}
    out = analysis.analyze({"A": base, "B": tool})
    assert out["summary"]["A"]["gpu_kernel_us"]["median"] == 10.0
    cmp = out["comparison"]["B_vs_A"]["app_metrics"]["gpu_kernel_us"]
    assert cmp["abs"] == 5.0 and cmp["pct"] == 50.0


def test_metric_unit_from_name_suffix():
    from bench.text_report import metric_unit

    assert [
        metric_unit(n)
        for n in ("gpu_kernel_us", "gpu_interval_ms", "x_ns", "t_s", "m_kb", "plain")
    ] == ["µs", "ms", "ns", "s", "KB", ""]


def test_is_real_has_floor_when_baseline_mad_is_zero():
    base = {"median": 1000.0, "mad": 0.0, "n": 100}
    assert not analysis.is_real(base, {"median": 1000.5, "mad": 0.0, "n": 100})
    assert analysis.is_real(base, {"median": 1020.0, "mad": 0.0, "n": 100})


def test_is_real_uses_standard_error_not_per_run_scatter():
    # Real total-CPU row: delta 169.7, per-run MAD ~110-124, n=150 -> SE-based threshold ~50.
    base = {"median": 3059.62, "mad": 108.93, "n": 150}
    assert analysis.is_real(base, {"median": 3229.33, "mad": 124.11, "n": 150})
    assert not analysis.is_real(base, {"median": 3034.0, "mad": 122.41, "n": 150})


def test_pct_is_nan_for_near_zero_baseline():
    assert math.isnan(analysis.pct(40e6, 0.2e6, analysis.MIN_PCT_BASE_NS))
    assert analysis.pct(10e6, 100e6, analysis.MIN_PCT_BASE_NS) == pytest.approx(10.0)


def test_stats_sum_mean_sample_stddev():
    values = list(range(1, 101))
    s = analysis.stats(values)
    assert (s["n"], s["sum"], s["mean"]) == (100, 5050, 50.5)
    assert s["stddev"] == pytest.approx(statistics.stdev(values))


def test_stats_stddev_is_nan_for_single_run():
    s = analysis.stats([5.0])
    assert (s["sum"], s["mean"]) == (5.0, 5.0)
    assert math.isnan(s["stddev"])


def test_compare_mean_overhead_is_separate_from_median():
    # One slow outlier moves the mean but not the median.
    base = analysis.summarize([run(w) for w in (10, 10, 10, 10, 10)])
    tool = analysis.summarize([run(w) for w in (10, 10, 10, 10, 60)])
    cmp = analysis.compare(base, tool)
    assert cmp["wall_pct"] == 0
    assert cmp["wall_mean_abs_ns"] == pytest.approx(10e6)
    assert cmp["wall_mean_pct"] == pytest.approx(100.0)


def test_compare_app_metric_mean_overhead():
    base, tool = [run(10, i=i) for i in range(4)], [run(10, i=i) for i in range(4)]
    for r, v in zip(base, (10.0, 10.0, 10.0, 10.0)):
        r.metrics = {"gpu_kernel_us": v}
    for r, v in zip(tool, (10.0, 10.0, 10.0, 50.0)):
        r.metrics = {"gpu_kernel_us": v}
    cmp = analysis.analyze({"A": base, "B": tool})["comparison"]["B_vs_A"]["app_metrics"]
    assert cmp["gpu_kernel_us"]["mean_abs"] == 10.0
    assert cmp["gpu_kernel_us"]["mean_pct"] == 100.0
