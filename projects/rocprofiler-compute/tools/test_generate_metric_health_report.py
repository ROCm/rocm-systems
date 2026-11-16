# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the metric health report."""

import csv
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from tools import generate_metric_health_report as report

ARCH = "gfx942"
MI300X_WORKLOAD = (
    Path(__file__).resolve().parents[1] / "tests" / "workloads" / "path" / "MI300X_A1"
)
# The only kernel in the MI300X fixture.
FIXTURE_KERNEL = "vecCopy(double*, double*, double*, int, int) (.kd)"
# MIN(SQ_INSTS / SQ_WAVES): 327680 / 16384 in every dispatch of the fixture.
INSTS_PER_WAVE_KEY = "7.2.2|Instructions per wavefront"
INSTS_PER_WAVE = 20.0
PERCENT_KEY = "2.1.9|CP Busy"
PERCENT_FORMULAS = {PERCENT_KEY: ("", "Percent", False)}
KERNEL_COLUMNS = ["kernel_uuid", "kernel_name", "dispatch_count", "duration_ns_sum"]
METRIC_COLUMNS = [
    "kernel_uuid",
    "metric_id",
    "metric_name",
    "unit",
    "value_name",
    "value",
]


def make_metrics(values: Dict[str, Optional[float]]) -> report.WorkloadMetrics:
    """Build one workload's metrics from metric key -> Median value."""
    return report.WorkloadMetrics(
        metrics={key: report.MetricValue(value=value) for key, value in values.items()}
    )


def make_run(root: Path, label: str) -> report.RunArtifacts:
    return report.RunArtifacts(label=label, artifact_dir=root)


def render(data: report.ReportData, tmp_path: Path) -> str:
    """Render comparison data with fixed run labels."""
    return report.render_report(
        data,
        make_run(tmp_path / "candidate", "this branch"),
        make_run(tmp_path / "develop", "rocprofiler-compute-develop"),
        {},
        "2026-01-01",
        "host",
        ARCH,
        ARCH,
        None,
    )


def write_csv(path: Path, columns: List[str], rows: List[Tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(rows)


def write_run_dir(root: Path, values: List[float]) -> Path:
    """Write one run with a vcopy workload: sysinfo plus an analyze CSV export.

    Metric 2.1.N gets values[N] in its Value column.
    """
    profile_dir = root / "vcopy" / "profile"
    profile_dir.mkdir(parents=True)
    (profile_dir / "sysinfo.csv").write_text(f"gpu_arch\n{ARCH}\n", encoding="utf-8")
    analyze_dir = root / "vcopy" / "analyze"
    write_csv(analyze_dir / "kernel.csv", KERNEL_COLUMNS, [(1, "copy", 4, 100)])
    write_csv(
        analyze_dir / "kernel_metric.csv",
        METRIC_COLUMNS,
        [
            (1, f"2.1.{index}", f"Metric {index}", "Gflop", "Value", value)
            for index, value in enumerate(values)
        ],
    )
    return root


def write_two_pass_workload(dest: Path, numerator_pass: int) -> Path:
    """Copy the MI300X workload and split SQ_INSTS / SQ_WAVES across two passes.

    SQ_WAVES is collected in both passes. The pass holding SQ_INSTS keeps the
    real SQ_WAVES; the other pass doubles it, so binding the numerator to the
    wrong pass's divisor halves the metric.
    """
    shutil.copytree(MI300X_WORKLOAD, dest)
    source = dest / "results_pmc_perf_0.csv.gz"
    rows = pd.read_csv(source)
    source.unlink()
    names = rows["Counter_Name"]
    waves = rows[names == "SQ_WAVES"]
    decoy_waves = waves.assign(Counter_Value=waves["Counter_Value"] * 2)
    numerator_rows = pd.concat([rows[names == "SQ_INSTS"], waves])
    other_rows = pd.concat([
        rows[~names.isin(["SQ_INSTS", "SQ_WAVES"])],
        decoy_waves,
    ])
    passes = {numerator_pass: numerator_rows, 1 - numerator_pass: other_rows}
    for index, frame in passes.items():
        frame.to_csv(dest / f"results_pmc_perf_{index}.csv.gz", index=False)
    return dest


def skip_workload_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the parts that need a real profiled workload."""
    monkeypatch.setattr(
        report, "compute_dispatch_stats_for_workload", lambda *_: ({}, {})
    )
    monkeypatch.setattr(report, "collect_formulas", lambda *_: {})


def test_discover_finds_profile_and_csv_dirs(tmp_path: Path) -> None:
    root = write_run_dir(tmp_path / "develop", [5.0])

    run = report.discover_run(root, "rocprofiler-compute-develop")

    assert run.profile_dirs == {"vcopy": root / "vcopy" / "profile"}
    assert run.csv_dirs == {"vcopy": root / "vcopy" / "analyze"}


def test_discover_missing_run_dir_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="run directory not found"):
        report.discover_run(tmp_path / "no-such-run", "rocprofiler-compute-develop")


def test_discover_workload_without_analyze_csv_is_clear_error(tmp_path: Path) -> None:
    root = write_run_dir(tmp_path / "develop", [5.0])
    shutil.rmtree(root / "vcopy" / "analyze")

    with pytest.raises(SystemExit, match="needs profile/"):
        report.discover_run(root, "rocprofiler-compute-develop")


def test_read_top_kernel_metrics_reads_median_and_std_dev(tmp_path: Path) -> None:
    write_csv(tmp_path / "kernel.csv", KERNEL_COLUMNS, [(1, "copy", 4, 100)])
    write_csv(
        tmp_path / "kernel_metric.csv",
        METRIC_COLUMNS,
        [
            (1, "2.1.21", "Write BW", "Gb/s", "Median", 0.072),
            (1, "2.1.21", "Write BW", "Gb/s", "Std Dev", 0.055),
        ],
    )

    result = report.read_top_kernel_metrics(tmp_path)
    metric = result.metrics["2.1.21|Write BW"]

    assert metric.value == pytest.approx(0.072)
    assert metric.std_v == pytest.approx(0.055)
    assert metric.source == "median"
    assert result.units == {"2.1.21|Write BW": "Gb/s"}


def test_read_top_kernel_metrics_uses_longest_kernel(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "kernel.csv",
        KERNEL_COLUMNS,
        [(1, "short", 9, 100), (2, "long", 4, 300)],
    )
    write_csv(
        tmp_path / "kernel_metric.csv",
        METRIC_COLUMNS,
        [
            (1, "2.1.0", "Busy", "Percent", "Value", 10.0),
            (2, "2.1.0", "Busy", "Percent", "Value", 30.0),
        ],
    )

    result = report.read_top_kernel_metrics(tmp_path)

    assert (result.kernel_name, result.dispatch_count) == ("long", 4)
    assert result.metrics["2.1.0|Busy"].value == pytest.approx(30.0)


def test_avg_only_metric_waits_for_recompute() -> None:
    metric = report.metric_value_from_columns({"Avg": "7.5", "Min": "1", "Max": "9"})

    assert metric.value is None
    assert (metric.min_v, metric.max_v) == (1.0, 9.0)


def test_report_compares_candidate_dir_against_baseline_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = write_run_dir(tmp_path / "candidate", [10.0, 10.0, 10.0])
    baseline = write_run_dir(tmp_path / "develop", [5.0, 5.0, 5.0])
    out = tmp_path / "health.html"
    skip_workload_reads(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_metric_health_report.py",
            "-p",
            str(baseline),
            "-p",
            str(candidate),
            "--out",
            str(out),
        ],
    )

    report.main()

    page = out.read_text(encoding="utf-8")
    assert "Value deltas vs rocprofiler-compute-develop" in page
    assert "<b>Candidate:</b> this branch" in page
    assert "100.0%" in page


def test_report_for_one_path_omits_comparison_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = write_run_dir(tmp_path / "develop", [5.0])
    out = tmp_path / "health.html"
    skip_workload_reads(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_metric_health_report.py", "-p", str(baseline), "--out", str(out)],
    )

    report.main()

    page = out.read_text(encoding="utf-8")
    assert "Value deltas vs" not in page
    assert "<b>Run:</b> rocprofiler-compute-develop" in page


def test_report_rejects_more_than_two_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_metric_health_report.py",
            "-p",
            str(tmp_path / "baseline"),
            "-p",
            str(tmp_path / "candidate"),
            "-p",
            str(tmp_path / "third"),
            "--out",
            str(tmp_path / "health.html"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        report.main()

    assert "accepts at most two paths" in capsys.readouterr().err


@pytest.mark.parametrize("arch", ["gfx942"])
def test_cdna3_total_note_counts_reported_channels(arch: str) -> None:
    """The CDNA3 total note describes the counted channel expansion."""
    note = report.build_total_note(arch, arch, {"vcopy": {"total": 12}}, 12)

    assert "<code>$total_l2_chan</code>" in note
    assert "l2_banks × num_xcd" in note
    assert "profile sysinfo" in note
    assert "Total counts each of those rows" in note


def test_formulas_use_expanded_channel_names() -> None:
    """Block 18 rows are keyed as analyze writes them and keep their formulas."""
    arch_config, _sys_info = report.load_workload_metric_tables(MI300X_WORKLOAD)

    formulas = report.metric_formulas(arch_config)

    assert "TCC_HIT[0]" in formulas["18.2.0|Channel 0"]
    assert any(
        report.uses_level_counter(formula)
        for key, formula in formulas.items()
        if key.startswith("18.")
    )


def test_recompute_medians_on_real_workload() -> None:
    """Median recompute runs the analyze filters and evaluator end to end."""
    medians, _stddevs = report.compute_dispatch_stats_for_workload(
        MI300X_WORKLOAD, FIXTURE_KERNEL
    )

    assert medians[INSTS_PER_WAVE_KEY] == pytest.approx(INSTS_PER_WAVE)


def test_recompute_for_unknown_kernel_is_clear_error() -> None:
    with pytest.raises(SystemExit, match="has no dispatches"):
        report.compute_dispatch_stats_for_workload(MI300X_WORKLOAD, "other")


@pytest.mark.parametrize("numerator_pass", [0, 1])
def test_recompute_medians_keep_divisor_on_numerator_pass(
    tmp_path: Path, numerator_pass: int
) -> None:
    """A counter copied across passes is read from the numerator's pass."""
    workload = write_two_pass_workload(tmp_path / "workload", numerator_pass)

    medians, _stddevs = report.compute_dispatch_stats_for_workload(
        workload, FIXTURE_KERNEL
    )

    assert medians[INSTS_PER_WAVE_KEY] == pytest.approx(INSTS_PER_WAVE)


def test_f64_valu_flops_zero_is_optional_path() -> None:
    assert report.classify_all_workload_zero("VALU FLOPs (F64)", "") == "zero_optional"


def test_valu_flops_zero_is_sdk_bug_suspect() -> None:
    assert report.classify_all_workload_zero("VALU FLOPs", "") == "zero_sdk_bug"


def test_na_candidate_is_not_an_overflow_improvement() -> None:
    data = report.compute_report(
        {"vcopy": make_metrics({PERCENT_KEY: None})},
        {"vcopy": make_metrics({PERCENT_KEY: 150.0})},
        PERCENT_FORMULAS,
    )

    assert data.improvements == []


def test_real_candidate_value_is_an_overflow_improvement() -> None:
    data = report.compute_report(
        {"vcopy": make_metrics({PERCENT_KEY: 80.0})},
        {"vcopy": make_metrics({PERCENT_KEY: 150.0})},
        PERCENT_FORMULAS,
    )

    assert [(c.workload, c.key) for c in data.improvements] == [("vcopy", PERCENT_KEY)]


def test_one_value_delta_is_rendered(tmp_path: Path) -> None:
    """A single regression still gets the value-delta section."""
    data = report.compute_report(
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0})},
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 10.0})},
        {},
    )

    page = render(data, tmp_path)

    assert len(data.deltas) == 1
    assert "Value deltas vs rocprofiler-compute-develop" in page


def test_empty_deltas_omit_the_section(tmp_path: Path) -> None:
    """Equal values leave the value-delta section out."""
    data = report.compute_report(
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0})},
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0})},
        {},
    )

    page = render(data, tmp_path)

    assert data.deltas == []
    assert "Value deltas vs" not in page


def test_delta_fraction_counts_unique_metrics(tmp_path: Path) -> None:
    """One metric changing in three workloads is one changed metric."""
    workloads = ["vcopy", "nbody", "mega_kernel"]
    data = report.compute_report(
        {wl: make_metrics({"2.1.0|VALU FLOPs": 10.0}) for wl in workloads},
        {wl: make_metrics({"2.1.0|VALU FLOPs": 5.0}) for wl in workloads},
        {},
    )

    page = render(data, tmp_path)

    assert len(data.deltas) == 3
    assert "3 rows, 1 of 1 metrics (100.0%)" in page
    assert "300.0%" not in page


def test_baseline_only_metric_is_reported_missing(tmp_path: Path) -> None:
    lost_key = "2.1.1|VALU IOPs"
    data = report.compute_report(
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0})},
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0, lost_key: 3.0})},
        {},
    )

    page = render(data, tmp_path)

    assert [(c.workload, c.key, c.new) for c in data.missing] == [
        ("vcopy", lost_key, None)
    ]
    assert data.all_keys == ["2.1.0|VALU FLOPs"]
    assert "Missing from this branch vs rocprofiler-compute-develop" in page
    assert "VALU IOPs" in page


def test_sum_ratio_uses_per_dispatch_median_and_std() -> None:
    expression = "to_sum(a) / to_sum(b)"

    assert report.per_dispatch_stat_expr(expression, "median") == "to_median((a) / (b))"
    assert report.per_dispatch_stat_expr(expression, "stddev") == "to_std((a) / (b))"


def test_avg_formula_uses_per_dispatch_median_and_std() -> None:
    """AVG(SQC_LDS_BANK_CONFLICT) becomes to_avg(...) and gets per-dispatch stats."""
    expression = "to_avg(raw_pmc_df['SQC_LDS_BANK_CONFLICT'])"

    assert report.per_dispatch_stat_expr(expression, "median") == (
        "to_median((raw_pmc_df['SQC_LDS_BANK_CONFLICT']))"
    )
    assert report.per_dispatch_stat_expr(expression, "stddev") == (
        "to_std((raw_pmc_df['SQC_LDS_BANK_CONFLICT']))"
    )


def test_max_and_collect_sum_stay_without_per_dispatch_stats() -> None:
    """Max and COLLECT_SUM are not rewritten; Min still is."""
    assert report.per_dispatch_stat_expr("MAX(SQ_INSTS)", "median") is None
    assert report.per_dispatch_stat_expr("COLLECT_SUM(a)", "stddev") is None
    assert report.per_dispatch_stat_expr("MIN(SQ_INSTS)", "median") == (
        "to_median(SQ_INSTS)"
    )


def test_mixed_sum_and_min_has_no_per_dispatch_std() -> None:
    assert report.per_dispatch_stat_expr("to_min(to_sum(a))", "stddev") is None


def test_cv_is_stddev_over_absolute_median() -> None:
    metric = report.MetricValue(value=0.072, std_v=0.055)

    assert report.fmt_cv(metric) == "76.4%"


def test_cv_is_blank_when_median_is_near_zero() -> None:
    metric = report.MetricValue(value=0.0, std_v=1.0)

    assert report.fmt_cv(metric) == "—"


def test_delta_row_shows_stddev_and_cv(tmp_path: Path) -> None:
    key = "2.1.21|L2-Fabric Write BW"
    workloads = ["vcopy", "nbody", "mega_kernel"]
    old = report.MetricValue(value=0.072, source="median", std_v=0.055)
    new = report.MetricValue(value=0.036, source="median", std_v=0.018)
    data = report.compute_report(
        {wl: report.WorkloadMetrics(metrics={key: new}) for wl in workloads},
        {wl: report.WorkloadMetrics(metrics={key: old}) for wl in workloads},
        {key: ("", "Gb/s", False)},
    )

    page = render(data, tmp_path)

    assert "Old (Std Dev)" in page
    assert "Old (CV)" in page
    assert "76.4%" in page
    assert "50.0%" in page


def test_recompute_stddevs_on_real_workload() -> None:
    """Constant per-dispatch values have a sample standard deviation of zero."""
    _medians, stddevs = report.compute_dispatch_stats_for_workload(
        MI300X_WORKLOAD, FIXTURE_KERNEL
    )

    assert stddevs[INSTS_PER_WAVE_KEY] == pytest.approx(0.0)


def test_compute_report_without_baseline_has_no_comparison() -> None:
    data = report.compute_report(
        {"vcopy": make_metrics({"2.1.0|VALU FLOPs": 5.0})}, None, {}
    )

    assert data.summary["vcopy"]["total"] == 1
    assert data.summary["vcopy"]["good"] == 1
    assert (data.improvements, data.deltas, data.missing) == ([], [], [])
