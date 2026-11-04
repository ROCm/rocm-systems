# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the two-directory metric health report."""

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from tools import generate_metric_health_report as report

ARCH = "gfx942"
METRIC_LINE = "| 2.1.0 | VALU FLOPs | {value} | Gflop |"
MI300X_WORKLOAD = (
    Path(__file__).resolve().parents[1] / "workloads" / "path" / "MI300X_A1"
)
# MIN(SQ_INSTS / SQ_WAVES): 327680 / 16384 in every dispatch of the fixture.
INSTS_PER_WAVE_KEY = "7.2.2|Instructions per wavefront"
INSTS_PER_WAVE = 20.0
PERCENT_KEY = "2.1.9|CP Busy"
PERCENT_FORMULAS = {PERCENT_KEY: ("", "Percent", False)}


def make_log(values: Dict[str, Optional[float]]) -> report.ParsedLog:
    """Build a parsed analyze log from metric key -> Median value."""
    return report.ParsedLog(
        metrics={
            key: report.MetricValue(
                value=value, raw="N/A" if value is None else f"{value:g}"
            )
            for key, value in values.items()
        }
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
        "Median",
    )


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


def write_artifact_dir(root: Path, run: str, value: float) -> Path:
    """Write one health artifact dir with a vcopy analyze log and sysinfo."""
    log_dir = root / "logs" / ARCH
    log_dir.mkdir(parents=True)
    (log_dir / f"vcopy_{run}.log").write_text(
        "| Metric_ID | Metric | Value | Unit |\n"
        + METRIC_LINE.format(value=value)
        + "\n",
        encoding="utf-8",
    )
    workload_dir = root / ARCH / f"vcopy_261011-{ARCH}-500_{run}" / "MI300X_A1"
    workload_dir.mkdir(parents=True)
    (workload_dir / "sysinfo.csv").write_text("gpu_arch\ngfx942\n", encoding="utf-8")
    return root


def append_metric_lines(root: Path, run: str, values: List[float]) -> None:
    """Append metric rows that produce distinct delta instances."""
    log_path = root / "logs" / ARCH / f"vcopy_{run}.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        for index, value in enumerate(values, start=1):
            log_file.write(f"| 2.1.{index} | Metric {index} | {value} | Gflop |\n")


def test_discover_finds_log_and_workload_dir(tmp_path: Path) -> None:
    root = write_artifact_dir(tmp_path / "develop", "develop", 5.0)

    run = report.discover_run_artifacts(
        root, ARCH, "develop", "rocprofiler-compute-develop", ["vcopy", "nbody"]
    )

    assert run.logs == {"vcopy": root / "logs" / ARCH / "vcopy_develop.log"}
    assert run.workload_dirs == {
        "vcopy": root / ARCH / f"vcopy_261011-{ARCH}-500_develop" / "MI300X_A1"
    }


def test_discover_missing_baseline_dir_is_clear_error(tmp_path: Path) -> None:
    missing = tmp_path / "no-such-run"

    with pytest.raises(SystemExit, match="artifact directory not found"):
        report.discover_run_artifacts(
            missing, ARCH, "develop", "rocprofiler-compute-develop", ["vcopy"]
        )


def test_discover_dir_without_run_logs_is_clear_error(tmp_path: Path) -> None:
    root = write_artifact_dir(tmp_path / "candidate", "single-pass", 10.0)

    with pytest.raises(SystemExit, match=r"no \*_develop\.log"):
        report.discover_run_artifacts(
            root, ARCH, "develop", "rocprofiler-compute-develop", ["vcopy"]
        )


def test_report_compares_candidate_dir_against_baseline_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = write_artifact_dir(tmp_path / "candidate", "single-pass", 10.0)
    baseline = write_artifact_dir(tmp_path / "develop", "develop", 5.0)
    append_metric_lines(candidate, "single-pass", [10.0, 10.0])
    append_metric_lines(baseline, "develop", [5.0, 5.0])
    out = tmp_path / "health.html"
    monkeypatch.setattr(report, "compute_medians_for_workload", lambda *_: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_metric_health_report.py",
            "--arch",
            ARCH,
            "-p",
            str(baseline),
            "-p",
            str(candidate),
            "--workloads",
            "vcopy",
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
    baseline = write_artifact_dir(tmp_path / "develop", "develop", 5.0)
    out = tmp_path / "health.html"
    monkeypatch.setattr(report, "compute_medians_for_workload", lambda *_: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_metric_health_report.py",
            "--arch",
            ARCH,
            "-p",
            str(baseline),
            "--workloads",
            "vcopy",
            "--out",
            str(out),
        ],
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
            "--arch",
            ARCH,
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
    note = report.build_total_note(
        arch,
        arch,
        {"vcopy": {"total": 12}},
        12,
        "Median",
    )

    assert "<code>$total_l2_chan</code>" in note
    assert "l2_banks × num_xcd" in note
    assert "profile sysinfo" in note
    assert "expanded row count" in note
    lowered = note.lower()
    assert "reconcil" not in lowered
    assert "16 channel" not in lowered
    assert "cpx" not in lowered
    assert "cu_per_gpu" not in lowered


def test_recompute_medians_on_real_workload() -> None:
    """Median recompute runs the analyze filters and evaluator end to end."""
    medians = report.compute_medians_for_workload(MI300X_WORKLOAD, [])

    assert medians[INSTS_PER_WAVE_KEY] == pytest.approx(INSTS_PER_WAVE)


@pytest.mark.parametrize("numerator_pass", [0, 1])
def test_recompute_medians_keep_divisor_on_numerator_pass(
    tmp_path: Path, numerator_pass: int
) -> None:
    """A counter copied across passes is read from the numerator's pass."""
    workload = write_two_pass_workload(tmp_path / "workload", numerator_pass)

    medians = report.compute_medians_for_workload(workload, [])

    assert medians[INSTS_PER_WAVE_KEY] == pytest.approx(INSTS_PER_WAVE)


def test_f64_valu_flops_zero_is_optional_path() -> None:
    assert report.classify_all_workload_zero("VALU FLOPs (F64)", "") == "zero_optional"


def test_valu_flops_zero_is_sdk_bug_suspect() -> None:
    assert report.classify_all_workload_zero("VALU FLOPs", "") == "zero_sdk_bug"


def test_na_candidate_is_not_an_overflow_improvement() -> None:
    data = report.compute_report(
        {"vcopy": make_log({PERCENT_KEY: None})},
        {"vcopy": make_log({PERCENT_KEY: 150.0})},
        PERCENT_FORMULAS,
    )

    assert data.improvements == []


def test_real_candidate_value_is_an_overflow_improvement() -> None:
    data = report.compute_report(
        {"vcopy": make_log({PERCENT_KEY: 80.0})},
        {"vcopy": make_log({PERCENT_KEY: 150.0})},
        PERCENT_FORMULAS,
    )

    assert [(c.workload, c.key) for c in data.improvements] == [("vcopy", PERCENT_KEY)]


def test_delta_fraction_counts_unique_metrics(tmp_path: Path) -> None:
    """One metric changing in three workloads is one changed metric."""
    workloads = ["vcopy", "nbody", "mega_kernel"]
    data = report.compute_report(
        {wl: make_log({"2.1.0|VALU FLOPs": 10.0}) for wl in workloads},
        {wl: make_log({"2.1.0|VALU FLOPs": 5.0}) for wl in workloads},
        {},
    )

    page = render(data, tmp_path)

    assert len(data.deltas) == 3
    assert "3 rows, 1 of 1 metrics (100.0%)" in page
    assert "300.0%" not in page


def test_baseline_only_metric_is_reported_missing(tmp_path: Path) -> None:
    lost_key = "2.1.1|VALU IOPs"
    data = report.compute_report(
        {"vcopy": make_log({"2.1.0|VALU FLOPs": 5.0})},
        {"vcopy": make_log({"2.1.0|VALU FLOPs": 5.0, lost_key: 3.0})},
        {},
    )

    page = render(data, tmp_path)

    assert [(c.workload, c.key, c.new) for c in data.missing] == [
        ("vcopy", lost_key, None)
    ]
    assert data.all_keys == ["2.1.0|VALU FLOPs"]
    assert "Missing from this branch vs rocprofiler-compute-develop" in page
    assert "VALU IOPs" in page


def test_compute_report_without_baseline_has_no_comparison() -> None:
    data = report.compute_report(
        {"vcopy": make_log({"2.1.0|VALU FLOPs": 5.0})}, None, {}
    )

    assert data.summary["vcopy"]["total"] == 1
    assert data.summary["vcopy"]["good"] == 1
    assert (data.improvements, data.deltas, data.missing) == ([], [], [])
