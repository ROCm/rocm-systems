# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the two-directory metric health report."""

import sys
from pathlib import Path
from typing import List

import pytest

from tools import generate_metric_health_report as report

ARCH = "gfx942"
METRIC_LINE = "| 2.1.0 | VALU FLOPs | {value} | Gflop |"


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


@pytest.mark.parametrize("arch", ["gfx940", "gfx941", "gfx942"])
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
