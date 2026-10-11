# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the two-directory metric health report."""

import sys
from pathlib import Path

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
    root = write_artifact_dir(tmp_path / "candidate", "spp", 10.0)

    with pytest.raises(SystemExit, match=r"no \*_develop\.log"):
        report.discover_run_artifacts(
            root, ARCH, "develop", "rocprofiler-compute-develop", ["vcopy"]
        )


def test_report_compares_candidate_dir_against_develop_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = write_artifact_dir(tmp_path / "candidate", "spp", 10.0)
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
            "--candidate",
            str(candidate),
            "--baseline",
            str(baseline),
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
