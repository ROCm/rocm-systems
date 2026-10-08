# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""End-to-end checks for Memory Chart HTML written by CLI analysis."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from memory_chart.html.report import MODEL_ID

ROOT = Path(__file__).resolve().parents[2]
WORKLOADS = ROOT / "tests" / "workloads"
CLI = ROOT / "src" / "rocprof-compute"


def copy_workload(tmp_path: Path, name: str, model: str) -> Path:
    """Give each CLI invocation its own writable profile copy."""
    target = tmp_path / name / model
    shutil.copytree(WORKLOADS / name / model, target)
    return target


def analyze(*args: str) -> str:
    """Run the real analyze entry point and return its terminal output."""
    with tempfile.TemporaryDirectory() as output_dir:
        result = subprocess.run(
            [sys.executable, str(CLI), "analyze", *args],
            cwd=output_dir,
            capture_output=True,
            text=True,
            check=False,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def model_at(path: Path) -> dict:
    """Read the page model from its embedded JSON script."""
    html = path.read_text(encoding="utf-8")
    match = re.search(
        rf'<script id="{MODEL_ID}" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert match is not None
    assert "Plotly" not in html
    assert "<script src" not in html
    return json.loads(match.group(1))


def test_report_keeps_kernel_statistics_order(tmp_path: Path) -> None:
    workload = copy_workload(tmp_path, "torch_trace", "MI300X_A1")
    analyze("-p", str(workload), "-b", "3")

    model = model_at(workload / "mem_chart.html")
    with (workload / "pmc_kernel_top.csv").open(newline="", encoding="utf-8") as file:
        names = [row["Kernel_Name"] for row in csv.DictReader(file)]
    assert len(names) == 11
    assert [kernel["name"] for kernel in model["kernels"]] == names
    assert model["initialKernel"] is None
    assert model["heading"] == "3. Memory Chart"
    assert model["normalization"] == "per_kernel"
    assert model["scopeNote"] == ""
    aggregate_slots = model["aggregate"]["slots"]
    first_kernel_slots = model["kernels"][0]["view"]["slots"]
    assert any(
        aggregate_slots[slot]["text"] != first_kernel_slots[slot]["text"]
        for slot in aggregate_slots
    )


@pytest.mark.parametrize(("ids", "initial"), [(["0"], 0), (["0", "1"], None)])
def test_kernel_filter_sets_initial_selection_without_shortening_list(
    tmp_path: Path, ids: List[str], initial: Optional[int]
) -> None:
    workload = copy_workload(tmp_path, "torch_trace", "MI300X_A1")
    analyze("-p", str(workload), "-b", "3", "-k", *ids)

    model = model_at(workload / "mem_chart.html")
    assert model["initialKernel"] == initial
    assert len(model["kernels"]) == 11


def test_dispatch_and_gpu_filters_limit_report(tmp_path: Path) -> None:
    workload = copy_workload(tmp_path, "torch_trace", "MI300X_A1")
    analyze("-p", str(workload), "-b", "3", "-d", "1", "6")
    model = model_at(workload / "mem_chart_d-1_6.html")
    with (workload / "pmc_dispatch_info.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    selected = {row["Kernel_Name"] for row in rows if row["Dispatch_ID"] in {"1", "6"}}
    assert {kernel["name"] for kernel in model["kernels"]} == selected
    assert model["scopeNote"] == "dispatches 1, 6"

    analyze("-p", str(workload), "-b", "3", "--gpu-id", "0")
    assert (workload / "mem_chart_gpu-0.html").exists()


@pytest.mark.parametrize(
    ("name", "model", "layout_arch"),
    [("vcopy", "MI350", "gfx950"), ("vcopy", "RDNA35_HALO", "gfx115x")],
)
def test_supported_architectures_and_table_view(
    tmp_path: Path, name: str, model: str, layout_arch: str
) -> None:
    workload = copy_workload(tmp_path, name, model)
    analyze("-p", str(workload), "-b", "3", "--view", "table")
    assert model_at(workload / "mem_chart.html")["layout"]["arch"] == layout_arch


@pytest.mark.parametrize(
    "extra",
    [
        ("-b", "2"),
        ("--list-stats",),
        ("-b", "3", "--output-format", "csv"),
        ("-b", "3", "--output-format", "db"),
    ],
)
def test_modes_without_chart_do_not_write_html(
    tmp_path: Path, extra: Tuple[str, ...]
) -> None:
    workload = copy_workload(tmp_path, "vcopy", "MI350")
    analyze("-p", str(workload), *extra)
    assert not list(workload.glob("mem_chart*.html"))


def test_multiple_paths_write_one_report_each(tmp_path: Path) -> None:
    first = copy_workload(tmp_path, "vcopy", "MI350")
    second = copy_workload(tmp_path, "vcopy", "RDNA35_HALO")
    analyze("-p", str(first), "-p", str(second), "-b", "3")
    assert (first / "mem_chart.html").exists()
    assert (second / "mem_chart.html").exists()


def test_text_output_still_writes_html(tmp_path: Path) -> None:
    workload = copy_workload(tmp_path, "vcopy", "MI350")
    analyze("-p", str(workload), "-b", "3", "--output-format", "txt")
    assert (workload / "mem_chart.html").exists()


def test_pc_sampling_only_does_not_write_html(tmp_path: Path) -> None:
    workload = copy_workload(tmp_path, "vcopy_pc_sampling_only", "MI350")
    analyze("-p", str(workload), "-b", "21")
    assert not list(workload.glob("mem_chart*.html"))


def test_terminal_output_is_unchanged_before_saved_report(tmp_path: Path) -> None:
    """Writing the report preserves the terminal chart and table output."""
    workload = copy_workload(tmp_path, "vcopy", "MI350")
    script = (
        "import runpy, sys; "
        "import rocprof_compute_analyze.analysis_cli as cli; "
        "cli.write_mem_chart_report = lambda *args: None; "
        "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    baseline = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(CLI),
            "analyze",
            "-p",
            str(workload),
            "-b",
            "3",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    actual = analyze("-p", str(workload), "-b", "3")
    assert (
        re.sub(r"^.*Saved HTML report:.*\n", "", actual, flags=re.MULTILINE)
        == baseline.stdout
    )
