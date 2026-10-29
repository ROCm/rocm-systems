# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""End-to-end checks for Memory Chart HTML written by CLI analysis."""

import csv
import json
import re
from itertools import count
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import common
import pandas as pd
import pytest

from memory_chart.html.report import MODEL_ID
from tests.integration import common as integration_common

WORKLOADS = Path(common.ROOT).resolve() / "tests" / "workloads"


@pytest.fixture
def workload_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[str, str], Path]:
    """Create isolated workload copies in the temporary analysis directory."""
    monkeypatch.chdir(tmp_path)
    copy_ids = count()

    def make_workload(name: str, model: str) -> Path:
        source = WORKLOADS / name / model
        workload_dir = integration_common.setup_workload_dir(
            str(source), param_id=f"{name}_{model}_{next(copy_ids)}"
        )
        return Path(workload_dir).resolve()

    return make_workload


@pytest.fixture
def multi_kernel_workload(workload_factory) -> Path:
    """Give recorded vcopy dispatches distinct names for multi-kernel reports."""
    workload = workload_factory("vcopy", "MI350")

    for counter_path in workload.glob("results_*.csv.gz"):
        counters = pd.read_csv(counter_path)
        counters["Kernel_Name"] = "kernel_" + counters["Dispatch_ID"].astype(str)
        counters["Kernel_ID"] = counters["Dispatch_ID"]
        common.write_gzip_csv(counter_path, counters.to_csv(index=False))

    return workload


def model_at(path: Path) -> dict:
    """Read the page model from its embedded JSON script."""
    html = path.read_text(encoding="utf-8")
    match = re.search(
        rf'<script id="{MODEL_ID}" type="application/json">(.*?)</script>',
        html,
        re.DOTALL,
    )

    assert match is not None

    return json.loads(match.group(1))


def test_report_is_standalone_document(
    binary_handler_analyze_rocprof_compute, workload_factory
) -> None:
    """Embed the report's assets in a complete standalone HTML document."""
    workload = workload_factory("vcopy", "MI350")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
    ])
    assert code == 0

    html = (workload / "mem_chart.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "<style>" in html
    assert "<script>" in html


def test_report_keeps_kernel_statistics_order(
    binary_handler_analyze_rocprof_compute, multi_kernel_workload
) -> None:
    """Preserve kernel statistics order in the generated report."""
    workload = multi_kernel_workload
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
    ])
    assert code == 0

    model = model_at(workload / "mem_chart.html")

    with (workload / "pmc_kernel_top.csv").open(newline="", encoding="utf-8") as file:
        names = [row["Kernel_Name"] for row in csv.DictReader(file)]

    assert len(names) == 3
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


@pytest.mark.parametrize(
    ("ids", "initial"),
    [
        pytest.param(["0"], 0, id="single-kernel-selected"),
        pytest.param(["0", "1"], None, id="multiple-kernels-show-aggregate"),
    ],
)
def test_kernel_filter_sets_initial_selection_without_shortening_list(
    binary_handler_analyze_rocprof_compute,
    multi_kernel_workload,
    ids: List[str],
    initial: Optional[int],
) -> None:
    """Set the initial kernel selection without removing other kernels."""
    workload = multi_kernel_workload
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
        "-k",
        *ids,
    ])
    assert code == 0

    model = model_at(workload / "mem_chart.html")
    assert model["initialKernel"] == initial
    assert len(model["kernels"]) == 3


def test_dispatch_filter_limits_report(
    binary_handler_analyze_rocprof_compute, multi_kernel_workload
) -> None:
    """Apply dispatch filters to report contents and filenames."""
    workload = multi_kernel_workload
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
        "-d",
        "1",
        "2",
    ])
    assert code == 0

    model = model_at(workload / "mem_chart_d-1_2.html")

    with (workload / "pmc_dispatch_info.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    selected = {row["Kernel_Name"] for row in rows if row["Dispatch_ID"] in {"1", "2"}}

    assert {kernel["name"] for kernel in model["kernels"]} == selected
    assert model["scopeNote"] == "dispatches 1, 2"


def test_gpu_filter_scopes_report(
    binary_handler_analyze_rocprof_compute, workload_factory
) -> None:
    """Include the selected GPU in the report's filename and scope note."""
    workload = workload_factory("vcopy", "MI350")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
        "--gpu-id",
        "0",
    ])
    assert code == 0

    model = model_at(workload / "mem_chart_gpu-0.html")
    assert model["scopeNote"] == "GPU 0"


@pytest.mark.parametrize(
    ("name", "model", "layout_arch"),
    [
        pytest.param("vcopy", "MI350", "gfx950", id="cdna4-table-view"),
        pytest.param("vcopy", "RDNA35_HALO", "gfx115x", id="rdna35-table-view"),
    ],
)
def test_supported_architectures_and_table_view(
    binary_handler_analyze_rocprof_compute,
    workload_factory,
    name: str,
    model: str,
    layout_arch: str,
) -> None:
    """Render table reports for supported GPU architecture families."""
    workload = workload_factory(name, model)
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
        "--view",
        "table",
    ])
    assert code == 0

    assert model_at(workload / "mem_chart.html")["layout"]["arch"] == layout_arch


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param(("-b", "2"), id="non-memory-chart-block"),
        pytest.param(("--list-stats",), id="list-stats"),
        pytest.param(("-b", "3", "--output-format", "csv"), id="csv-output"),
        pytest.param(("-b", "3", "--output-format", "db"), id="database-output"),
    ],
)
def test_modes_without_chart_do_not_write_html(
    binary_handler_analyze_rocprof_compute, workload_factory, extra: Tuple[str, ...]
) -> None:
    """Skip report generation for modes that do not produce memory charts."""
    workload = workload_factory("vcopy", "MI350")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        *extra,
    ])
    assert code == 0

    assert not list(workload.glob("mem_chart*.html"))


def test_multiple_paths_write_one_report_each(
    binary_handler_analyze_rocprof_compute, workload_factory
) -> None:
    """Write one HTML report for each analyzed workload path."""
    first = workload_factory("vcopy", "MI350")
    second = workload_factory("vcopy", "RDNA35_HALO")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(first),
        "-p",
        str(second),
        "-b",
        "3",
    ])
    assert code == 0

    assert (first / "mem_chart.html").exists()
    assert (second / "mem_chart.html").exists()


def test_text_output_still_writes_html(
    binary_handler_analyze_rocprof_compute, workload_factory
) -> None:
    """Write the HTML report when terminal output uses the text format."""
    workload = workload_factory("vcopy", "MI350")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "3",
        "--output-format",
        "txt",
    ])
    assert code == 0

    assert (workload / "mem_chart.html").exists()


def test_pc_sampling_only_does_not_write_html(
    binary_handler_analyze_rocprof_compute, workload_factory
) -> None:
    """Skip HTML report generation for PC-sampling-only analysis."""
    workload = workload_factory("vcopy_pc_sampling_only", "MI350")
    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "-p",
        str(workload),
        "-b",
        "21",
    ])
    assert code == 0

    assert not list(workload.glob("mem_chart*.html"))


@pytest.mark.parametrize(
    "view",
    [pytest.param(None, id="chart-view"), pytest.param("table", id="table-view")],
)
def test_terminal_output_coexists_with_html_generation(
    binary_handler_analyze_rocprof_compute,
    workload_factory,
    capsys,
    view: Optional[str],
) -> None:
    """Display terminal chart or table output alongside the saved HTML report."""
    workload = workload_factory("vcopy", "MI350")
    arguments = ["analyze", "-p", str(workload), "-b", "3"]

    if view is not None:
        arguments.extend(["--view", view])

    code = binary_handler_analyze_rocprof_compute(arguments)
    assert code == 0

    output = capsys.readouterr().out
    assert "3. Memory Chart" in output

    if view is None:
        assert "VL1D" in output
        assert "HBM" in output
    else:
        assert "Metric" in output
        assert "HBM Read BW" in output

    assert f"Saved HTML report: {workload / 'mem_chart.html'}" in output
    assert (workload / "mem_chart.html").exists()
