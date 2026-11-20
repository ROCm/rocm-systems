# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""analyze writes the memory chart it prints as an HTML page."""

import re
from pathlib import Path
from typing import Callable

import common
import pandas as pd
import pytest

from memory_chart.loader import Layouts
from memory_chart.mem_chart_html import MODEL_ID
from tests.integration import common as integration_common
from utils.utils_common import strip_ansi

VCOPY_WORKLOADS = sorted(
    str(path)
    for path in Path("tests/workloads/vcopy").iterdir()
    if (path / "sysinfo.csv").exists()
)


def embedded_model(page: Path) -> dict:
    """The JSON model of a memory chart page."""
    return common.embedded_model(page.read_text(encoding="utf-8"), MODEL_ID)


def pages(workload_dir: str) -> list:
    return sorted(path.name for path in Path(workload_dir).glob("mem_chart*.html"))


@pytest.mark.parametrize("workload", VCOPY_WORKLOADS)
def test_analyze_writes_the_chart_it_prints(
    workload: str,
    binary_handler_analyze_rocprof_compute: Callable[[list], int],
    capsys: pytest.CaptureFixture,
) -> None:
    """The page holds the values the terminal chart shows, on every arch."""
    gpu_arch = pd.read_csv(Path(workload) / "sysinfo.csv")["gpu_arch"][0]
    workload_dir = integration_common.setup_workload_dir(workload)
    try:
        analyze = binary_handler_analyze_rocprof_compute
        assert analyze(["analyze", "-p", workload_dir]) == 0
        output = strip_ansi(capsys.readouterr().out)

        if Layouts.for_arch(gpu_arch) is None:
            assert pages(workload_dir) == []
            return
        assert pages(workload_dir) == ["mem_chart_gpu-0.html"]
        page = Path(workload_dir) / "mem_chart_gpu-0.html"
        saved = re.search(r"Interactive memory chart saved to (\S+)", output)
        assert saved is not None
        assert Path(saved.group(1)).resolve() == page.resolve()
        for shown in embedded_model(page)["views"]["aggregate"]["values"].values():
            assert shown["text"] in output
    finally:
        common.clean_output_dir(True, workload_dir)


def test_analyze_names_the_page_like_roofline_and_writes_it_only_with_the_chart(
    binary_handler_analyze_rocprof_compute: Callable[[list], int],
) -> None:
    """-k names the page; table view and comparisons, which draw no chart, don't
    write one."""
    first = integration_common.setup_workload_dir("tests/workloads/vcopy/MI350")
    second = integration_common.setup_workload_dir(
        "tests/workloads/vcopy/MI350", suffix="_second"
    )
    try:
        analyze = binary_handler_analyze_rocprof_compute
        assert analyze(["analyze", "-p", first, "-k", "0"]) == 0
        assert pages(first) == ["mem_chart_gpu-0_0.html"]

        assert analyze(["analyze", "-p", first, "--debug-table-view"]) == 0
        assert analyze(["analyze", "-p", first, "-p", second]) == 0
        assert pages(first) == ["mem_chart_gpu-0_0.html"]
        assert pages(second) == []
    finally:
        common.clean_output_dir(True, first)
        common.clean_output_dir(True, second)
