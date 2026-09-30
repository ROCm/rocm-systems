# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for aggregate and per-kernel HTML memory chart values."""

from argparse import Namespace
from collections import OrderedDict
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import pytest

from membw_analysis.models import (
    BottleneckNode,
    MemBwAnalysisResult,
    SupportingMetric,
)
from memory_chart.html import views
from memory_chart.mem_chart import (
    format_scientific,
    format_value,
    plot_mem_chart,
    progress_bar,
    strip_ansi,
)
from utils.metrics.expression import build_eval_string
from utils.schema import ArchConfig, Workload

SYSINFO = Path(__file__).resolve().parents[2] / "workloads/vcopy/MI350/sysinfo.csv"


def make_inputs(
    *,
    kernel_filter: Optional[List[Union[int, str]]] = None,
    with_membw: bool = False,
) -> Tuple[Workload, ArchConfig, Namespace]:
    """Make small counter frames and matching architecture templates."""
    workload = Workload()
    workload.sys_info = pd.read_csv(SYSINFO)
    workload.raw_pmc = pd.DataFrame({
        "Kernel_Name": ["alpha", "beta", "alpha", "beta", "gamma", "beta"],
        "GPU_ID": [0, 0, 1, 0, 0, 0],
        "Dispatch_ID": [1, 2, 3, 4, 5, 0],
        "Start_Timestamp": [0, 0, 0, 0, 0, 0],
        "End_Timestamp": [100, 200, 300, 400, 500, 600],
        "Counter": [1.234, 2.234, 100, 4.234, 9, 999],
        "Bandwidth": [1.2e9, 1.0e9, 99e9, 0.5e9, 0, 99e9],
        "Hit": [10, 20, 99, 30, 40, 99],
        "Allocation": [2048, 4096, 99, 6144, 8192, 999],
        "GRBM_GUI_ACTIVE": [1, 1, 1, 1, 1, 1],
    })
    workload.filter_gpu_ids = [0]
    workload.filter_dispatch_ids = ["> 0"]
    workload.filter_kernel_ids = kernel_filter or []
    workload.dfs[1] = pd.DataFrame({"Kernel_Name": ["beta", "alpha", "gamma"]})
    chart_df = pd.DataFrame({
        "Metric": [
            "Wavefront Occupancy",
            "VL1 Hit",
            "VL1_L2 Read BW",
            "LDS Allocation",
            "Flat Read",
        ],
        "Value": list(
            map(
                build_eval_string,
                [
                    "SUM(Counter)",
                    "SUM(Hit)",
                    "SUM(Bandwidth)",
                    "SUM(Allocation)",
                    "SUM(Counter)",
                ],
            )
        ),
    })
    config = ArchConfig()
    config.dfs[301] = chart_df
    config.dfs_type[301] = "metric_table"
    workload.dfs[301] = chart_df.copy()
    sources = [
        {
            "metric_table": {
                "id": 301,
                "title": "Memory Chart",
                "header": {},
            }
        }
    ]
    config.panel_configs = OrderedDict({
        300: {"data source": sources},
    })
    if with_membw:
        bw_df = pd.DataFrame({
            "Metric": ["Synthetic stall"],
            "Avg": [build_eval_string("SUM(Counter)")],
        })
        config.dfs[3001] = bw_df
        config.dfs_type[3001] = "metric_table"
        workload.dfs[3001] = bw_df.copy()
        config.panel_configs[3000] = {
            "data source": [
                {
                    "metric_table": {
                        "id": 3001,
                        "title": "Memory bandwidth",
                        "header": {},
                    }
                }
            ]
        }
    args = Namespace(
        time_unit=None,
        cols=None,
        include_cols=[],
        decimal=2,
        verbose=0,
        report_diff=0,
        debug=False,
    )
    return workload, config, args


def slot(view: views.ChartView, metric: str) -> views.SlotValue:
    return next(item for item in view.slots if item.metric == metric)


def make_bottleneck(value: float, active: bool) -> MemBwAnalysisResult:
    """Return one result whose active leaf carries the view's counter value."""
    node = BottleneckNode(
        id="synthetic",
        label="TCP stall",
        level="GL1",
        state="active" if active else "inactive",
        supporting=(
            SupportingMetric(
                key="Synthetic stall",
                value=value,
                unit="Percent",
                display=f"{value:.1f}%",
            ),
        ),
        children=(),
    )
    return MemBwAnalysisResult(
        arch="gfx950",
        availability="full",
        availability_reason=None,
        nodes=(node,),
        guidance_blocks=(f"TCP stall\n  Measured: {value:.1f}%",) if active else (),
    )


def test_views_filter_raw_rows_and_keep_duration_sorted_kernel_list() -> None:
    workload, config, args = make_inputs(kernel_filter=[1])
    result = views.build_chart_views(workload, config, args)

    assert result.arch == "gfx950"
    assert result.diagram["arch"] == "gfx950"
    assert [(entry.index, entry.name) for entry in result.kernels] == [
        (0, "beta"),
        (1, "alpha"),
        (2, "gamma"),
    ]
    assert [view.key for view in result.kernel_views] == [
        "kernel-0",
        "kernel-1",
        "kernel-2",
    ]
    assert result.initial_kernel == 1
    assert result.aggregate.membw is None
    assert slot(result.aggregate, "Wavefront Occupancy").text == "16.7"
    assert [slot(view, "Wavefront Occupancy").text for view in result.kernel_views] == [
        "6.5",
        "1.2",
        "9.0",
    ]
    assert workload.dfs[301]["Value"].iloc[0] == build_eval_string("SUM(Counter)")
    with pytest.raises(FrozenInstanceError):
        result.aggregate.key = "changed"


def test_processed_tables_and_slots_follow_terminal_formatting() -> None:
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)
    aggregate = result.aggregate
    table = aggregate.tables[0]

    assert table.table_id == 301
    assert table.columns == ("Metric", "Value")
    assert table.rows[0] == ("Wavefront Occupancy", 16.7)
    assert slot(aggregate, "VL1_L2 Read BW").text == format_value(2.7e9, "Bytes/s")
    assert slot(aggregate, "VL1 Hit").text == format_value(100, "%")
    assert slot(aggregate, "VL1 Hit").bar == progress_bar(100)
    assert slot(aggregate, "LDS Allocation").text == format_value(20, " KB")
    assert slot(aggregate, "Wavefront Occupancy").unit_label == "waves/CU"
    assert slot(aggregate, "Flat Read").text == format_scientific(16.7)
    assert slot(aggregate, "VL1_L2 Write BW").text == "N/A"
    assert slot(aggregate, "L2 Hit").bar == progress_bar(None)
    assert result.initial_kernel is None

    rendered = strip_ansi(
        plot_mem_chart(
            {row[0]: row[1] for row in table.rows},
            chart_title="Chart",
            gpu_arch="gfx950",
        )
    )
    assert slot(aggregate, "VL1_L2 Read BW").text in rendered
    assert slot(aggregate, "LDS Allocation").text in rendered


def test_column_selection_and_missing_tables_produce_na_slots() -> None:
    workload, config, args = make_inputs()
    args.cols = [0]
    selected = views.build_chart_views(workload, config, args)
    assert selected.aggregate.tables[0].columns == ("Metric",)
    assert slot(selected.aggregate, "VL1 Hit").text == "N/A"

    del workload.dfs[301]
    missing = views.build_chart_views(workload, config, args)
    assert missing.aggregate.tables == ()
    assert all(item.text == "N/A" for item in missing.aggregate.slots)


@pytest.mark.parametrize(
    "filter_ids,expected",
    [
        ([], None),
        ([0], 0),
        ([0, 1], None),
        (["beta"], 0),
    ],
)
def test_initial_kernel_requires_one_effective_match(filter_ids, expected) -> None:
    workload, config, args = make_inputs(kernel_filter=filter_ids)
    result = views.build_chart_views(workload, config, args)
    assert result.initial_kernel == expected


def test_each_view_has_its_own_membw_annotations_and_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, config, args = make_inputs(with_membw=True)
    loaded: List[str] = []

    def load_spec(arch: str) -> object:
        loaded.append(arch)
        return object()

    def evaluate_membw(dfs: dict, _spec: object, _arch: str) -> MemBwAnalysisResult:
        value = float(dfs[3001]["Avg"].iloc[0])
        return make_bottleneck(value, value > 5)

    monkeypatch.setattr(views, "load_membw_spec", load_spec)
    monkeypatch.setattr(views, "evaluate_membw_dfs", evaluate_membw)
    result = views.build_chart_views(workload, config, args)

    assert loaded == ["gfx950"]
    assert [
        view.membw.availability for view in (result.aggregate, *result.kernel_views)
    ] == ["full"] * 4
    assert result.aggregate.membw.annotations[0].block_id == "vl1d"
    assert result.aggregate.membw.annotations[0].label == "[!] TCP stall"
    assert result.aggregate.membw.annotations[0].value == format_value(16.702, "%")
    assert result.aggregate.membw.guidance_blocks == ("TCP stall\n  Measured: 16.7%",)
    assert result.kernel_views[0].membw.annotations[0].value == format_value(6.468, "%")
    assert result.kernel_views[1].membw.annotations == ()
    assert result.kernel_views[1].membw.status == (
        "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA)."
    )
    rich = strip_ansi(
        plot_mem_chart(
            {},
            chart_title="Chart",
            gpu_arch="gfx950",
            membw=make_bottleneck(16.702, True),
        )
    )
    assert "[!] TCP stall" in rich
    assert result.aggregate.membw.annotations[0].value in rich
