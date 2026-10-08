# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for aggregate and per-kernel HTML memory chart values."""

from argparse import Namespace
from collections import OrderedDict
from dataclasses import FrozenInstanceError
from typing import List, Optional, Tuple, Union

import pandas as pd
import pytest

from membw_analysis.models import (
    BottleneckNode,
    MemBwAnalysisResult,
    SupportingMetric,
)
from memory_chart.html import views
from memory_chart.mem_chart import progress_bar
from memory_chart.units import display_unit, format_value
from utils.metrics.expression import build_eval_string
from utils.schema import ArchConfig, Workload


def make_inputs(
    *,
    kernel_filter: Optional[List[Union[int, str]]] = None,
    with_membw: bool = False,
) -> Tuple[Workload, ArchConfig, Namespace]:
    """Make small counter frames and matching architecture templates."""
    workload = Workload()
    workload.sys_info = pd.DataFrame([
        {
            "gpu_arch": "gfx950",
            "ip_blocks": "compute",
            "se_per_gpu": 1,
            "pipes_per_gpu": 1,
            "cu_per_gpu": 1,
            "simd_per_cu": 1,
            "sqc_per_gpu": 1,
            "lds_banks_per_cu": 1,
            "cur_sclk": 1.0,
            "cur_mclk": 1.0,
            "max_mclk": 1.0,
            "max_sclk": 1.0,
            "max_waves_per_cu": 1,
            "wave_size": 1,
            "total_l2_chan": 1,
            "num_xcd": 1,
        }
    ])
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
                "metric": {
                    "Wavefront Occupancy": {"unit": "Workgroups per CU"},
                    "VL1 Hit": {"unit": "Percent"},
                    "VL1_L2 Read BW": {"unit": "Bytes/s"},
                    "VL1_L2 Write BW": {"unit": "Bytes/s"},
                    "LDS Allocation": {"unit": "Bytes per Workgroup"},
                    "Flat Read": {"unit": "(Requests + $normUnit)"},
                    "L2 Hit": {"unit": "Percent"},
                },
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
    """Return the slot with the requested metric from a chart view."""
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


def test_filtered_rows_drive_aggregate_and_per_kernel_values() -> None:
    """Build aggregate and per-kernel values from filtered counter rows."""
    workload, config, args = make_inputs(kernel_filter=[1])
    result = views.build_chart_views(workload, config, args)

    assert slot(result.aggregate, "Wavefront Occupancy").text == "16.7"
    assert [slot(view, "Wavefront Occupancy").text for view in result.kernel_views] == [
        "6.5",
        "1.2",
        "9.0",
    ]


def test_kernel_views_follow_workload_kernel_order() -> None:
    """Keep per-kernel views in the workload's kernel order."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)

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


def test_chart_views_report_workload_architecture() -> None:
    """Expose the workload architecture in chart and diagram data."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)

    assert result.arch == "gfx950"
    assert result.diagram["arch"] == "gfx950"


def test_chart_view_is_immutable() -> None:
    """Prevent callers from mutating a constructed chart view."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)

    with pytest.raises(FrozenInstanceError):
        result.aggregate.key = "changed"


def test_build_chart_views_preserves_metric_template() -> None:
    """Leave configured metric expressions unchanged during evaluation."""
    workload, config, args = make_inputs()
    views.build_chart_views(workload, config, args)

    assert workload.dfs[301]["Value"].iloc[0] == build_eval_string("SUM(Counter)")


def test_views_without_membw_have_no_membw_details() -> None:
    """Omit bandwidth details when no bandwidth table is configured."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)

    assert all(view.membw is None for view in (result.aggregate, *result.kernel_views))


def test_processed_table_contains_evaluated_metric_rows() -> None:
    """Preserve the evaluated metric table's identity, columns, and rows."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)
    table = result.aggregate.tables[0]

    assert table.table_id == 301
    assert table.columns == ("Metric", "Value")
    assert table.rows[0] == ("Wavefront Occupancy", 16.7)


def test_slots_use_terminal_metric_formatting() -> None:
    """Format slot values, units, and bars like the terminal chart."""
    workload, config, args = make_inputs()
    result = views.build_chart_views(workload, config, args)
    aggregate = result.aggregate

    assert slot(aggregate, "VL1_L2 Read BW").text == format_value(
        2.7e9, display_unit("Bytes/s")
    )
    assert slot(aggregate, "VL1 Hit").text == format_value(100, display_unit("Percent"))
    assert slot(aggregate, "VL1 Hit").bar == progress_bar(100)
    assert slot(aggregate, "LDS Allocation").text == format_value(
        20480, display_unit("Bytes per Workgroup")
    )
    assert slot(aggregate, "Wavefront Occupancy").unit_label == ""
    assert slot(aggregate, "Flat Read").text == format_value(
        16.7, display_unit("(Requests + $normUnit)")
    )
    assert slot(aggregate, "VL1_L2 Write BW").text == "N/A"
    assert slot(aggregate, "L2 Hit").bar == progress_bar(None)


def test_column_selection_limits_processed_table_and_slots() -> None:
    """Show only selected columns and mark omitted slot values unavailable."""
    workload, config, args = make_inputs()
    args.cols = [0]
    selected = views.build_chart_views(workload, config, args)
    assert selected.aggregate.tables[0].columns == ("Metric",)
    assert slot(selected.aggregate, "VL1 Hit").text == "N/A"


def test_missing_table_produces_unavailable_slots() -> None:
    """Show unavailable slots when a configured metric table is missing."""
    workload, config, args = make_inputs()
    del workload.dfs[301]
    missing = views.build_chart_views(workload, config, args)
    assert missing.aggregate.tables == ()
    assert all(item.text == "N/A" for item in missing.aggregate.slots)


@pytest.mark.parametrize(
    "filter_ids,expected",
    [
        ([], None),
        ([0], 0),
        ([1], 1),
        ([0, 1], None),
        (["beta"], 0),
        ([0, 0], 0),
        ([99], None),
        (["missing"], None),
    ],
)
def test_initial_kernel_requires_one_effective_match(filter_ids, expected) -> None:
    """Select an initial kernel only when filters resolve to one kernel."""
    workload, config, args = make_inputs(kernel_filter=filter_ids)
    result = views.build_chart_views(workload, config, args)
    assert result.initial_kernel == expected


def test_membw_spec_is_loaded_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load one bandwidth tree spec for all views in a chart build."""
    workload, config, args = make_inputs(with_membw=True)
    loaded: List[str] = []

    def load_spec(arch: str) -> object:
        """Record the architecture and return a stand-in tree spec."""
        loaded.append(arch)
        return object()

    def evaluate_membw(dfs: dict, _spec: object, _arch: str) -> MemBwAnalysisResult:
        """Build a synthetic bottleneck result from the evaluated table value."""
        value = float(dfs[3001]["Avg"].iloc[0])
        return make_bottleneck(value, value > 5)

    monkeypatch.setattr(views, "load_membw_spec", load_spec)
    monkeypatch.setattr(views, "evaluate_membw_dfs", evaluate_membw)
    views.build_chart_views(workload, config, args)

    assert loaded == ["gfx950"]


def test_each_view_has_its_own_membw_annotations_and_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep bandwidth annotations and guidance specific to each chart view."""
    workload, config, args = make_inputs(with_membw=True)

    def load_spec(arch: str) -> object:
        """Return a stand-in tree spec for the requested architecture."""
        return object()

    def evaluate_membw(dfs: dict, _spec: object, _arch: str) -> MemBwAnalysisResult:
        """Build a synthetic bottleneck result from the evaluated table value."""
        value = float(dfs[3001]["Avg"].iloc[0])
        return make_bottleneck(value, value > 5)

    monkeypatch.setattr(views, "load_membw_spec", load_spec)
    monkeypatch.setattr(views, "evaluate_membw_dfs", evaluate_membw)
    result = views.build_chart_views(workload, config, args)

    assert [
        view.membw.availability for view in (result.aggregate, *result.kernel_views)
    ] == ["full"] * 4
    assert result.aggregate.membw.annotations[0].block_id == "vl1d"
    assert result.aggregate.membw.annotations[0].label == "[!] TCP stall"
    assert result.aggregate.membw.annotations[0].value == format_value(
        16.702, display_unit("Percent")
    )
    assert result.aggregate.membw.guidance_blocks == ("TCP stall\n  Measured: 16.7%",)
    assert result.kernel_views[0].membw.annotations[0].value == format_value(
        6.468, display_unit("Percent")
    )
    assert result.kernel_views[1].membw.annotations == ()
    assert result.kernel_views[1].membw.status == (
        "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA)."
    )


@pytest.mark.parametrize(
    "unit",
    [
        "Percent",
        "Bytes/s",
        "KB per Wave",
        "Bytes per Workgroup",
        "Cycles",
        "Registers",
        "Workgroups per CU",
        "Workgroups per WGP",
        "(Requests + $normUnit)",
        None,
    ],
)
@pytest.mark.parametrize("value", [12.345, 12345, 1.5e9, None, float("nan"), "N/A"])
def test_slot_formatting_matches_current_terminal_units(unit, value):
    from memory_chart.html.diagram import SlotSpec
    from memory_chart.units import PLAIN

    rule = display_unit(unit) or PLAIN
    spec = SlotSpec("custom.0", "Arbitrary name", rule, "neutral", False, False, False)
    assert views._slot_value(spec, value).text == format_value(value, rule)


def test_stall_annotations_follow_resolved_metadata_instead_of_block_names():
    from dataclasses import replace

    from memory_chart.loader import Layouts

    layout = Layouts.for_arch("gfx1250")
    detail = views._membw_detail(make_bottleneck(18.7, True), layout)
    assert detail.annotations == ()
    columns = tuple(
        tuple(replace(b, stall_level="GL1" if b.id == "hbm" else None) for b in col)
        for col in layout.columns
    )
    detail = views._membw_detail(
        make_bottleneck(18.7, True), replace(layout, columns=columns)
    )
    assert [a.block_id for a in detail.annotations] == ["hbm"]
