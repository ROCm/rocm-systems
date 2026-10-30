# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for aggregate and per-kernel HTML memory chart values."""

import json
from argparse import Namespace
from collections import OrderedDict
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import pandas as pd
import pytest

from membw_analysis.models import (
    BottleneckNode,
    MemBwAnalysisResult,
    SupportingMetric,
)
from memory_chart.html import views
from memory_chart.html.db_rows import KernelRows, WorkloadChartRows
from memory_chart.html.diagram import SlotSpec
from memory_chart.loader import Layouts
from memory_chart.units import PLAIN, DisplayUnit, display_unit, format_value
from utils.metrics.expression import build_eval_string
from utils.schema import ArchConfig, Workload


def make_inputs(
    *,
    kernel_filter: Optional[List[int]] = None,
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


def test_chart_metric_table_formats_missing_values_as_unavailable() -> None:
    """Keep a missing table value consistent with its diagram slot."""
    table = pd.DataFrame({
        "Metric": ["Read BW", "Hit Rate"],
        "Value": [1.25e9, None],
        "Unit": ["Bytes/s", "Percent"],
    })

    specs = {
        "Read BW": SlotSpec(
            "arrow.0",
            "Read BW",
            DisplayUnit("bandwidth", " GB/s", 3, 1e-9),
            "read",
            True,
            False,
            False,
        ),
        "Hit Rate": SlotSpec(
            "l2.0",
            "Hit Rate",
            DisplayUnit("percent", "%"),
            "hit",
            False,
            False,
            True,
        ),
    }

    rows, values = views._format_chart_metric_rows(table, specs)

    assert rows == (("Read BW", "1.250", "GB/s"), ("Hit Rate", "N/A", "%"))
    assert views._slot_value(specs["Hit Rate"], values["Hit Rate"]).text == rows[1][1]

    assert (
        f"{rows[0][1]} {rows[0][2]}"
        == views._slot_value(specs["Read BW"], values["Read BW"]).text
    )

    assert values["Read BW"] == 1.25e9
    assert pd.isna(values["Hit Rate"])


@pytest.mark.parametrize(
    "unit,value,expected_value,expected_unit",
    [
        ("Registers", 32, "32.0", "Registers"),
        ("Workgroups per CU", 12.345, "12.3", "Workgroups per CU"),
        ("Workgroups per WGP", 12.345, "12.3", "Workgroups per WGP"),
        ("(Requests + $normUnit)", 999, "999", "Requests per kernel"),
        ("(Requests + $normUnit)", 12345, "1.23e+04", "Requests per kernel"),
        ("Bytes/s", 1.25e9, "1.250", "GB/s"),
        ("Bytes per Workgroup", 2048, "2.0", "KB"),
        ("KB per Wave", 1.25, "1.250", "KB"),
        ("Percent", 12.345, "12.3", "%"),
        ("Cycles", 12.345, "12.3", "cycles"),
    ],
)
def test_chart_metric_table_preserves_units_with_converted_values(
    unit: str, value: float, expected_value: str, expected_unit: str
) -> None:
    """Keep resolved table labels when the diagram formatter has no suffix."""
    table_unit = "Requests per kernel" if "$normUnit" in unit else unit
    table = pd.DataFrame({
        "Metric": ["Synthetic metric"],
        "Value": [value],
        "Unit": [table_unit],
    })
    original_table = table.copy()
    spec = SlotSpec(
        "custom.0",
        "Synthetic metric",
        display_unit(unit),
        "neutral",
        False,
        False,
        False,
    )

    rows, _ = views._format_chart_metric_rows(table, {spec.metric: spec})

    assert rows == (("Synthetic metric", expected_value, expected_unit),)
    pd.testing.assert_frame_equal(table, original_table)


@pytest.mark.parametrize("value", [None, float("nan"), "N/A"])
@pytest.mark.parametrize(
    "unit,table_unit,expected_unit",
    [
        ("Registers", "Registers", "Registers"),
        ("Workgroups per CU", "Workgroups per CU", "Workgroups per CU"),
        ("(Requests + $normUnit)", "Requests per kernel", "Requests per kernel"),
        ("Bytes/s", "Bytes/s", "GB/s"),
        ("Bytes per Workgroup", "Bytes per Workgroup", "KB"),
        ("Percent", "Percent", "%"),
    ],
)
def test_chart_metric_table_retains_units_for_unavailable_values(
    value: Union[float, str, None], unit: str, table_unit: str, expected_unit: str
) -> None:
    """Keep original or converted unit labels even when a value is unavailable."""
    table = pd.DataFrame({
        "Metric": ["Synthetic metric"],
        "Value": [value],
        "Unit": [table_unit],
    })
    spec = SlotSpec(
        "custom.0",
        "Synthetic metric",
        display_unit(unit),
        "neutral",
        False,
        False,
        False,
    )

    rows, _ = views._format_chart_metric_rows(table, {spec.metric: spec})

    assert rows == (("Synthetic metric", "N/A", expected_unit),)


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
    """Match slot formatting to terminal units across values and unit types."""
    rule = display_unit(unit) or PLAIN
    spec = SlotSpec("custom.0", "Arbitrary name", rule, "neutral", False, False, False)

    assert views._slot_value(spec, value).text == format_value(value, rule)


def test_stall_annotations_follow_resolved_metadata_instead_of_block_names():
    """Use resolved layout metadata to choose stall annotation blocks."""
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


def make_stored_inputs() -> Tuple[WorkloadChartRows, ArchConfig, Namespace]:
    """Stored rows for three chart metrics, with one template cell left empty."""
    config = ArchConfig()
    config.dfs[301] = pd.DataFrame(
        {
            "Metric": ["Hit", "Scratch", "Empty"],
            "Value": [
                build_eval_string("ROUND(SUM(Counter), 1)"),
                build_eval_string("SUM(Counter)"),
                "",
            ],
            "Unit": ["Percent", "KB per Wave", "Registers"],
        },
        index=pd.Index(["3.1.0", "3.1.1", "3.1.2"], name="Metric_ID"),
    )
    config.dfs_type[301] = "metric_table"
    config.panel_configs = OrderedDict({
        300: {
            "data source": [
                {
                    "metric_table": {
                        "id": 301,
                        "title": "Memory Chart",
                        "header": {},
                        "metric": {
                            "Hit": {"unit": "Percent"},
                            "Scratch": {"unit": "KB per Wave"},
                            "Empty": {"unit": "Registers"},
                        },
                    }
                }
            ]
        }
    })
    args = Namespace(
        time_unit=None,
        cols=None,
        include_cols=[],
        decimal=2,
        verbose=0,
        report_diff=0,
        debug=False,
        normal_unit="per_kernel",
    )
    stored = WorkloadChartRows(
        arch="gfx950",
        layout_json=json.loads(Layouts.for_arch("gfx950").path.read_text("utf-8")),
        aggregate={
            "3.1.0": {"Value": 45.678},
            "3.1.1": {"Value": 2560.0},
            "3.1.2": {"Value": None},
        },
        kernels=(KernelRows("kernel alpha", {"3.1.0": {"Value": 10.26}}),),
    )
    return stored, config, args


def test_stored_rows_fill_chart_cells_as_the_terminal_shows_them() -> None:
    """Round to the to_round digits, convert to config units, and show gaps as N/A."""
    stored, config, args = make_stored_inputs()

    result = views.build_chart_views_from_db(stored, config, args, views.ChartScope())

    table = result.aggregate.tables[0]
    assert table.rows == (
        ("Hit", 45.7, "Percent"),
        ("Scratch", 2.5, "KB per Wave"),
        ("Empty", "N/A", "Registers"),
    )


def test_stored_rows_give_kernel_views_in_stored_order() -> None:
    """Kernel views read each kernel's own rows and keep the stored order."""
    stored, config, args = make_stored_inputs()

    result = views.build_chart_views_from_db(
        stored, config, args, views.ChartScope(kernel_ids=[0])
    )

    assert result.kernels == (views.KernelEntry(0, "kernel alpha"),)
    assert result.initial_kernel == 0
    assert result.kernel_views[0].tables[0].rows[0][:2] == ("Hit", 10.3)


def test_chart_metric_ids_are_the_chart_table_rows() -> None:
    """The ids the database reader fetches are the rows of the chart tables."""
    _, config, _ = make_stored_inputs()

    assert views.chart_metric_ids(config) == ["3.1.0", "3.1.1", "3.1.2"]
