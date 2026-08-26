# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/tty.py."""

import argparse
from io import StringIO
from types import SimpleNamespace

import pandas as pd
import pytest

from utils.tty import (
    convert_time_columns,
    format_duration,
    format_node_stats,
    format_table_output,
    has_time_data,
    print_operator_node,
    show_all,
    show_call_tree,
    show_operator_summary,
)
from utils.utils_analysis import (
    CallTreeNode,
    KernelStats,
    build_call_trees,
    build_operator_summary,
)
from utils.utils_common import is_gfx115x

TIME_UNITS = {"s": 10**9, "ms": 10**6, "us": 10**3, "ns": 1}

_OPERATOR_SUMMARY_COLUMNS = [
    "Operator",
    "Location",
    "Calls",
    "Dispatches",
    "Dispatches_Per_Call",
    "Total_GPU",
    "Pct_Total_GPU",
    "Mean_Per_Call",
    "Mean_Per_Dispatch",
    "Min_Dispatch",
    "Max_Dispatch",
]


def _build_summary_from_dataframe(rows):
    call_trees = build_call_trees(pd.DataFrame(rows))
    return build_operator_summary(call_trees)


def make_args() -> argparse.Namespace:
    """Minimal args for the plain-table render path."""
    return argparse.Namespace(decimal=2, view=None, normal_unit="per_wave")


def _sample_time_data() -> pd.DataFrame:
    """Metric table mixing a time row, a cycle row, and a count row."""
    return pd.DataFrame({
        "Metric_ID": ["7.2.0", "7.2.1", "7.2.2"],
        "Metric": ["Kernel Time", "Kernel Time (Cycles)", "Non-Time Metric"],
        "Avg": [3446.64, 64499.39, 1000.0],
        "Min": [1769.25, 17269.25, 500.0],
        "Max": [12532.12, 337030.50, 2000.0],
        "Unit": ["ns", "Cycle", "Count"],
    })


def _original_ns_values() -> dict[str, float]:
    """Original nanosecond values for the time row of _sample_time_data."""
    return {"Avg": 3446.64, "Min": 1769.25, "Max": 12532.12}


def test_format_table_output_suppresses_empty_column() -> None:
    """A non-PC-sampling table with an all-'N/A' column is suppressed."""
    df = pd.DataFrame({"Metric": ["a", "b"], "Value": ["N/A", "N/A"]})
    content = format_table_output(
        make_args(),
        {"id": 1101, "title": "Some Table"},
        df,
        "metric_table",
        runs={"only": object()},
    )
    assert content == ""


def test_format_table_output_keeps_pc_sampling_table_21_1() -> None:
    """PC sampling table 21.1 is shown even with an all-'N/A' source column."""
    df = pd.DataFrame({
        "source_line": ["N/A", "N/A"],
        "instruction": ["v_mov", "v_add"],
        "count": [3, 1],
    })
    content = format_table_output(
        make_args(),
        {"id": 2101, "title": "PC Sampling"},
        df,
        "pc_sampling_table",
        runs={"only": object()},
    )
    assert content != ""
    assert "v_mov" in content
    assert "Stall reason definitions:" in content
    assert "cdna3-cdna4-pc-sampling.html#stall-reasons" in content


def test_has_time_data_detection() -> None:
    """has_time_data is True only when a 'ns' Unit column is present."""
    assert has_time_data(_sample_time_data())

    no_time_data = pd.DataFrame({
        "Metric": ["Non-Time Metric"],
        "Avg": [1000.0],
        "Unit": ["Count"],
    })
    assert not has_time_data(no_time_data)

    no_unit_column = pd.DataFrame({"Metric": ["Some Metric"], "Avg": [1000.0]})
    assert not has_time_data(no_unit_column)


def test_default_unit_is_nanoseconds() -> None:
    """The fixture's time row defaults to nanoseconds."""
    sample_time_data = _sample_time_data()
    time_rows = sample_time_data["Unit"].str.lower().str.contains("ns", na=False)
    assert time_rows.any()
    assert sample_time_data.loc[0, "Unit"] == "ns"


def test_conversion_to_seconds() -> None:
    """Converting to seconds divides the time row and leaves others untouched."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "s")

    assert converted_df.loc[0, "Unit"] == "s"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["s"], abs=1e-10
    )
    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"


def test_conversion_to_milliseconds() -> None:
    """Converting to milliseconds divides the time row by 10^6."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "ms")

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["ms"], abs=1e-6
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["ms"], abs=1e-6
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["ms"], abs=1e-6
    )


def test_conversion_to_microseconds() -> None:
    """Converting to microseconds divides the time row by 10^3."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "us")

    assert converted_df.loc[0, "Unit"] == "us"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["us"], abs=1e-3
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"] / TIME_UNITS["us"], abs=1e-3
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"] / TIME_UNITS["us"], abs=1e-3
    )


def test_conversion_to_nanoseconds() -> None:
    """Converting to nanoseconds leaves the time row unchanged."""
    original_ns_values = _original_ns_values()
    converted_df = convert_time_columns(_sample_time_data(), "ns")

    assert converted_df.loc[0, "Unit"] == "ns"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"], abs=1e-10
    )
    assert converted_df.loc[0, "Min"] == pytest.approx(
        original_ns_values["Min"], abs=1e-10
    )
    assert converted_df.loc[0, "Max"] == pytest.approx(
        original_ns_values["Max"], abs=1e-10
    )


def test_non_time_rows_unchanged() -> None:
    """Cycle and Count rows keep their unit and value after conversion."""
    converted_df = convert_time_columns(_sample_time_data(), "ms")

    assert converted_df.loc[1, "Unit"] == "Cycle"
    assert converted_df.loc[2, "Unit"] == "Count"
    assert converted_df.loc[1, "Avg"] == 64499.39
    assert converted_df.loc[2, "Avg"] == 1000.0


def test_invalid_time_unit_is_noop() -> None:
    """An unrecognised target unit leaves the frame unchanged."""
    sample_time_data = _sample_time_data()
    original_df = sample_time_data.copy()
    converted_df = convert_time_columns(sample_time_data, "invalid_unit")
    pd.testing.assert_frame_equal(converted_df, original_df)


def test_missing_unit_column_is_noop() -> None:
    """A frame with no Unit column is returned unchanged."""
    df_no_unit = pd.DataFrame({"Metric": ["Test Metric"], "Avg": [1000.0]})
    converted_df = convert_time_columns(df_no_unit, "ms")
    pd.testing.assert_frame_equal(converted_df, df_no_unit)


def test_conversion_with_missing_columns() -> None:
    """Conversion works when Min/Max columns are absent."""
    original_ns_values = _original_ns_values()
    df_partial = _sample_time_data()[["Metric_ID", "Metric", "Avg", "Unit"]].copy()
    converted_df = convert_time_columns(df_partial, "ms")

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(
        original_ns_values["Avg"] / TIME_UNITS["ms"], abs=1e-6
    )


def test_mathematical_correctness_all_units() -> None:
    """Every supported unit divides the time row by the correct factor."""
    original_ns_values = _original_ns_values()
    for target_unit, divisor in TIME_UNITS.items():
        converted_df = convert_time_columns(_sample_time_data(), target_unit)

        assert converted_df.loc[0, "Avg"] == pytest.approx(
            original_ns_values["Avg"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Min"] == pytest.approx(
            original_ns_values["Min"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Max"] == pytest.approx(
            original_ns_values["Max"] / divisor, abs=1e-10
        )
        assert converted_df.loc[0, "Unit"] == target_unit


def test_integration_conversion_flow() -> None:
    """has_time_data gates convert_time_columns in the show_all flow."""
    args = argparse.Namespace(time_unit="ms", decimal=2)

    sample_df = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446640.0],
        "Min": [1769250.0],
        "Max": [12532120.0],
        "Unit": ["ns"],
    })

    if has_time_data(sample_df):
        converted_df = convert_time_columns(sample_df, args.time_unit)
    else:
        converted_df = sample_df

    assert converted_df.loc[0, "Unit"] == "ms"
    assert converted_df.loc[0, "Avg"] == pytest.approx(3.44664, abs=1e-5)
    assert converted_df.loc[0, "Min"] == pytest.approx(1.76925, abs=1e-5)
    assert converted_df.loc[0, "Max"] == pytest.approx(12.53212, abs=1e-5)


def test_show_all_with_time_unit_conversion() -> None:
    """Mixed-case 'Ns' unit converts correctly across every target unit."""
    test_data = pd.DataFrame({
        "Metric_ID": ["7.2.0"],
        "Metric": ["Kernel Time"],
        "Avg": [3446.64],
        "Min": [1769.25],
        "Max": [12532.12],
        "Unit": ["Ns"],
    })

    for time_unit in ["s", "ms", "us", "ns"]:
        converted_df = convert_time_columns(test_data, time_unit)
        assert converted_df.loc[0, "Unit"] == time_unit
        assert converted_df.loc[0, "Avg"] == pytest.approx(
            3446.64 / TIME_UNITS[time_unit], abs=1e-10
        )


@pytest.mark.parametrize(
    "membw_analysis",
    [
        pytest.param(False, id="disabled"),
        pytest.param(True, id="enabled"),
    ],
)
def test_show_all_membw_analysis_panel_gate(
    monkeypatch: pytest.MonkeyPatch,
    membw_analysis: bool,
) -> None:
    """Panel 3000 is rendered only when memory bandwidth analysis is enabled."""
    args = argparse.Namespace(
        decimal=2,
        filter_metrics=None,
        include_cols=None,
        membw_analysis=membw_analysis,
        normal_unit="per_wave",
        path=[["fixture"]],
        time_unit="ns",
        view=None,
    )
    metric_dataframe = pd.DataFrame({
        "Metric": ["EA read request fraction - HBM"],
        "Avg": [50.0],
        "Unit": ["Percent"],
    })
    table_config = {
        "id": 3013,
        "title": "EA Interface",
        "header": {"metric": "Metric", "value": "Avg", "unit": "Unit"},
    }
    arch_configs = SimpleNamespace(
        panel_configs={
            3000: {
                "id": 3000,
                "title": "Memory Bandwidth Analysis",
                "data source": [{"metric_table": table_config}],
            }
        }
    )
    runs = {
        "fixture": SimpleNamespace(
            dfs={3013: metric_dataframe},
            sys_info=pd.DataFrame([{"gpu_arch": "gfx950"}]),
        )
    }
    actual_calls: list[str] = []

    def record_process_table_data(*_args, **_kwargs):
        actual_calls.append("process_table_data")
        return metric_dataframe

    def record_format_table_output(*args, **kwargs):
        actual_calls.append("format_table_output")
        return format_table_output(*args, **kwargs)

    monkeypatch.setattr("utils.tty.process_table_data", record_process_table_data)
    monkeypatch.setattr("utils.tty.format_table_output", record_format_table_output)
    rendered_output = StringIO()

    show_all(
        args,
        runs,
        arch_configs,
        rendered_output,
        profiling_config={"filter_blocks": []},
    )

    output_lines = rendered_output.getvalue().splitlines()
    expected_calls = (
        ["process_table_data", "format_table_output"] if membw_analysis else []
    )
    assert actual_calls == expected_calls

    if not membw_analysis:
        assert output_lines == []
        return

    assert "30. Memory Bandwidth Analysis" in output_lines
    assert "30.13 EA Interface" in output_lines


def test_edge_cases_and_error_handling() -> None:
    """Empty, NaN, and mixed-case unit frames convert without error."""
    empty_df = pd.DataFrame()
    assert convert_time_columns(empty_df, "ms").empty

    nan_df = pd.DataFrame({
        "Avg": [float("nan"), 1000.0],
        "Unit": ["ns", "Count"],
    })
    result = convert_time_columns(nan_df, "ms")
    assert result.loc[0, "Unit"] == "ms"

    mixed_case_df = pd.DataFrame({
        "Avg": [1000.0, 2000.0],
        "Unit": ["ns", "NS"],
    })
    result = convert_time_columns(mixed_case_df, "ms")
    assert result.loc[0, "Unit"] == "ms"
    assert result.loc[1, "Unit"] == "ms"


@pytest.mark.parametrize(
    "gpu_arch",
    [
        pytest.param("gfx1151", id="rdna35"),
        pytest.param("gfx942", id="cdna"),
    ],
)
def test_format_table_output_dispatches_memory_chart_renderer(
    monkeypatch: pytest.MonkeyPatch,
    gpu_arch: str,
) -> None:
    """Memory Chart output uses the architecture renderer and shared heading."""
    calls: dict[str, dict] = {}

    def gfx11_stub(mem_data: dict, *, chart_title: str) -> str:
        calls["gfx11"] = {
            "mem_data": mem_data,
            "chart_title": chart_title,
        }
        return "rendered RDNA3.5 memory chart"

    def gfx9_stub(
        mem_data: dict,
        *,
        chart_title: str = "",
        gpu_arch: str = "",
    ) -> str:
        calls["gfx9"] = {
            "mem_data": mem_data,
            "chart_title": chart_title,
        }
        return "rendered CDNA memory chart"

    monkeypatch.setattr(
        "utils.tty.mem_chart_gfx11.plot_mem_chart",
        gfx11_stub,
    )
    monkeypatch.setattr(
        "utils.tty.mem_chart_gfx9.plot_mem_chart",
        gfx9_stub,
    )
    df = pd.DataFrame({"Metric": ["Metric A"], "Value": [1]})

    content = format_table_output(
        make_args(),
        {
            "id": 701,
            "title": "Memory Chart",
            "cli_style": "mem_chart",
        },
        df,
        "metric_table",
        runs={"only": object()},
        gpu_arch=gpu_arch,
    )

    expected = "gfx11" if is_gfx115x(gpu_arch) else "gfx9"
    unexpected = "gfx9" if is_gfx115x(gpu_arch) else "gfx11"
    assert calls[expected] == {
        "mem_data": {"Metric A": 1},
        "chart_title": "7. Memory Chart (Normalization: per_wave)",
    }
    assert unexpected not in calls
    return_value = (
        "rendered RDNA3.5 memory chart"
        if is_gfx115x(gpu_arch)
        else "rendered CDNA memory chart"
    )
    assert content == f"{return_value}\n"


def test_format_duration_microseconds_below_threshold():
    assert format_duration(0.005) == "5.00 us"


def test_format_duration_milliseconds_above_threshold():
    assert format_duration(1.5) == "1.50 ms"


def test_format_duration_boundary_value_is_milliseconds():
    assert format_duration(0.01) == "0.01 ms"


def test_format_duration_none_renders_na():
    assert format_duration(None) == "N/A"


def test_format_duration_nan_renders_na():
    assert format_duration(float("nan")) == "N/A"


def test_format_node_stats_omits_calls_when_no_invocation_ids():
    node = CallTreeNode(name="x")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    node.mean_dispatch_ns = 1_000_000.0
    node.min_dispatch_ns = 1_000_000.0
    node.max_dispatch_ns = 1_000_000.0
    rendered = format_node_stats(node)
    assert "calls:" not in rendered
    assert "dispatches: 1" in rendered
    assert "total: 1.00 ms" in rendered


def test_format_node_stats_includes_calls_when_invocation_ids_present():
    node = CallTreeNode(name="x")
    node.invocation_ids.add("ctx1")
    node.invocation_ids.add("ctx2")
    node.kernel_launches = 4
    node.total_duration_ms = 2.0
    node.mean_dispatch_ns = 500_000.0
    node.min_dispatch_ns = 500_000.0
    node.max_dispatch_ns = 500_000.0
    rendered = format_node_stats(node)
    assert "calls: 2" in rendered
    assert "dispatches: 4" in rendered


def test_format_node_stats_renders_na_when_dispatch_stats_missing():
    node = CallTreeNode(name="x")
    node.kernel_launches = 0
    rendered = format_node_stats(node)
    assert "dispatch_mean: N/A" in rendered
    assert "dispatch_min: N/A" in rendered
    assert "dispatch_max: N/A" in rendered


def test_show_call_tree_prints_location_and_stats(capsys):
    root = CallTreeNode(name="main.py:10")
    root.kernel_launches = 1
    root.total_duration_ms = 0.5
    child = CallTreeNode(name="op_a")
    child.kernel_launches = 1
    child.total_duration_ms = 0.5
    child.kernels["kern"] = KernelStats(launches=1, total_duration_ns=500_000.0)
    root.children["op_a"] = child
    show_call_tree({"main.py:10": root})
    output = capsys.readouterr().out
    assert "main.py:10" in output
    assert "dispatches: 1" in output
    assert "kern" in output


def test_show_call_tree_sorted_by_duration(capsys):
    root_a = CallTreeNode(name="a.py:1")
    root_a.total_duration_ms = 10.0
    root_a.kernel_launches = 1
    root_b = CallTreeNode(name="b.py:1")
    root_b.total_duration_ms = 20.0
    root_b.kernel_launches = 2
    show_call_tree({"a.py:1": root_a, "b.py:1": root_b})
    output = capsys.readouterr().out
    assert output.index("b.py:1") < output.index("a.py:1")


def test_show_call_tree_kernel_id_printed(capsys):
    root = CallTreeNode(name="f.py:1")
    root.kernel_launches = 1
    root.total_duration_ms = 1.0
    child = CallTreeNode(name="op")
    child.kernel_launches = 1
    child.total_duration_ms = 1.0
    child.kernels["kern_x"] = KernelStats(
        launches=1, total_duration_ns=1_000_000.0, kernel_id=42
    )
    root.children["op"] = child
    show_call_tree({"f.py:1": root})
    output = capsys.readouterr().out
    assert "(id 42)" in output


def test_print_operator_node_branching_shows_stats(capsys):
    node = CallTreeNode(name="branch")
    node.kernel_launches = 2
    node.total_duration_ms = 5.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    node.kernels["k2"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    print_operator_node(node)
    output = capsys.readouterr().out
    assert "dispatches: 2" in output
    assert "k1" in output
    assert "k2" in output


def test_print_operator_node_non_branching_omits_stats(capsys):
    node = CallTreeNode(name="single")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=1_000_000.0)
    print_operator_node(node)
    output = capsys.readouterr().out
    lines = output.strip().split("\n")
    assert "└─ single" in lines[0]
    assert "dispatches" not in lines[0]


def test_print_operator_node_long_kernel_wraps(capsys):
    node = CallTreeNode(name="single")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    long_kernel_name = "K" * 220
    node.kernels[long_kernel_name] = KernelStats(
        launches=1,
        total_duration_ns=1_000_000.0,
        kernel_id=7,
    )
    print_operator_node(node)
    output_lines = capsys.readouterr().out.splitlines()
    assert any("└─ single" in line for line in output_lines)
    kernel_lines = [
        line for line in output_lines if "(id 7)" in line or line.startswith("   ")
    ]
    assert any(line.startswith("   └─ ") for line in kernel_lines)
    wrapped_kernel_lines = [
        line
        for line in kernel_lines
        if line.startswith("   ") and not line.startswith("   └─ ")
    ]
    assert wrapped_kernel_lines
    assert not any(line.strip().startswith("(id 7)") for line in output_lines)


# ---------------------------------------------------------------------------
# show_operator_summary
# ---------------------------------------------------------------------------


def test_show_operator_summary_empty_prints_no_dispatches_message(capsys):
    show_operator_summary(pd.DataFrame(columns=_OPERATOR_SUMMARY_COLUMNS))
    output = capsys.readouterr().out
    assert "no operators with recorded dispatches" in output


def test_show_operator_summary_renders_per_cell_unit_suffix(capsys):
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 2_000_000,
        }
    ])
    show_operator_summary(summary)
    output = capsys.readouterr().out
    assert "ms" in output or "us" in output
    assert "Operator" in output
    assert "Total" in output


def test_show_operator_summary_renders_na_for_nan_cells(capsys):
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 0.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    show_operator_summary(summary)
    output = capsys.readouterr().out
    assert "N/A" in output
