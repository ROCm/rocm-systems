# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests covering functions in src/utils/tty.py (call-tree display, operator summary,
format_duration, format_node_stats, etc.) and TestBuildMetricList
(display/formatting)."""

import math

import pandas as pd
import pytest

from utils.tty import (
    format_duration,
    format_node_stats,
    print_operator_node,
    show_call_tree,
    show_operator_summary,
)
from utils.utils_analysis import (
    CallTreeNode,
    KernelStats,
    NodeRollup,
    build_call_trees,
    build_operator_summary,
    parse_top_level_location,
    rollup_node_stats,
)

# ---------------------------------------------------------------------------
# Tests for call-tree functions (build/display)
# ---------------------------------------------------------------------------


def test_parse_location_normal():
    assert parse_top_level_location("10@main.py:60/#10@main.py:21") == "main.py:60"


def test_parse_location_single_entry():
    assert parse_top_level_location("5@train.py:42") == "train.py:42"


def test_parse_location_nan():
    assert parse_top_level_location(float("nan")) == "unknown:0"


def test_parse_location_none():
    assert parse_top_level_location(None) == "unknown:0"


def test_parse_location_empty():
    assert parse_top_level_location("") == "unknown:0"
    assert parse_top_level_location("   ") == "unknown:0"


def test_parse_location_no_at_sign():
    assert parse_top_level_location("no_at_sign") == "unknown:0"


def test_parse_location_no_colon():
    assert parse_top_level_location("10@mainpy") == "unknown:0"


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


def test_kernel_stats_defaults_min_max_to_none():
    stats = KernelStats()
    assert stats.min_duration_ns is None
    assert stats.max_duration_ns is None


def test_call_tree_node_defaults_dispatch_stats_to_none():
    node = CallTreeNode(name="x")
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns is None


def test_call_tree_node_call_count_is_property_of_invocation_ids():
    node = CallTreeNode(name="x")
    assert node.call_count == 0
    node.invocation_ids.add("ctx1")
    node.invocation_ids.add("ctx2")
    assert node.call_count == 2


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


def test_rollup_leaf_node():
    node = CallTreeNode(name="leaf")
    node.kernels["kern_a"] = KernelStats(launches=2, total_duration_ns=1000.0)
    rollup = rollup_node_stats(node)
    assert rollup.launches == 2
    assert rollup.total_duration_ns == 1000.0
    assert node.kernel_launches == 2


def test_rollup_leaf_node_with_no_min_max_returns_none():
    node = CallTreeNode(name="leaf")
    node.kernels["kern"] = KernelStats(launches=1, total_duration_ns=0.0)
    rollup = rollup_node_stats(node)
    assert isinstance(rollup, NodeRollup)
    assert rollup.min_dispatch_ns is None
    assert rollup.max_dispatch_ns is None
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns == 0.0


def test_rollup_leaf_node_with_zero_launches_has_mean_none():
    node = CallTreeNode(name="leaf")
    rollup = rollup_node_stats(node)
    assert rollup.launches == 0
    assert node.mean_dispatch_ns is None


def test_rollup_propagates_min_max_from_kernel_stats():
    node = CallTreeNode(name="leaf")
    node.kernels["k"] = KernelStats(
        launches=2,
        total_duration_ns=3000.0,
        min_duration_ns=1000.0,
        max_duration_ns=2000.0,
    )
    rollup_node_stats(node)
    assert node.min_dispatch_ns == 1000.0
    assert node.max_dispatch_ns == 2000.0
    assert node.mean_dispatch_ns == 1500.0


def test_rollup_parent_rolls_up_children():
    child = CallTreeNode(name="child")
    child.kernels["kern_a"] = KernelStats(launches=3, total_duration_ns=3000.0)
    parent = CallTreeNode(name="parent")
    parent.children["child"] = child
    parent.kernels["kern_b"] = KernelStats(launches=1, total_duration_ns=500.0)
    rollup_node_stats(parent)
    assert parent.kernel_launches == 4
    assert child.kernel_launches == 3


def test_rollup_deep_hierarchy():
    grandchild = CallTreeNode(name="grandchild")
    grandchild.kernels["k"] = KernelStats(launches=1, total_duration_ns=100.0)
    child = CallTreeNode(name="child")
    child.children["grandchild"] = grandchild
    child.kernels["k2"] = KernelStats(launches=2, total_duration_ns=200.0)
    root = CallTreeNode(name="root")
    root.children["child"] = child
    rollup_node_stats(root)
    assert grandchild.kernel_launches == 1
    assert child.kernel_launches == 3
    assert root.kernel_launches == 3


def test_build_call_trees_empty_df():
    assert build_call_trees(pd.DataFrame()) == {}


def test_build_call_trees_missing_columns():
    assert build_call_trees(pd.DataFrame([{"Operator_Name": "a"}])) == {}


def test_build_call_trees_single_dispatch():
    df = pd.DataFrame([
        {
            "Operator_Name": "torch.nn.Linear",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@train.py:42",
            "Start_Timestamp_kernel": 1000,
            "End_Timestamp_kernel": 2000,
        }
    ])
    call_trees = build_call_trees(df)
    assert "train.py:42" in call_trees
    assert call_trees["train.py:42"].kernel_launches == 1
    assert "torch.nn.Linear" in call_trees["train.py:42"].children


def test_build_call_trees_hierarchy_split():
    df = pd.DataFrame([
        {
            "Operator_Name": "aten/linear/addmm",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@file.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    call_trees = build_call_trees(df)
    root = call_trees["file.py:1"]
    assert "aten" in root.children
    assert "linear" in root.children["aten"].children
    assert "addmm" in root.children["aten"].children["linear"].children


def test_build_call_trees_multiple_dispatches_same_kernel():
    rows = [
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": i * 1000,
            "End_Timestamp_kernel": (i + 1) * 1000,
        }
        for i in range(3)
    ]
    call_trees = build_call_trees(pd.DataFrame(rows))
    assert call_trees["f.py:1"].kernel_launches == 3
    assert call_trees["f.py:1"].children["op_a"].kernels["kern"].launches == 3


def test_build_call_trees_dedup_identical_timestamps():
    row = {
        "Operator_Name": "op",
        "Kernel_Name": "kern",
        "Context_Id": "10@f.py:1",
        "Start_Timestamp_kernel": 1000,
        "End_Timestamp_kernel": 2000,
    }
    assert build_call_trees(pd.DataFrame([row, row]))["f.py:1"].kernel_launches == 1


def test_build_call_trees_no_context_id():
    df = pd.DataFrame([
        {
            "Operator_Name": "op",
            "Kernel_Name": "kern",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    assert "unknown:0" in build_call_trees(df)


def test_build_call_trees_duration_rollup():
    df = pd.DataFrame([
        {
            "Operator_Name": "parent/child",
            "Kernel_Name": "kern_a",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "parent",
            "Kernel_Name": "kern_b",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 2_000_000,
            "End_Timestamp_kernel": 3_000_000,
        },
    ])
    call_trees = build_call_trees(df)
    root = call_trees["f.py:1"]
    assert root.kernel_launches == 2
    assert root.children["parent"].kernel_launches == 2
    assert root.children["parent"].children["child"].kernel_launches == 1


def test_build_call_trees_multiple_source_locations():
    df = pd.DataFrame([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@a.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "10@b.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
    ])
    call_trees = build_call_trees(df)
    assert "a.py:1" in call_trees
    assert "b.py:2" in call_trees


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
# build_operator_summary
# ---------------------------------------------------------------------------


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


def test_build_operator_summary_empty_input_returns_empty_with_full_schema():
    summary = build_operator_summary({})
    assert list(summary.columns) == _OPERATOR_SUMMARY_COLUMNS
    assert summary.empty


def test_build_operator_summary_skips_synthetic_location_root():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        }
    ])
    assert "f.py:1" not in summary["Operator"].tolist()
    assert "op_a" in summary["Operator"].tolist()


def test_build_operator_summary_row_values_for_single_dispatch():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 2_000_000,
        }
    ])
    row = summary.loc[summary["Operator"] == "op_a"].iloc[0]
    assert row["Location"] == "f.py:1"
    assert row["Calls"] == 1
    assert row["Dispatches"] == 1
    assert row["Dispatches_Per_Call"] == 1.0
    assert row["Total_GPU"] == pytest.approx(2.0)
    assert row["Pct_Total_GPU"] == pytest.approx(100.0)
    assert row["Mean_Per_Call"] == pytest.approx(2.0)
    assert row["Mean_Per_Dispatch"] == pytest.approx(2.0)
    assert row["Min_Dispatch"] == pytest.approx(2.0)
    assert row["Max_Dispatch"] == pytest.approx(2.0)


def test_build_operator_summary_sort_by_total_descending():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "small_op",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "big_op",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 10_000_000,
        },
    ])
    operators_in_order = summary["Operator"].tolist()
    assert operators_in_order.index("big_op") < operators_in_order.index("small_op")


def test_build_operator_summary_pct_total_gpu_sums_to_100_at_top_level():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 3_000_000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
    ])
    op_a_pct = summary.loc[summary["Operator"] == "op_a", "Pct_Total_GPU"].iloc[0]
    op_b_pct = summary.loc[summary["Operator"] == "op_b", "Pct_Total_GPU"].iloc[0]
    assert op_a_pct == pytest.approx(75.0)
    assert op_b_pct == pytest.approx(25.0)


def test_build_operator_summary_pct_total_gpu_is_nan_when_grand_total_zero():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 0.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    pct = summary.loc[summary["Operator"] == "op", "Pct_Total_GPU"].iloc[0]
    assert math.isnan(pct)


def test_build_operator_summary_min_max_mean_are_nan_when_no_dispatch_stats():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 5.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "op"].iloc[0]
    assert math.isnan(row["Min_Dispatch"])
    assert math.isnan(row["Max_Dispatch"])
    assert math.isnan(row["Mean_Per_Dispatch"])


def test_build_operator_summary_calls_nan_when_no_invocation_ids():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="torch.ops.x")
    op.kernel_launches = 2
    op.total_duration_ms = 4.0
    op.mean_dispatch_ns = 2_000_000.0
    op.min_dispatch_ns = 2_000_000.0
    op.max_dispatch_ns = 2_000_000.0
    root.children["torch.ops.x"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "torch.ops.x"].iloc[0]
    assert math.isnan(row["Calls"])
    assert math.isnan(row["Dispatches_Per_Call"])
    assert math.isnan(row["Mean_Per_Call"])
    assert row["Dispatches"] == 2


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


# =============================================================================
# BUILD METRIC LIST TESTS
# =============================================================================


class TestBuildMetricList:
    """Tests for build_metric_list and _metric_has_valid_expr."""

    # Maps YAML metric expression keys to their SUPPORTED_FIELD display names.
    _EXPR_KEY_TO_HEADER_DISPLAY = {
        "value": "Value",
        "avg": "Avg",
        "min": "Min",
        "max": "Max",
        "expr": "Expression",
        "median": "Median",
        "count": "Count",
    }

    @classmethod
    def setup_class(cls):
        from utils.utils_common import build_metric_list

        cls.build_metric_list = staticmethod(build_metric_list)

    def _build_test_panel_configs_for_single_metric(
        self, metric_name: str, expression_values: dict
    ):
        """
        Build panel_configs containing a single metric for testing.
        """
        from collections import OrderedDict

        header = {"metric": "Metric"}
        for key in expression_values:
            if key in self._EXPR_KEY_TO_HEADER_DISPLAY:
                header[key] = self._EXPR_KEY_TO_HEADER_DISPLAY[key]

        table = {
            "id": 201,
            "title": "Test Table",
            "header": header,
            "metric": {metric_name: expression_values},
        }
        if "expr" in expression_values:
            table["cli_style"] = "simple_box"

        panel_configs = OrderedDict()
        panel_configs[200] = {
            "id": 200,
            "title": "Test Panel",
            "data source": [{"metric_table": table}],
        }

        return panel_configs

    @staticmethod
    def _extract_leaf_metric_entries(metric_list):
        """Return only leaf metric entries whose ID has format 'panel.table.index'."""
        return {k: v for k, v in metric_list.items() if k.count(".") == 2}

    def test_given_metric_with_valid_value__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Valid Metric A", {"value": "AVG(COUNTER_A)"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Valid Metric A" in leaf_entries.values()

    def test_given_metric_with_python_none__it_doesnt_present_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Unsupported Metric B", {"value": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Unsupported Metric B" not in leaf_entries.values()

    def test_given_metric_with_string_none__it_doesnt_present_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Unsupported Metric C", {"value": "None"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Unsupported Metric C" not in leaf_entries.values()

    def test_given_expr_metric__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Expr Metric", {"expr": "(100 * COUNTER_B / COUNTER_C)"}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Expr Metric" in leaf_entries.values()

    def test_given_metric_with_partial_avg_min_max__it_presents_in_metric_list(self):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "Partial Metric", {"avg": "AVG(COUNTER_E)", "min": None, "max": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "Partial Metric" in leaf_entries.values()

    def test_given_metric_with_all_none_avg_min_max__it_doesnt_present_in_metric_list(
        self,
    ):
        panel_configs = self._build_test_panel_configs_for_single_metric(
            "All None Metric", {"avg": None, "min": None, "max": None}
        )
        metric_list = self.build_metric_list(panel_configs, None)
        leaf_entries = self._extract_leaf_metric_entries(metric_list)
        assert "All None Metric" not in leaf_entries.values()
