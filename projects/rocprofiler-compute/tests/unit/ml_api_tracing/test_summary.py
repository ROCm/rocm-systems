# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API operator summaries."""

import copy

import pandas as pd
import pytest

from ml_api_tracing.models import CallTreeNode
from ml_api_tracing.summary import build_operator_summary

OPERATOR_SUMMARY_COLUMNS = [
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


def summary_node(
    name,
    calls=("1",),
    dispatches=1,
    total_gpu_ms=1.0,
    mean_dispatch_ms=1.0,
    min_dispatch_ms=1.0,
    max_dispatch_ms=1.0,
    file_name="",
    line_number=None,
):
    return CallTreeNode(
        name=name,
        invocation_ids=set(calls),
        kernel_launches=dispatches,
        total_duration_ms=total_gpu_ms,
        mean_dispatch_ns=(
            None if mean_dispatch_ms is None else mean_dispatch_ms * 1_000_000
        ),
        min_dispatch_ns=(
            None if min_dispatch_ms is None else min_dispatch_ms * 1_000_000
        ),
        max_dispatch_ns=(
            None if max_dispatch_ms is None else max_dispatch_ms * 1_000_000
        ),
        file_name=file_name,
        line_number=line_number,
    )


def test_build_operator_summary_empty_input_returns_empty_with_full_schema():
    summary = build_operator_summary({})
    assert list(summary.columns) == OPERATOR_SUMMARY_COLUMNS
    assert summary.empty


def test_build_operator_summary_location_from_node_file_line():
    located = CallTreeNode(
        name="linear",
        file_name="simple_torch_code.py",
        line_number=19,
    )
    located.kernel_launches = 1
    located.total_duration_ms = 1.0
    located.invocation_ids.add("10")
    missing = CallTreeNode(name="addmm")
    missing.kernel_launches = 1
    missing.total_duration_ms = 0.5
    missing.invocation_ids.add("20")
    located.children = [missing]
    summary = build_operator_summary({"1": [located]})
    by_name = summary.set_index("Operator")
    assert by_name.loc["linear", "Location"] == "simple_torch_code.py:19"
    assert by_name.loc["linear/addmm", "Location"] == ""


@pytest.mark.parametrize(
    ("second_file", "second_line", "expected_location"),
    [
        pytest.param("first.py", 10, "first.py:10", id="same-location"),
        pytest.param("second.py", 20, "", id="different-location"),
    ],
)
def test_build_operator_summary_aggregates_equal_paths_across_threads(
    second_file, second_line, expected_location
):
    first = summary_node(
        "linear",
        calls=("1",),
        total_gpu_ms=2.0,
        mean_dispatch_ms=2.0,
        min_dispatch_ms=2.0,
        max_dispatch_ms=2.0,
        file_name="first.py",
        line_number=10,
    )
    second = summary_node(
        "linear",
        calls=("2", "3"),
        dispatches=3,
        total_gpu_ms=12.0,
        mean_dispatch_ms=4.0,
        min_dispatch_ms=3.0,
        max_dispatch_ms=5.0,
        file_name=second_file,
        line_number=second_line,
    )
    forest = {"101": [first], "202": [second]}
    original = copy.deepcopy(forest)

    summary = build_operator_summary(forest)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["Operator"] == "linear"
    assert row["Location"] == expected_location
    assert row["Calls"] == 3.0
    assert row["Dispatches"] == 4.0
    assert row["Dispatches_Per_Call"] == pytest.approx(4.0 / 3.0)
    assert row["Total_GPU"] == 14.0
    assert row["Pct_Total_GPU"] == 100.0
    assert row["Mean_Per_Call"] == pytest.approx(14.0 / 3.0)
    assert row["Mean_Per_Dispatch"] == 3.5
    assert row["Min_Dispatch"] == 2.0
    assert row["Max_Dispatch"] == 5.0
    assert forest == original


def test_build_operator_summary_uses_root_gpu_time_for_inclusive_percentages():
    child = summary_node(
        "child",
        calls=("2",),
        dispatches=2,
        total_gpu_ms=8.0,
        mean_dispatch_ms=4.0,
        min_dispatch_ms=3.0,
        max_dispatch_ms=5.0,
    )
    root = summary_node(
        "root",
        dispatches=3,
        total_gpu_ms=12.0,
        mean_dispatch_ms=4.0,
        min_dispatch_ms=3.0,
        max_dispatch_ms=5.0,
    )
    root.children = [child]
    other = summary_node(
        "other",
        calls=("3",),
        total_gpu_ms=3.0,
        mean_dispatch_ms=3.0,
        min_dispatch_ms=3.0,
        max_dispatch_ms=3.0,
    )

    summary = build_operator_summary({"101": [root], "202": [other]})
    by_name = summary.set_index("Operator")

    assert by_name.loc["root", "Pct_Total_GPU"] == 80.0
    assert by_name.loc["root/child", "Pct_Total_GPU"] == pytest.approx(100.0 * 8 / 15)
    assert by_name.loc["other", "Pct_Total_GPU"] == 20.0
    assert summary["Pct_Total_GPU"].sum() == pytest.approx(100.0 * 23 / 15)


def test_build_operator_summary_preserves_missing_calls_and_dispatch_statistics():
    known = summary_node(
        "operator",
        calls=("1",),
        total_gpu_ms=2.0,
        mean_dispatch_ms=2.0,
        min_dispatch_ms=2.0,
        max_dispatch_ms=2.0,
    )
    missing = summary_node(
        "operator",
        calls=(),
        dispatches=2,
        total_gpu_ms=4.0,
        mean_dispatch_ms=None,
        min_dispatch_ms=None,
        max_dispatch_ms=None,
    )
    entirely_missing = summary_node(
        "missing",
        calls=(),
        mean_dispatch_ms=None,
        min_dispatch_ms=None,
        max_dispatch_ms=None,
    )

    summary = build_operator_summary({
        "101": [known],
        "202": [missing, entirely_missing],
    }).set_index("Operator")

    assert pd.isna(summary.loc["operator", "Calls"])
    assert pd.isna(summary.loc["operator", "Dispatches_Per_Call"])
    assert pd.isna(summary.loc["operator", "Mean_Per_Call"])
    assert summary.loc["operator", "Dispatches"] == 3.0
    assert summary.loc["operator", "Total_GPU"] == 6.0
    assert summary.loc["operator", "Mean_Per_Dispatch"] == 2.0
    assert summary.loc["operator", "Min_Dispatch"] == 2.0
    assert summary.loc["operator", "Max_Dispatch"] == 2.0
    assert pd.isna(summary.loc["missing", "Mean_Per_Dispatch"])
    assert pd.isna(summary.loc["missing", "Min_Dispatch"])
    assert pd.isna(summary.loc["missing", "Max_Dispatch"])


def test_build_operator_summary_sorts_duration_then_path_and_keeps_path_identity():
    child = summary_node("shared", calls=("2",), total_gpu_ms=1.0)
    parent = summary_node("parent", total_gpu_ms=1.0)
    parent.children = [child]
    first = summary_node("alpha", calls=("3",), total_gpu_ms=2.0)
    second = summary_node("zeta", calls=("4",), total_gpu_ms=2.0)
    different_path = summary_node("shared", calls=("5",), total_gpu_ms=1.0)

    summary = build_operator_summary({
        "101": [second, parent],
        "202": [first, different_path],
    })

    assert summary["Operator"].tolist() == [
        "alpha",
        "zeta",
        "parent",
        "parent/shared",
        "shared",
    ]
    assert summary.index.tolist() == [0, 1, 2, 3, 4]


def test_build_operator_summary_skips_cpu_only_nodes_and_handles_zero_root_total():
    cpu_only = CallTreeNode("cpu-only", invocation_ids={"1"})
    dispatch = summary_node(
        "zero-duration",
        total_gpu_ms=0.0,
        mean_dispatch_ms=0.0,
        min_dispatch_ms=None,
        max_dispatch_ms=None,
    )

    summary = build_operator_summary({"101": [cpu_only, dispatch]})

    assert summary["Operator"].tolist() == ["zero-duration"]
    assert summary.iloc[0]["Total_GPU"] == 0.0
    assert summary.iloc[0]["Mean_Per_Dispatch"] == 0.0
    assert pd.isna(summary.iloc[0]["Pct_Total_GPU"])
