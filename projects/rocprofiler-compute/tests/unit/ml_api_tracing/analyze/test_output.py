# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API operator reports and summaries."""

import sys
from typing import Dict, List

import pandas as pd
import pytest

from ml_api_tracing.analyze.call_tree import (
    rollup_node_stats,
)
from ml_api_tracing.analyze.definitions import CallTreeForest, CallTreeNode, KernelStats
from ml_api_tracing.analyze.operator_patterns import operator_backend_labels
from ml_api_tracing.analyze.output import (
    build_continuation_indent,
    build_operator_summary,
    format_args_variant_lines,
    format_duration,
    format_node_args,
    format_node_stats,
    format_operator_args,
    list_ml_operators,
    print_operator_node,
    show_call_tree,
    show_matched_operator_report,
    show_matched_operators,
    show_operator_summary,
)
from utils.schema import Workload

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


@pytest.fixture(autouse=True)
def fixed_display_terminal_width(monkeypatch):
    """Keep tree and summary assertions independent of the host terminal."""
    monkeypatch.setenv("COLUMNS", "120")


def build_display_node(
    name: str,
    backend: str,
    kernels: Dict[str, KernelStats],
    children: List[CallTreeNode],
) -> CallTreeNode:
    invocation_ids = {f"{name}:1", f"{name}:2"}
    return CallTreeNode(
        name=name,
        backend=backend,
        children=children,
        kernels=kernels,
        kernel_launches=99,
        total_duration_ms=999.0,
        min_dispatch_ns=123.0,
        max_dispatch_ns=456.0,
        mean_dispatch_ns=789.0,
        invocation_ids=invocation_ids,
        args_invocations={"(self=float32[2x2])": set(invocation_ids)},
        file_name="model.py",
        line_number=10,
        start_timestamp=100.0,
        end_timestamp=200.0,
        thread_id="7",
        launcher_thread_id="9",
    )


def build_display_kernel(duration_ms: float, kernel_id: int) -> KernelStats:
    # kernel_id only varies the PMC keys; ids shown are passed to the printers.
    return KernelStats(
        launches=1,
        total_duration_ns=duration_ms * 1_000_000.0,
        min_duration_ns=duration_ms * 1_000_000.0,
        max_duration_ns=duration_ms * 1_000_000.0,
        pmc_group_keys=[
            ("guid", kernel_id, 256, 64, 0),
            ("other-guid", kernel_id + 1, 512, 64, 1024),
        ],
    )


def build_display_forest() -> CallTreeForest:
    epilogue = build_display_node(
        "triton_epilogue",
        "triton",
        {"unmapped_triton": build_display_kernel(4.0, 74)},
        [],
    )
    triton_node = build_display_node(
        "triton_gemm",
        "triton",
        {"shared_kernel": build_display_kernel(3.0, 73)},
        [epilogue],
    )
    torch_node = build_display_node(
        "aten::mm",
        "torch",
        {
            "  shared_kernel  ": build_display_kernel(1.0, 71),
            "unmapped_torch": build_display_kernel(2.0, 72),
        },
        [triton_node],
    )
    user_node = build_display_node(
        "capture",
        "user",
        {"user_kernel": build_display_kernel(5.0, 70)},
        [torch_node],
    )
    return {"7": [user_node]}


def operator_node(name, invocation, children=(), kernel=None):
    node = CallTreeNode(name=name, file_name="net.py", line_number=1)
    node.invocation_ids.add(invocation)
    if kernel:
        node.kernels[kernel] = KernelStats(
            launches=1,
            total_duration_ns=1000.0,
            min_duration_ns=1000.0,
            max_duration_ns=1000.0,
        )
    node.children = list(children)
    return node


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


def test_print_operator_node_branching_shows_stats(capsys):
    node = CallTreeNode(name="branch")
    node.kernel_launches = 2
    node.total_duration_ms = 5.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    node.kernels["k2"] = KernelStats(launches=1, total_duration_ns=2_500_000.0)
    print_operator_node(node, {})
    output = capsys.readouterr().out
    assert "dispatches: 2" in output
    assert "k1" in output
    assert "k2" in output


def test_print_operator_node_non_branching_omits_stats(capsys):
    node = CallTreeNode(name="single")
    node.kernel_launches = 1
    node.total_duration_ms = 1.0
    node.kernels["k1"] = KernelStats(launches=1, total_duration_ns=1_000_000.0)
    print_operator_node(node, {})
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
    )
    print_operator_node(node, {long_kernel_name: 7})
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


def test_print_operator_node_folds_identical_children(capsys):
    first = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    first.invocation_ids.add("1")
    first.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=2_000_000.0,
        min_duration_ns=2_000_000.0,
        max_duration_ns=2_000_000.0,
    )
    second = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    second.invocation_ids.add("2")
    second.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=3_000_000.0,
        min_duration_ns=3_000_000.0,
        max_duration_ns=3_000_000.0,
    )
    parent = CallTreeNode(name="forward")
    parent.invocation_ids.add("0")
    parent.children = [first, second]
    rollup_node_stats(first)
    rollup_node_stats(second)
    rollup_node_stats(parent)
    print_operator_node(parent, {})
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 1
    assert "calls: 2" in output


def test_show_call_tree_folds_identical_sibling_leaves(capsys):
    first = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    first.invocation_ids.add("1")
    first.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=2_000_000.0,
        min_duration_ns=2_000_000.0,
        max_duration_ns=2_000_000.0,
    )
    second = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    second.invocation_ids.add("2")
    second.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=3_000_000.0,
        min_duration_ns=3_000_000.0,
        max_duration_ns=3_000_000.0,
    )
    parent = CallTreeNode(name="forward")
    parent.invocation_ids.add("0")
    parent.children = [first, second]
    rollup_node_stats(first)
    rollup_node_stats(second)
    rollup_node_stats(parent)
    show_call_tree({"1": [parent]}, {})
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 1
    assert "calls: 2" in output


def test_format_node_args_present() -> None:
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={"(self=float32[2x2])": {"1@m.py:1", "2@m.py:1"}},
    )
    assert format_node_args(node) == " args=(self=float32[2x2])"


def test_format_node_args_absent() -> None:
    assert format_node_args(CallTreeNode(name="aten::mm")) == ""
    empty_parens = CallTreeNode(name="aten::mm", args_invocations={"()": set()})
    assert format_node_args(empty_parens) == ""


def test_format_args_variant_lines_orders_by_call_count() -> None:
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            "(self=float32[2x2])": {"1@m.py:1"},
            "(self=float32[4x4])": {"2@m.py:1", "3@m.py:1"},
        },
    )
    assert format_args_variant_lines(node) == [
        "args variants:",
        "  2 calls  (self=float32[4x4])",
        "  1 call   (self=float32[2x2])",
    ]


def test_format_args_variant_lines_empty_below_two_variants() -> None:
    assert format_args_variant_lines(CallTreeNode(name="aten::mm")) == []
    single = CallTreeNode(
        name="aten::mm", args_invocations={"(self=float32[2x2])": {"1@m.py:1"}}
    )
    assert format_args_variant_lines(single) == []


def test_format_args_variant_lines_caps_variant_count() -> None:
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            f"(self=float32[{index}x{index}])": {str(index)} for index in range(7)
        },
    )
    lines = format_args_variant_lines(node)
    assert lines[0] == "args variants:"
    assert len(lines) == 7
    assert lines[-1] == "  ... 2 more variants"


def test_format_args_variant_lines_omits_counts_without_invocation_ids() -> None:
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            "(self=float32[2x2])": set(),
            "(self=float32[4x4])": set(),
        },
    )
    assert format_args_variant_lines(node) == [
        "args variants:",
        "    (self=float32[2x2])",
        "    (self=float32[4x4])",
    ]


def test_print_operator_node_shows_args(capsys) -> None:
    node = CallTreeNode(
        name="aten::mm", args_invocations={"(self=float32[2x2])": {"1@m.py:1"}}
    )
    node.kernels["kernel_gemm"] = KernelStats(launches=1, total_duration_ns=1000.0)
    print_operator_node(node, {})
    output = capsys.readouterr().out
    assert "aten::mm" in output
    assert "args=(self=float32[2x2])" in output


def test_print_operator_node_shows_args_variants_block(capsys) -> None:
    node = CallTreeNode(
        name="aten::mm",
        args_invocations={
            "(self=float32[2x2])": {"1@m.py:1"},
            "(self=float32[4x4])": {"2@m.py:1", "3@m.py:1"},
        },
    )
    node.kernels["kernel_gemm"] = KernelStats(launches=1, total_duration_ns=1000.0)
    print_operator_node(node, {})
    output = capsys.readouterr().out
    assert "args=" not in output
    assert "args variants:" in output
    assert "2 calls  (self=float32[4x4])" in output
    assert "1 call   (self=float32[2x2])" in output


def test_build_continuation_indent_keeps_pipes() -> None:
    assert build_continuation_indent("   |  └─ ", "   |  ") == "   |     "
    assert build_continuation_indent("   |  ", "   |  ") == "   |  "
    assert build_continuation_indent("└─ ", "") == "   "


def test_show_call_tree_does_not_fold_roots_across_threads(capsys):
    first = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    first.invocation_ids.add("1")
    first.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=2_000_000.0,
        min_duration_ns=2_000_000.0,
        max_duration_ns=2_000_000.0,
    )
    second = CallTreeNode(name="aten::addmm", file_name="net.py", line_number=10)
    second.invocation_ids.add("2")
    second.kernels["addmm"] = KernelStats(
        launches=1,
        total_duration_ns=3_000_000.0,
        min_duration_ns=3_000_000.0,
        max_duration_ns=3_000_000.0,
    )
    rollup_node_stats(first)
    rollup_node_stats(second)
    show_call_tree({"1": [first], "2": [second]}, {})
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 2


# show_operator_summary


def test_show_operator_summary_empty_prints_no_dispatches_message(capsys):
    show_operator_summary(pd.DataFrame(columns=OPERATOR_SUMMARY_COLUMNS))
    output = capsys.readouterr().out
    assert "no operators with recorded dispatches" in output


def test_show_call_tree_prints_file_line_sort_and_kernel_id(capsys):
    """Constructed nodes: file:line, duration order, and kernel_id print."""
    longer = CallTreeNode(name="nn.Module.SimpleModel.forward")
    longer.total_duration_ms = 8.0
    longer.invocation_ids.add("0")
    child = CallTreeNode(
        name="nn.Module.Linear.forward",
        file_name="simple_torch_code.py",
        line_number=19,
    )
    child.total_duration_ms = 5.0
    child.invocation_ids.add("10")
    child.kernels["addmm_kernel"] = KernelStats(
        launches=1,
        total_duration_ns=5_000_000.0,
    )
    longer.children = [child]
    shorter = CallTreeNode(name="aten::relu")
    shorter.total_duration_ms = 1.0
    shorter.invocation_ids.add("90")
    show_call_tree({"1": [longer], "2": [shorter]}, {"addmm_kernel": 4})
    output = capsys.readouterr().out
    assert "Thread_Id" not in output
    assert output.index("nn.Module.SimpleModel.forward") < output.index("aten::relu")
    assert "nn.Module.Linear.forward simple_torch_code.py:19" in output
    assert "aten::relu simple_torch_code.py" not in output
    assert "(id 4)" in output


def test_format_operator_args_caps_item_count():
    args_blob = "(" + ", ".join(f"a{i}={i}" for i in range(12)) + ")"
    formatted_args = format_operator_args(args_blob, max_items=3, max_chars=200)
    assert formatted_args == "(a0=0, a1=1, a2=2, ...)"


def test_format_operator_args_caps_length():
    long_token = "x" * 80
    formatted_args = format_operator_args(f"({long_token})", max_items=8, max_chars=20)
    assert len(formatted_args) <= 20
    assert formatted_args.endswith("...)")


def test_format_operator_args_empty_blob():
    assert format_operator_args("") == ""
    assert format_operator_args("()") == ""


@pytest.mark.parametrize(
    ("backends", "framework_labels", "kernel_lines"),
    [
        pytest.param(
            ["torch"],
            ["PyTorch"],
            {"shared_kernel": [4], "unmapped_torch": [72]},
            id="torch",
        ),
        pytest.param(
            ["triton"],
            ["Triton"],
            {"shared_kernel": [4], "unmapped_triton": [74]},
            id="triton-keeps-its-ancestors",
        ),
        pytest.param(
            ["torch", "triton"],
            ["PyTorch", "Triton"],
            {
                "shared_kernel": [4, 4],
                "unmapped_torch": [72],
                "unmapped_triton": [74],
            },
            id="combined-backends-share-top-kernel-id",
        ),
        pytest.param(
            [],
            ["PyTorch", "Triton"],
            {
                "user_kernel": [0],
                "shared_kernel": [4, 4],
                "unmapped_torch": [72],
                "unmapped_triton": [74],
            },
            id="unfiltered-lists-all",
        ),
    ],
)
def test_list_ml_operators_prints_tree_and_summary_with_kernel_ids(
    capsys, backends, framework_labels, kernel_lines
) -> None:
    forest = build_display_forest()
    name_to_id = {
        "user_kernel": 0,
        "shared_kernel": 4,
        "unmapped_torch": 72,
        "unmapped_triton": 74,
        "absent_kernel": 8,
    }

    list_ml_operators(
        "workloads/model",
        forest,
        framework_labels,
        backends,
        name_to_id,
    )
    first_output = capsys.readouterr().out
    list_ml_operators(
        "workloads/model",
        forest,
        framework_labels,
        backends,
        name_to_id,
    )
    assert capsys.readouterr().out == first_output
    assert (
        f"{', '.join(framework_labels)} Operator Call Tree: workloads/model"
        in first_output
    )
    assert "Sorted by total GPU kernel duration." in first_output
    assert "Operator summary (Min/Max/Mean are per-dispatch" in first_output
    assert "capture model.py:10" in first_output
    assert "aten::mm model.py:10" in first_output
    assert "args=(self=float32[2x2])" in first_output

    for kernel_name in (
        "user_kernel",
        "shared_kernel",
        "unmapped_torch",
        "unmapped_triton",
    ):
        rendered_lines = [
            line for line in first_output.splitlines() if f"{kernel_name} (id " in line
        ]
        expected_ids = kernel_lines.get(kernel_name, [])
        assert len(rendered_lines) == len(expected_ids)
        for rendered_line, expected_id in zip(rendered_lines, expected_ids):
            assert f"(id {expected_id})" in rendered_line


@pytest.mark.parametrize(
    ("backends", "use_forest"),
    [
        pytest.param([], False, id="empty-unfiltered"),
        pytest.param(["torch"], False, id="empty-filtered"),
        pytest.param(["unknown-backend"], True, id="backend-excludes-every-node"),
    ],
)
def test_list_ml_operators_empty_view_prints_count(capsys, backends, use_forest):
    forest = build_display_forest() if use_forest else {}
    list_ml_operators("workloads/model", forest, ["PyTorch"], backends, {})
    assert capsys.readouterr().out == (
        "\nPyTorch Operators in: workloads/model\nTotal: 0 operators\n"
    )


@pytest.mark.parametrize(
    ("framework_labels", "patterns_str"),
    [
        pytest.param(["PyTorch"], "aten::*", id="torch-pattern"),
        pytest.param(
            ["PyTorch", "Triton"],
            "aten::*, triton_*",
            id="combined-patterns",
        ),
    ],
)
def test_show_matched_operators_prints_selected_totals_and_kernel_ids(
    capsys, framework_labels, patterns_str
) -> None:
    forest = build_display_forest()
    kernel_ids = {
        "user_kernel": 70,
        "shared_kernel": 71,
        "unmapped_torch": 72,
        "unmapped_triton": 74,
    }

    rollup_node_stats(forest["7"][0])
    show_matched_operators(forest, framework_labels, patterns_str, kernel_ids)
    first_output = capsys.readouterr().out
    assert (
        f"Matched {', '.join(framework_labels)} Operators: {patterns_str}"
        in first_output
    )
    assert "Sorted by total GPU kernel duration." in first_output
    assert "Operator summary (Min/Max/Mean are per-dispatch" in first_output
    assert "dispatches: 5, total: 15.00 ms" in first_output
    assert "999.00 ms" not in first_output
    for kernel_id in (70, 71, 72, 74):
        assert f"(id {kernel_id})" in first_output
    assert "(id 73)" not in first_output


@pytest.mark.parametrize(
    ("backends", "expected_kernels", "heading"),
    [
        pytest.param(
            [],
            [
                "user_kernel",
                "addmm_kernel",
                "nested_kernel",
                "triton_kernel",
                "outside_kernel",
                "scaffolding_kernel",
            ],
            "ML API",
            id="bare-lists-all",
        ),
        pytest.param(
            ["user"],
            ["user_kernel", "addmm_kernel", "nested_kernel", "triton_kernel"],
            "User",
            id="user-retains-mixed-subtree",
        ),
        pytest.param(
            ["triton"], ["triton_kernel"], "Triton", id="framework-clears-scaffolding"
        ),
    ],
)
def test_list_ml_operators_lists_mixed_subtree_by_backend(
    capsys, mixed_user_region_forest, backends, expected_kernels, heading
):
    forest = mixed_user_region_forest
    name_to_id = {name: index for index, name in enumerate(expected_kernels)}

    list_ml_operators(
        "/workload", forest, [operator_backend_labels(backends)], backends, name_to_id
    )
    first_output = capsys.readouterr().out

    assert f"{heading} Operator Call Tree: /workload" in first_output
    for kernel_name in (
        "user_kernel",
        "addmm_kernel",
        "nested_kernel",
        "triton_kernel",
        "outside_kernel",
        "scaffolding_kernel",
    ):
        assert (f"{kernel_name} (id " in first_output) == (
            kernel_name in expected_kernels
        )


def test_show_matched_operator_report_names_the_bound_patterns(
    capsys, mixed_user_region_forest
):
    workload = Workload(
        ml_api_call_trees=mixed_user_region_forest,
        operator_patterns={
            "triton": ["*common*", "*triton*"],
            "user": ["*common*", "region1"],
        },
    )

    show_matched_operator_report(workload)

    output = capsys.readouterr().out
    assert "Matched Triton, User Operators: *common*, *triton*, region1" in output


def test_call_tree_folds_siblings_again_at_each_level(capsys):
    """Calls of A whose children repeat B a different number of times fold again."""

    first_a = operator_node(
        "A",
        "1",
        [operator_node("B", "11", kernel="k"), operator_node("B", "12", kernel="k")],
    )
    second_a = operator_node("A", "2", [operator_node("B", "21", kernel="k")])
    parent = operator_node("P", "0", [first_a, second_a])
    rollup_node_stats(parent)

    show_call_tree({"1": [parent]}, {})

    output = capsys.readouterr().out
    assert output.count("A net.py:1") == 1
    assert "A net.py:1 (calls: 2)" in output
    assert "B net.py:1 (calls: 3)" in output


# build_operator_summary


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
        total_gpu_ms=20.0,
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
    assert summary.loc["operator", "Total_GPU"] == 22.0
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


def test_print_operator_node_handles_nesting_deeper_than_the_recursion_limit(capsys):
    # Printing folds every level, so its cost grows with the square of the
    # depth. A lower limit keeps this test fast while the depth stays above it.
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(250)
    try:
        nesting = 500
        node = CallTreeNode(
            name="leaf",
            kernels={"gemm": KernelStats(1, 10.0, 10.0, 10.0)},
            invocation_ids={"0"},
        )
        for level in range(nesting - 1):
            node = CallTreeNode(
                name=f"op{level}", children=[node], invocation_ids={str(level)}
            )
        rollup_node_stats(node)

        print_operator_node(node, {"gemm": 0})
    finally:
        sys.setrecursionlimit(original_limit)

    output = capsys.readouterr().out
    operator_lines = [
        line for line in output.splitlines() if "└─ op" in line or "└─ leaf" in line
    ]
    assert len(operator_lines) == nesting
    assert "leaf" in operator_lines[-1]
