# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API operator display and argument formatting."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
from typing import Dict, List, Tuple

import pandas as pd
import pytest

from ml_api_tracing.call_tree import (
    nest_marker_intervals,
    rollup_node_stats,
    validate_all_markers_nested,
)
from ml_api_tracing.display import (
    build_continuation_indent,
    format_args_variant_lines,
    format_duration,
    format_node_args,
    format_node_stats,
    format_operator_args,
    list_ml_operators,
    print_operator_node,
    show_call_tree,
    show_matched_operators,
    show_operator_summary,
)
from ml_api_tracing.models import CallTreeForest, CallTreeNode, KernelStats
from ml_api_tracing.selection import select_operators

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
    return KernelStats(
        launches=1,
        total_duration_ns=duration_ms * 1_000_000.0,
        min_duration_ns=duration_ms * 1_000_000.0,
        max_duration_ns=duration_ms * 1_000_000.0,
        kernel_id=kernel_id,
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


def build_deep_display_forest(depth: int) -> CallTreeForest:
    """Build valid nested marker intervals with dispatches at the deepest operator."""
    marker_rows = []
    for index in range(depth):
        has_dispatches = index == depth - 1
        marker_rows.append({
            "Thread_Id": "7",
            "Start_Timestamp": index * 1_000_000,
            "End_Timestamp": (2 * depth - index) * 1_000_000,
            "Operator_Name": f"deep_op_{index:03}",
            "Backend": "torch",
            "File_Name": "model.py",
            "Line_Number": index + 1,
            "launcher_thread_id": "9",
            "args": "(self=float32[2x2])",
            "Kernel_Names": ["leaf", "extra"] if has_dispatches else [],
            "Kernel_Start_Timestamps": [depth * 1_000_000] * 2
            if has_dispatches
            else [],
            "Kernel_End_Timestamps": [
                (depth + 1) * 1_000_000,
                (depth + 0.5) * 1_000_000,
            ]
            if has_dispatches
            else [],
            "Kernel_Pmc_Group_Keys": [
                ("guid", "leaf", 256, 64, 0),
                ("guid", "extra", 512, 64, 1024),
            ]
            if has_dispatches
            else [],
        })
    marker_frame = pd.DataFrame(marker_rows)
    forest = nest_marker_intervals(marker_frame)
    validate_all_markers_nested(marker_frame, forest)
    deepest = forest["7"][0]
    while deepest.children:
        deepest = deepest.children[0]
    deepest.kernels["leaf"].kernel_id = 70
    deepest.kernels["extra"].kernel_id = 71
    rollup_node_stats(forest["7"][0])
    return forest


def snapshot_display_forest(forest: CallTreeForest) -> List[Tuple]:
    """Flatten every declared field and collection without recursive tree copying.

    Object identities preserve evidence of replaced nodes, containers, or payloads.
    """
    snapshots = []
    pending = [forest]
    visited = set()
    while pending:
        payload = pending.pop()
        if id(payload) in visited:
            continue
        visited.add(id(payload))
        if is_dataclass(payload):
            entries = [
                (declared_field.name, getattr(payload, declared_field.name))
                for declared_field in fields(payload)
            ]
            snapshots.append((
                id(payload),
                type(payload),
                tuple((name, id(value)) for name, value in entries),
            ))
            pending.extend(value for _, value in entries)
        elif isinstance(payload, dict):
            snapshots.append((
                id(payload),
                type(payload),
                tuple((id(key), id(value)) for key, value in payload.items()),
            ))
            pending.extend(payload.keys())
            pending.extend(payload.values())
        elif isinstance(payload, (list, tuple, set, frozenset)):
            values = sorted(payload, key=id) if isinstance(payload, set) else payload
            snapshots.append((id(payload), type(payload), tuple(map(id, values))))
            pending.extend(values)
        else:
            snapshots.append((id(payload), type(payload), payload))
    return snapshots


def assert_complete_deep_display(output: str, depth: int, leaf_kernel_id: int) -> None:
    call_tree_output, summary_output = output.split(
        "Operator summary (Min/Max/Mean are per-dispatch", 1
    )
    call_tree_output = call_tree_output.split(
        "Sorted by total GPU kernel duration.\n", 1
    )[1]
    assert "dispatches: 2, total: 1.50 ms" in call_tree_output
    for index in range(depth):
        assert call_tree_output.count(f"deep_op_{index:03}") == 1
        assert f"model.py:{index + 1}" in call_tree_output
    assert "args=(self=float32[2x2])" in call_tree_output
    assert f"leaf (id {leaf_kernel_id})" in call_tree_output
    assert "extra (id 71)" in call_tree_output
    assert "deep_op_000" in summary_output
    assert f"deep_op_{depth - 1:03}" in summary_output


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
    print_operator_node(parent)
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 1
    assert "calls: 2" in output
    assert parent.children == [first, second]


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
    show_call_tree({"1": [parent]})
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 1
    assert "calls: 2" in output
    assert parent.children == [first, second]


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
    print_operator_node(node)
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
    print_operator_node(node)
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
    show_call_tree({"1": [first], "2": [second]})
    output = capsys.readouterr().out
    assert output.count("aten::addmm") == 2


# ---------------------------------------------------------------------------
# show_operator_summary
# ---------------------------------------------------------------------------


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
        kernel_id=4,
    )
    longer.children = [child]
    shorter = CallTreeNode(name="aten::relu")
    shorter.total_duration_ms = 1.0
    shorter.invocation_ids.add("90")
    show_call_tree({"1": [longer], "2": [shorter]})
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
            id="triton-with-ancestor-scaffolding",
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
            id="unfiltered-forest-is-copied",
        ),
    ],
)
def test_list_ml_operators_preserves_full_forest_and_repeats_output(
    capsys, backends, framework_labels, kernel_lines
) -> None:
    forest = build_display_forest()
    original = deepcopy(forest)
    name_to_id = {"user_kernel": 0, "shared_kernel": 4, "absent_kernel": 8}
    mapping_before = dict(name_to_id)
    backends_before = list(backends)
    labels_before = list(framework_labels)

    list_ml_operators(
        "workloads/model",
        forest,
        framework_labels,
        backends,
        name_to_id,
    )
    first_output = capsys.readouterr().out
    assert forest == original
    list_ml_operators(
        "workloads/model",
        forest,
        framework_labels,
        backends,
        name_to_id,
    )
    assert capsys.readouterr().out == first_output
    assert forest == original
    assert name_to_id == mapping_before
    assert backends == backends_before
    assert framework_labels == labels_before
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
    original = deepcopy(forest)
    list_ml_operators("workloads/model", forest, ["PyTorch"], backends, {})
    assert capsys.readouterr().out == (
        "\nPyTorch Operators in: workloads/model\nTotal: 0 operators\n"
    )
    assert forest == original


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
def test_show_matched_operators_rerolls_copy_and_repeats_output(
    capsys, framework_labels, patterns_str
) -> None:
    forest = build_display_forest()
    original = deepcopy(forest)

    show_matched_operators(forest, framework_labels, patterns_str)
    first_output = capsys.readouterr().out
    assert forest == original
    show_matched_operators(forest, framework_labels, patterns_str)
    assert capsys.readouterr().out == first_output
    assert forest == original
    assert (
        f"Matched {', '.join(framework_labels)} Operators: {patterns_str}"
        in first_output
    )
    assert "Sorted by total GPU kernel duration." in first_output
    assert "Operator summary (Min/Max/Mean are per-dispatch" in first_output
    assert "dispatches: 5, total: 15.00 ms" in first_output
    assert "999.00 ms" not in first_output
    for kernel_id in (70, 71, 72, 73, 74):
        assert f"(id {kernel_id})" in first_output


def test_show_matched_operators_empty_forest_prints_nothing(capsys) -> None:
    forest = {}
    show_matched_operators(forest, ["PyTorch"], "aten::*")
    assert capsys.readouterr().out == ""
    assert forest == {}


@pytest.mark.parametrize("depth", [180, 200])
def test_list_ml_operators_renders_deep_unfiltered_forest_without_mutation(
    capsys, depth
) -> None:
    forest = build_deep_display_forest(depth)
    original = snapshot_display_forest(forest)
    framework_labels = ["PyTorch"]
    backends = []
    name_to_id = {"leaf": 4}

    list_ml_operators("workloads/deep", forest, framework_labels, backends, name_to_id)
    first_output = capsys.readouterr().out
    assert snapshot_display_forest(forest) == original
    list_ml_operators("workloads/deep", forest, framework_labels, backends, name_to_id)
    assert capsys.readouterr().out == first_output
    assert snapshot_display_forest(forest) == original
    assert framework_labels == ["PyTorch"]
    assert backends == []
    assert name_to_id == {"leaf": 4}
    assert "PyTorch Operator Call Tree: workloads/deep" in first_output
    assert_complete_deep_display(first_output, depth, leaf_kernel_id=4)


@pytest.mark.parametrize("depth", [180, 200])
def test_show_matched_operators_renders_deep_selected_forest_without_mutation(
    capsys, depth
) -> None:
    forest = build_deep_display_forest(depth)
    original = snapshot_display_forest(forest)
    selected, pmc_group_keys = select_operators(forest, {"torch": ["deep_op_000"]})
    selected_original = snapshot_display_forest(selected)
    framework_labels = ["PyTorch"]
    assert pmc_group_keys == [
        ("guid", "leaf", 256, 64, 0),
        ("guid", "extra", 512, 64, 1024),
    ]

    show_matched_operators(selected, framework_labels, "deep_op_000")
    first_output = capsys.readouterr().out
    assert snapshot_display_forest(selected) == selected_original
    show_matched_operators(selected, framework_labels, "deep_op_000")
    assert capsys.readouterr().out == first_output
    assert snapshot_display_forest(selected) == selected_original
    assert snapshot_display_forest(forest) == original
    assert framework_labels == ["PyTorch"]
    assert "Matched PyTorch Operators: deep_op_000" in first_output
    assert_complete_deep_display(first_output, depth, leaf_kernel_id=70)
