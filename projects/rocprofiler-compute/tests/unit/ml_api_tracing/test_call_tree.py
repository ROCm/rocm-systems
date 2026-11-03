# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API call-tree processing."""

import numpy as np
import pandas as pd
import pytest

from ml_api_tracing.call_tree import (
    attach_unlocated_trees_by_launcher_thread,
    iter_call_tree_paths,
    nest_marker_intervals,
    prune_cpu_only_call_trees,
    record_missing_source_location_errors,
    rollup_node_stats,
    validate_all_markers_nested,
)
from ml_api_tracing.errors import (
    KernelSequenceLengthMismatchError,
    LauncherThreadNotFoundError,
    MarkerNotNestedError,
    MissingSourceLocationError,
    OverlappingMarkerRangeError,
    UncorrelatedLauncherIntervalError,
)
from ml_api_tracing.models import CallTreeNode, KernelStats, NodeRollup


def parsed_marker_row(
    operator_name,
    start,
    end,
    thread_id=1,
    file_name="",
    line_number="",
    backend="torch",
    kernel_names=None,
    kernel_starts=None,
    kernel_ends=None,
    launcher_thread_id="",
):
    """One already-parsed marker interval for nest_marker_intervals."""
    return {
        "Operator_Name": operator_name,
        "Thread_Id": thread_id,
        "Start_Timestamp": start,
        "End_Timestamp": end,
        "File_Name": file_name,
        "Line_Number": line_number,
        "Backend": backend,
        "Kernel_Names": [] if kernel_names is None else kernel_names,
        "Kernel_Start_Timestamps": [] if kernel_starts is None else kernel_starts,
        "Kernel_End_Timestamps": [] if kernel_ends is None else kernel_ends,
        "launcher_thread_id": launcher_thread_id,
        "Kernel_Pmc_Group_Keys": [],
    }


def test_kernel_stats_length_mismatch_records_error():
    trace_df = pd.DataFrame([
        parsed_marker_row(
            "aten::addmm",
            0,
            20,
            kernel_names=["gemm", "add"],
            kernel_starts=[0, 10],
            kernel_ends=[5],
        )
    ])
    errors = []
    forest = nest_marker_intervals(trace_df, errors)
    assert forest["1"][0].kernels == {}
    assert len(errors) == 1
    assert isinstance(errors[0], KernelSequenceLengthMismatchError)


def test_nest_marker_intervals_records_operator_args():
    trace_df = pd.DataFrame({
        "Thread_Id": ["1"],
        "Start_Timestamp": [0.0],
        "End_Timestamp": [10.0],
        "Operator_Name": ["aten::mm"],
        "args": ["(self=float32[2x2])"],
    })
    forest = nest_marker_intervals(trace_df)
    node = forest["1"][0]
    assert node.args_invocations == {"(self=float32[2x2])": {"0.0"}}


def test_nest_marker_intervals_skips_unavailable_args():
    trace_df = pd.DataFrame({
        "Thread_Id": ["1"],
        "Start_Timestamp": [0.0],
        "End_Timestamp": [10.0],
        "Operator_Name": ["aten::mm"],
        "args": ["n/a"],
    })
    forest = nest_marker_intervals(trace_df)
    assert forest["1"][0].args_invocations == {}


def test_attach_defers_uncorrelated_interval_and_keeps_worker_root():
    backward = CallTreeNode(
        name="torch.Tensor.backward",
        file_name="simple.py",
        line_number=28,
        backend="torch",
        start_timestamp=0.0,
        end_timestamp=10.0,
    )
    backward.invocation_ids.add("bw")
    child = CallTreeNode(
        name="SumBackward0",
        backend="torch",
        start_timestamp=51.0,
        end_timestamp=59.0,
    )
    child.invocation_ids.add("sum_bw")
    engine = CallTreeNode(
        name="autograd::engine::evaluate_function: SumBackward0",
        backend="torch",
        start_timestamp=50.0,
        end_timestamp=60.0,
        launcher_thread_id="14444",
    )
    engine.invocation_ids.add("eval")
    engine.children = [child]
    forest = {"14444": [backward], "14611": [engine]}
    errors = []
    attach_unlocated_trees_by_launcher_thread(forest, errors)
    assert len(errors) == 1
    assert isinstance(errors[0], UncorrelatedLauncherIntervalError)
    assert engine in forest["14611"]
    assert engine not in backward.children


def test_nest_overlap_collects_error_and_keeps_first_marker():
    trace_df = pd.DataFrame({
        "Thread_Id": ["1", "1"],
        "Start_Timestamp": [0.0, 5.0],
        "End_Timestamp": [10.0, 15.0],
        "Operator_Name": ["outer", "overlap"],
    })
    errors = []
    forest = nest_marker_intervals(trace_df, errors)
    assert len(errors) == 1
    assert isinstance(errors[0], OverlappingMarkerRangeError)
    assert [node.name for node in forest["1"]] == ["outer"]


def test_nest_overlap_raises_without_error_list():
    trace_df = pd.DataFrame({
        "Thread_Id": ["1", "1"],
        "Start_Timestamp": [0.0, 5.0],
        "End_Timestamp": [10.0, 15.0],
        "Operator_Name": ["outer", "overlap"],
    })
    with pytest.raises(OverlappingMarkerRangeError):
        nest_marker_intervals(trace_df)


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


def test_nest_marker_intervals_three_deep_with_file_line():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row(
                "nn.Module.SimpleModel.forward",
                0,
                100,
                file_name="simple_torch_code.py",
                line_number=27,
            ),
            parsed_marker_row(
                "nn.Module.Linear.forward",
                10,
                80,
                file_name="simple_torch_code.py",
                line_number=19,
            ),
            parsed_marker_row("aten::addmm", 20, 70),
        ])
    )
    roots = forest["1"]
    assert len(roots) == 1
    simple = roots[0]
    linear = simple.children[0]
    addmm = linear.children[0]
    assert simple.name == "nn.Module.SimpleModel.forward"
    assert simple.file_name == "simple_torch_code.py"
    assert simple.line_number == 27
    assert linear.name == "nn.Module.Linear.forward"
    assert linear.file_name == "simple_torch_code.py"
    assert linear.line_number == 19
    assert addmm.name == "aten::addmm"
    assert addmm.file_name is None
    assert addmm.line_number is None


def test_nest_marker_intervals_nests_when_child_row_precedes_parent():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row("aten::addmm", 20, 70),
            parsed_marker_row(
                "nn.Module.Linear.forward",
                10,
                80,
                file_name="simple_torch_code.py",
                line_number=19,
            ),
        ])
    )
    assert forest["1"][0].name == "nn.Module.Linear.forward"
    assert [child.name for child in forest["1"][0].children] == ["aten::addmm"]


def test_nest_marker_intervals_same_start_nests_contained_interval():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row("outer", 10, 100),
            parsed_marker_row("inner", 10, 40),
        ])
    )
    assert [node.name for node in forest["1"]] == ["outer"]
    assert [child.name for child in forest["1"][0].children] == ["inner"]


def test_nest_marker_intervals_disjoint_linear_siblings():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row(
                "nn.Module.SimpleModel.forward",
                0,
                200,
                file_name="simple_torch_code.py",
                line_number=27,
            ),
            parsed_marker_row(
                "nn.Module.Linear.forward",
                10,
                80,
                file_name="simple_torch_code.py",
                line_number=19,
            ),
            parsed_marker_row(
                "nn.Module.Linear.forward",
                90,
                160,
                file_name="simple_torch_code.py",
                line_number=19,
            ),
        ])
    )
    siblings = forest["1"][0].children
    assert [child.name for child in siblings] == [
        "nn.Module.Linear.forward",
        "nn.Module.Linear.forward",
    ]
    assert siblings[0].call_count == 1
    assert siblings[1].call_count == 1
    assert siblings[0] is not siblings[1]


def test_nest_marker_intervals_adjacent_ranges_are_siblings():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row("A", 0, 50, backend="user"),
            parsed_marker_row("B", 50, 80, backend="user"),
        ])
    )
    roots = forest["1"]
    assert [root.name for root in roots] == ["A", "B"]


def test_nest_marker_intervals_tensor_backward_and_autograd_worker():
    forest = nest_marker_intervals(
        pd.DataFrame([
            parsed_marker_row(
                "torch.Tensor.backward",
                0,
                100,
                file_name="simple_torch_code.py",
                line_number=52,
            ),
            parsed_marker_row(
                "autograd::engine::evaluate_function: AddmmBackward0",
                10,
                40,
                thread_id=2,
                launcher_thread_id="1",
            ),
        ])
    )
    assert forest["1"][0].name == "torch.Tensor.backward"
    worker = forest["2"][0]
    assert worker.name == "autograd::engine::evaluate_function: AddmmBackward0"
    errors = []
    attach_unlocated_trees_by_launcher_thread(forest, errors)
    assert errors == []
    assert worker in forest["1"][0].children
    assert "2" not in forest


def test_nest_marker_intervals_user_root_without_file():
    forest = nest_marker_intervals(
        pd.DataFrame([parsed_marker_row("training_loop", 0, 100, backend="user")])
    )
    root = forest["1"][0]
    assert root.name == "training_loop"
    assert root.backend == "user"
    assert root.file_name is None


def test_nest_marker_intervals_torch_root_without_file_stays_unlocated():
    forest = nest_marker_intervals(
        pd.DataFrame([parsed_marker_row("aten::addmm", 0, 50)])
    )
    root = forest["1"][0]
    assert root.name == "aten::addmm"
    assert root.file_name is None
    errors = []
    attach_unlocated_trees_by_launcher_thread(forest, errors)
    assert errors == []
    record_missing_source_location_errors(forest, errors)
    assert len(errors) == 1
    assert isinstance(errors[0], MissingSourceLocationError)
    assert errors[0].operator_name == "aten::addmm"
    assert root in forest["1"]


def test_nest_marker_intervals_empty_frame():
    assert nest_marker_intervals(pd.DataFrame()) == {}


def test_nest_marker_intervals_does_not_mutate_unsorted_input():
    trace_df = pd.DataFrame([
        parsed_marker_row("child", 20, 40),
        parsed_marker_row("parent", 0, 100),
    ])
    original = trace_df.copy(deep=True)

    forest = nest_marker_intervals(trace_df)

    pd.testing.assert_frame_equal(trace_df, original)
    assert forest["1"][0].children[0].name == "child"


@pytest.mark.parametrize(
    ("kernel_names", "kernel_starts", "kernel_ends", "pmc_group_keys"),
    [
        (["gemm"], [10], [15], [("gpu", 1)]),
        (("gemm",), (10,), (15,), (("gpu", 1),)),
        (
            np.array(["gemm"]),
            np.array([10]),
            np.array([15]),
            np.array([["gpu", 1]], dtype=object),
        ),
        (
            np.array("gemm"),
            np.array(10),
            np.array(15),
            [("gpu", 1)],
        ),
        ("gemm", 10, 15, [("gpu", 1)]),
    ],
    ids=["lists", "tuples", "arrays", "scalar-arrays", "scalars"],
)
def test_nest_marker_intervals_preserves_kernel_sequences_and_pmc_keys(
    kernel_names, kernel_starts, kernel_ends, pmc_group_keys
):
    marker_row = parsed_marker_row(
        "operator",
        0,
        20,
        kernel_names=kernel_names,
        kernel_starts=kernel_starts,
        kernel_ends=kernel_ends,
    )
    marker_row["Kernel_Pmc_Group_Keys"] = pmc_group_keys

    forest = nest_marker_intervals(pd.DataFrame([marker_row]))

    assert forest["1"][0].kernels["gemm"] == KernelStats(
        launches=1,
        total_duration_ns=5.0,
        min_duration_ns=5.0,
        max_duration_ns=5.0,
        pmc_group_keys=[("gpu", 1)],
    )


@pytest.mark.parametrize("kernel_end", [None, "invalid", 10, 5])
def test_nest_marker_intervals_preserves_nonpositive_and_invalid_durations(kernel_end):
    marker_row = parsed_marker_row(
        "operator",
        0,
        20,
        kernel_names=["gemm"],
        kernel_starts=[10],
        kernel_ends=[kernel_end],
    )
    marker_row["Kernel_Pmc_Group_Keys"] = [("gpu", 1)]

    forest = nest_marker_intervals(pd.DataFrame([marker_row]))

    stats = forest["1"][0].kernels["gemm"]
    assert stats.launches == 1
    assert stats.total_duration_ns == (-5.0 if kernel_end == 5 else 0.0)
    assert stats.min_duration_ns is None
    assert stats.max_duration_ns is None


def test_nest_marker_intervals_collector_keeps_later_markers_after_fatal_errors():
    malformed = parsed_marker_row(
        "malformed",
        0,
        20,
        kernel_names=["gemm"],
        kernel_starts=[1],
        kernel_ends=[],
    )
    second_malformed = parsed_marker_row(
        "second-malformed",
        30,
        50,
        kernel_names=["gemm"],
        kernel_starts=[31],
        kernel_ends=[40],
    )
    trace_df = pd.DataFrame([
        malformed,
        second_malformed,
        parsed_marker_row("valid", 60, 80),
    ])
    errors = []

    forest = nest_marker_intervals(trace_df, errors)

    assert [error.operator_name for error in errors] == [
        "malformed",
        "second-malformed",
    ]
    assert all(isinstance(error, KernelSequenceLengthMismatchError) for error in errors)
    assert [root.name for root in forest["1"]] == [
        "malformed",
        "second-malformed",
        "valid",
    ]
    with pytest.raises(KernelSequenceLengthMismatchError) as raised:
        nest_marker_intervals(trace_df)
    assert raised.value.operator_name == "malformed"


def test_validate_all_markers_nested_uses_original_worker_thread_after_attachment():
    trace_df = pd.DataFrame([
        parsed_marker_row("parent", 0, 100, file_name="model.py"),
        parsed_marker_row("worker", 10, 20, thread_id=2, launcher_thread_id="1"),
    ])
    forest = nest_marker_intervals(trace_df)
    errors = []
    attach_unlocated_trees_by_launcher_thread(forest, errors)

    validate_all_markers_nested(trace_df, forest, errors)

    assert errors == []
    assert forest["1"][0].children[0].thread_id == "2"


def test_validate_all_markers_nested_does_not_repeat_skipped_overlap():
    trace_df = pd.DataFrame([
        parsed_marker_row("first", 0, 10),
        parsed_marker_row("overlap", 5, 15),
    ])
    errors = []
    skipped_keys = set()
    forest = nest_marker_intervals(trace_df, errors, skipped_keys)

    validate_all_markers_nested(trace_df, forest, errors, skipped_keys)

    assert skipped_keys == {("1", "5")}
    assert len(errors) == 1
    assert isinstance(errors[0], OverlappingMarkerRangeError)


def test_validate_all_markers_nested_collector_and_first_fatal_order():
    trace_df = pd.DataFrame([
        parsed_marker_row("first", 0, 10),
        parsed_marker_row("second", 20, 30),
    ])
    errors = []

    validate_all_markers_nested(trace_df, {}, errors)

    assert [error.operator_name for error in errors] == ["first", "second"]
    assert all(isinstance(error, MarkerNotNestedError) for error in errors)
    with pytest.raises(MarkerNotNestedError) as raised:
        validate_all_markers_nested(trace_df, {})
    assert raised.value.operator_name == "first"


def test_attach_collects_coverage_errors_and_continues_in_forest_order():
    trace_df = pd.DataFrame([
        parsed_marker_row("parent", 0, 10, file_name="model.py"),
        parsed_marker_row(
            "missing-launcher", 1, 5, thread_id=2, launcher_thread_id="9"
        ),
        parsed_marker_row(
            "outside-parent", 20, 30, thread_id=3, launcher_thread_id="1"
        ),
        parsed_marker_row("attached", 1, 5, thread_id=4, launcher_thread_id="1"),
    ])
    forest = nest_marker_intervals(trace_df)
    errors = []

    returned = attach_unlocated_trees_by_launcher_thread(forest, errors)

    assert returned is None
    assert [type(error) for error in errors] == [
        LauncherThreadNotFoundError,
        UncorrelatedLauncherIntervalError,
    ]
    assert [error.operator_name for error in errors] == [
        "missing-launcher",
        "outside-parent",
    ]
    assert list(forest) == ["1", "2", "3"]
    assert forest["1"][0].children[0].name == "attached"


def test_attach_uses_descendant_launcher_and_deepest_containing_parent():
    trace_df = pd.DataFrame([
        parsed_marker_row("parent", 0, 100, file_name="model.py"),
        parsed_marker_row("inner-parent", 10, 80, file_name="model.py"),
        parsed_marker_row("worker", 20, 40, thread_id=2),
        parsed_marker_row("worker-child", 25, 35, thread_id=2, launcher_thread_id="1"),
    ])
    forest = nest_marker_intervals(trace_df)
    worker = forest["2"][0]
    errors = []

    attach_unlocated_trees_by_launcher_thread(forest, errors)

    assert errors == []
    assert forest["1"][0].children[0].children == [worker]
    assert "2" not in forest


def test_missing_source_collects_backend_roots_in_order():
    forest = {
        "1": [
            CallTreeNode("located", backend="torch", file_name="model.py"),
            CallTreeNode("user", backend="user"),
            CallTreeNode("launcher", backend="torch", launcher_thread_id="2"),
            CallTreeNode("torch", backend="torch", start_timestamp=10),
            CallTreeNode("triton", backend="triton", start_timestamp=20),
        ]
    }
    errors = []

    returned = record_missing_source_location_errors(forest, errors)

    assert returned is None
    assert [error.operator_name for error in errors] == ["torch", "triton"]
    assert all(isinstance(error, MissingSourceLocationError) for error in errors)
    assert [error.start_timestamp for error in errors] == [10.0, 20.0]
    assert [root.name for root in forest["1"]] == [
        "located",
        "user",
        "launcher",
        "torch",
        "triton",
    ]


def test_prune_cpu_only_call_trees_keeps_ancestors_of_kernel_work():
    gpu_leaf = CallTreeNode("gpu", kernels={"gemm": KernelStats(launches=1)})
    parent = CallTreeNode("parent", children=[CallTreeNode("cpu-child"), gpu_leaf])
    forest = {"1": [CallTreeNode("cpu-root"), parent], "2": [CallTreeNode("cpu")]}

    returned = prune_cpu_only_call_trees(forest)

    assert returned is None
    assert forest == {"1": [parent]}
    assert parent.children == [gpu_leaf]


def test_rollup_includes_children_and_updates_existing_aggregate_fields():
    child = CallTreeNode("child", kernels={"gemm": KernelStats(1, 10.0, 10.0, 10.0)})
    parent = CallTreeNode(
        "parent",
        children=[child],
        kernels={"add": KernelStats(2, 30.0, 5.0, 25.0)},
        kernel_launches=99,
        total_duration_ms=99.0,
    )

    rolled = rollup_node_stats(parent)

    assert rolled == NodeRollup(3, 40.0, 5.0, 25.0)
    assert parent.kernel_launches == 3
    assert parent.total_duration_ms == pytest.approx(0.00004)
    assert parent.mean_dispatch_ns == pytest.approx(40.0 / 3)
    assert child.kernel_launches == 1


def test_iter_call_tree_paths_visits_depth_first_in_sibling_order():
    grandchild = CallTreeNode("grandchild")
    child = CallTreeNode("child", children=[grandchild])
    root = CallTreeNode("root", children=[child, CallTreeNode("sibling")])

    walked = list(iter_call_tree_paths([root], ("prefix",)))

    assert [(node.name, path) for node, path in walked] == [
        ("root", ("prefix", "root")),
        ("child", ("prefix", "root", "child")),
        ("grandchild", ("prefix", "root", "child", "grandchild")),
        ("sibling", ("prefix", "root", "sibling")),
    ]
