# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the shared ML API call-tree models."""

from typing import Dict, List, Optional, Set, Tuple, get_type_hints

from ml_api_tracing.models import CallTreeForest, CallTreeNode, KernelStats, NodeRollup


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
    node.invocation_ids.add("ctx2")
    assert node.call_count == 2
    node.invocation_ids.remove("ctx1")
    assert node.call_count == 1
    node.invocation_ids.clear()
    assert node.call_count == 0


def test_args_variants_orders_by_call_count():
    node = CallTreeNode(name="aten::mm")
    node.args_invocations["(self=float32[2x2])"] = {"1"}
    node.args_invocations["(self=float32[4x4])"] = {"2", "3"}
    assert node.args_variants == [
        ("(self=float32[4x4])", 2),
        ("(self=float32[2x2])", 1),
    ]


def test_args_variants_recomputes_counts_and_breaks_ties_by_text():
    node = CallTreeNode(name="operator", args_invocations={"z": {"1"}, "a": {"2"}})
    assert node.args_variants == [("a", 1), ("z", 1)]
    node.args_invocations["z"].add("3")
    assert node.args_variants == [("z", 2), ("a", 1)]
    node.args_invocations.clear()
    assert node.args_variants == []


def test_call_tree_node_mutable_defaults_are_independent():
    first = CallTreeNode(name="first")
    second = CallTreeNode(name="second")
    first.children.append(CallTreeNode(name="child"))
    first.kernels["kernel"] = KernelStats(launches=1)
    first.invocation_ids.add("1")
    first.args_invocations["args"] = {"1"}
    assert second.children == []
    assert second.kernels == {}
    assert second.invocation_ids == set()
    assert second.args_invocations == {}


def test_kernel_stats_keeps_independent_pmc_keys_and_mutable_kernel_id():
    first = KernelStats()
    second = KernelStats()
    group_keys = [(1, "gemm", 1024, 256, 0), (2, "gemm", 1024, 256, 0)]
    first.pmc_group_keys.extend(group_keys)
    first.kernel_id = 9
    assert first.pmc_group_keys == group_keys
    assert first.kernel_id == 9
    assert second.pmc_group_keys == []
    assert second.kernel_id is None


def test_node_rollup_remains_mutable():
    rollup = NodeRollup(2, 3000.0, 1000.0, 2000.0)
    rollup.launches = 3
    rollup.min_dispatch_ns = None
    assert rollup.launches == 3
    assert rollup.min_dispatch_ns is None


def test_model_annotations_resolve_recursive_children_and_kernel_stats():
    node_hints = get_type_hints(CallTreeNode)
    assert node_hints["children"] == List[CallTreeNode]
    assert node_hints["kernels"] == Dict[str, KernelStats]
    assert node_hints["args_invocations"] == Dict[str, Set[str]]
    assert node_hints["mean_dispatch_ns"] == Optional[float]
    assert get_type_hints(KernelStats)["pmc_group_keys"] == List[Tuple]
    assert (
        get_type_hints(CallTreeNode.args_variants.fget)["return"]
        == List[Tuple[str, int]]
    )
    assert CallTreeForest == Dict[str, List[CallTreeNode]]


def test_models_import_fresh_with_only_stdlib_under_profile_guard(assert_stdlib_import):
    assert_stdlib_import("ml_api_tracing.models")
