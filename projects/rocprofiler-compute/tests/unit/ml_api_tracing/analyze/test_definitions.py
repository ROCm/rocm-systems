# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for the ML API call-tree records."""

from ml_api_tracing.analyze.definitions import CallTreeNode, KernelStats


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
