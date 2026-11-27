# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML operator selection."""

import sys

import pandas as pd
import pytest

from ml_api_tracing.analyze.call_tree import iter_call_tree_paths
from ml_api_tracing.analyze.definitions import CallTreeNode, KernelStats
from ml_api_tracing.analyze.selection import (
    analyze_dispatch_ids_from_pmc_keys,
    intersect_dispatch_ids,
    intersect_dispatch_ids_by_kernel,
    kernel_ids_from_top,
    select_operators,
)

ADDMM_PMC_GROUP_KEY = (1, "addmm_kernel", 0, 0, 0)
RELU_PMC_GROUP_KEY = (2, "relu_kernel", 0, 0, 0)


def forest_node_names(forest):
    return [
        node.name
        for roots in forest.values()
        for node, _path in iter_call_tree_paths(roots)
    ]


def test_kernel_ids_from_top_numbers_rows_by_stripped_name():
    kernel_top_df = pd.DataFrame({
        "Kernel_Name": [" first_kernel ", "\tsecond_kernel\n"]
    })

    assert kernel_ids_from_top(kernel_top_df) == {"first_kernel": 0, "second_kernel": 1}


def test_operator_glob_relu_returns_relu_keys(
    simple_model_forest_with_relu_and_addmm,
):
    subtree, pmc_group_keys = select_operators(
        simple_model_forest_with_relu_and_addmm(), {"torch": ["*relu*"]}
    )
    assert pmc_group_keys == [RELU_PMC_GROUP_KEY]
    assert forest_node_names(subtree) == [
        "nn.Module.SimpleModel.forward",
        "aten::relu",
    ]


def test_operator_glob_addmm_path_returns_addmm_keys(
    simple_model_forest_with_relu_and_addmm,
):
    subtree, pmc_group_keys = select_operators(
        simple_model_forest_with_relu_and_addmm(), {"torch": ["*/aten::addmm"]}
    )
    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert forest_node_names(subtree) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


def test_operator_glob_linear_includes_descendant_addmm_keys(
    simple_model_forest_with_relu_and_addmm,
):
    subtree, pmc_group_keys = select_operators(
        simple_model_forest_with_relu_and_addmm(), {"torch": ["*Linear.forward"]}
    )
    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert forest_node_names(subtree) == [
        "nn.Module.SimpleModel.forward",
        "nn.Module.Linear.forward",
        "aten::addmm",
    ]


def test_select_operators_keeps_nodes_from_requested_backend(
    torch_parent_triton_child_forest,
):
    forest = torch_parent_triton_child_forest()

    subtree, pmc_group_keys = select_operators(forest, {"torch": ["*"]})

    assert forest_node_names(subtree) == ["nn.Module.Linear.forward"]
    assert pmc_group_keys == [(3, "torch_gemm_kernel", 0, 0, 0)]


@pytest.mark.parametrize(
    ("forest", "patterns"),
    [
        pytest.param({}, {"torch": ["*"]}, id="empty-forest"),
        pytest.param(
            {"1": [CallTreeNode(name="relu", backend="torch")]},
            {"torch": ["nomatch"]},
            id="unmatched-pattern",
        ),
        pytest.param(
            {"1": [CallTreeNode(name="relu", backend="torch")]},
            {},
            id="no-backend-patterns",
        ),
        pytest.param(
            {"1": [CallTreeNode(name="relu", backend="torch")]},
            {"triton": ["*"]},
            id="unmatched-backend",
        ),
    ],
)
def test_select_operators_no_match_returns_empty_forest_and_keys(forest, patterns):

    assert select_operators(forest, patterns) == ({}, [])


def test_select_operators_preserves_payload_and_overlapping_matches_once(
    simple_model_forest_with_relu_and_addmm,
):
    forest = simple_model_forest_with_relu_and_addmm()
    root = forest["1"][0]
    root.kernels["scaffolding_kernel"] = KernelStats(
        launches=1, total_duration_ns=100.0, pmc_group_keys=[("scaffolding",)]
    )
    addmm = root.children[0].children[0]
    addmm.invocation_ids = {"10", "20"}
    addmm.args_invocations = {"(self=float32[2x2])": {"10", "20"}}
    addmm.file_name = "model.py"
    addmm.line_number = 19
    addmm.thread_id = "1"
    addmm.launcher_thread_id = "2"
    addmm.start_timestamp = 10.0
    addmm.end_timestamp = 60.0
    addmm.kernel_launches = 1
    addmm.total_duration_ms = 50.0 * 1e-6
    addmm.min_dispatch_ns = 50.0
    addmm.max_dispatch_ns = 50.0
    addmm.mean_dispatch_ns = 50.0
    stats = addmm.kernels["addmm_kernel"]
    stats.min_duration_ns = 50.0
    stats.max_duration_ns = 50.0
    stats.pmc_group_keys.append(ADDMM_PMC_GROUP_KEY)

    subtree, pmc_group_keys = select_operators(
        forest,
        {"torch": ["*Linear.forward", "*/aten::addmm", "*/aten::addmm"]},
    )

    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY]
    assert subtree["1"][0].kernels == {}
    copied_addmm = subtree["1"][0].children[0].children[0]
    assert copied_addmm == addmm


@pytest.mark.parametrize(
    ("patterns", "expected_keys", "expected_children"),
    [
        pytest.param(
            {"user": ["region1"]},
            [("user",), ("torch",), ("nested",), ("triton",)],
            ["region1"],
            id="complete-mixed-user-subtree",
        ),
        pytest.param(
            {"user": ["region1"], "torch": ["*relu*"]},
            [("user",), ("torch",), ("nested",), ("triton",), ("outside",)],
            ["region1", "aten::relu"],
            id="union-retains-user-work-and-outside-torch",
        ),
        pytest.param(
            {"triton": ["outer/region1/nested-region/triton.*"]},
            [("triton",)],
            ["region1"],
            id="matching-uses-original-mixed-path",
        ),
    ],
)
def test_select_operators_preserves_mixed_subtrees_and_ordered_keys(
    mixed_user_region_forest, patterns, expected_keys, expected_children
):
    forest = mixed_user_region_forest

    selected, keys = select_operators(forest, patterns)

    assert keys == expected_keys
    selected_root = selected["1"][0]
    assert selected_root.kernels == {}
    assert [child.name for child in selected_root.children] == expected_children
    selected_region = selected_root.children[0]
    if "user" in patterns:
        source_region = forest["1"][0].children[0]
        assert selected_region == source_region
        selected_torch = selected_region.children[0]
        assert selected_torch.kernels["addmm_kernel"].pmc_group_keys == [
            ("torch",),
            ("torch",),
        ]
        assert selected_torch.kernels["addmm_kernel"].launches == 2
        assert selected_region.kernel_launches == 5
        assert selected_region.total_duration_ms == pytest.approx(0.00011)
    else:
        assert selected_region.kernels == {}
        assert [child.name for child in selected_region.children] == ["nested-region"]
        assert selected_region.children[0].kernels == {}
        assert selected_region.kernel_launches == 1


def test_select_operators_deduplicates_returned_keys_across_roots_in_order(
    simple_model_forest_with_relu_and_addmm,
):
    forest = simple_model_forest_with_relu_and_addmm()
    forest["later-thread"] = simple_model_forest_with_relu_and_addmm()["1"]

    selected, keys = select_operators(forest, {"torch": ["*addmm*", "*relu*"]})

    assert keys == [ADDMM_PMC_GROUP_KEY, RELU_PMC_GROUP_KEY]
    assert list(selected) == ["1", "later-thread"]


@pytest.mark.parametrize(
    ("keys", "expected_ids"),
    [
        pytest.param(
            [RELU_PMC_GROUP_KEY, ADDMM_PMC_GROUP_KEY, RELU_PMC_GROUP_KEY],
            [8, 3, 8],
            id="preserves-order-and-duplicates",
        ),
        pytest.param([], [], id="empty-keys"),
    ],
)
def test_analyze_dispatch_ids_from_pmc_keys_returns_mapped_ids(keys, expected_ids):
    slot_to_id = {RELU_PMC_GROUP_KEY: 8, ADDMM_PMC_GROUP_KEY: 3}

    assert analyze_dispatch_ids_from_pmc_keys(keys, slot_to_id) == expected_ids


def test_dispatch_filter_keeps_only_operator_dispatches():
    assert intersect_dispatch_ids([1, 2, 3], requested_dispatch_ids=[2, 9]) == [2]


@pytest.mark.parametrize(
    ("operator_ids", "requested_ids", "minimum_id", "expected_ids"),
    [
        pytest.param([9, 3, 9, 1], None, None, [1, 3, 9], id="sorted-unique-default"),
        pytest.param([9, 3, 9, 1], [], None, [], id="empty-explicit-selection"),
        pytest.param([9, 3, 9, 1], [9, 9, 3], None, [3, 9], id="requested-duplicates"),
        pytest.param([9, 3, 9, 1], [7], None, [], id="disjoint-request"),
        pytest.param([9, 3, 9, 1], None, 3, [9], id="exclusive-threshold"),
        pytest.param([9, 3, 9, 1], None, 9, [], id="highest-threshold"),
        pytest.param([], None, None, [], id="empty-operator-selection"),
    ],
)
def test_intersect_dispatch_ids_returns_sorted_unique_selection(
    operator_ids, requested_ids, minimum_id, expected_ids
):
    assert (
        intersect_dispatch_ids(
            operator_ids,
            requested_dispatch_ids=requested_ids,
            minimum_dispatch_id=minimum_id,
        )
        == expected_ids
    )


def test_intersect_dispatch_ids_by_kernel_uses_labels_and_last_raw_row():
    raw_pmc = pd.DataFrame({
        "Dispatch_ID": ["1", "2", "3", 3],
        "Kernel_Name": [" target ", "other", "other", "\ttarget\n"],
    })
    kernel_top = pd.DataFrame({"Kernel_Name": ["other", " target "]}, index=[11, 7])
    operator_ids = [3, 1, 3, 99, 2]
    assert intersect_dispatch_ids_by_kernel(operator_ids, raw_pmc, kernel_top, [7]) == [
        3,
        1,
        3,
    ]


def test_intersect_dispatch_ids_by_kernel_empty_selection_returns_no_dispatches():
    raw_pmc = pd.DataFrame({"Dispatch_ID": [1], "Kernel_Name": ["target"]})
    kernel_top = pd.DataFrame({"Kernel_Name": ["target"]})

    assert intersect_dispatch_ids_by_kernel([1], raw_pmc, kernel_top, []) == []


def test_select_operators_handles_nesting_deeper_than_the_recursion_limit():
    nesting = sys.getrecursionlimit() + 100
    node = CallTreeNode(
        name="leaf",
        backend="torch",
        kernels={"gemm": KernelStats(1, 10.0, 10.0, 10.0, [(1, "gemm")])},
    )
    for level in range(nesting - 1):
        node = CallTreeNode(name=f"op{level}", backend="torch", children=[node])

    subtree, pmc_group_keys = select_operators({"1": [node]}, {"torch": ["leaf"]})

    assert pmc_group_keys == [(1, "gemm")]
    assert len(list(iter_call_tree_paths(subtree["1"]))) == nesting
