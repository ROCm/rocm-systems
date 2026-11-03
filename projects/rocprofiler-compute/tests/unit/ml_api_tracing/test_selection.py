# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML operator selection and shared kernel numbering."""

from copy import deepcopy

import pandas as pd
import pytest

from ml_api_tracing.call_tree import iter_call_tree_paths
from ml_api_tracing.models import CallTreeNode, KernelStats
from ml_api_tracing.selection import (
    analyze_dispatch_ids_from_pmc_keys,
    assign_kernel_ids_from_top,
    intersect_dispatch_ids,
    intersect_dispatch_ids_by_kernel,
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


def build_numbering_node() -> CallTreeNode:
    grandchild = CallTreeNode(
        name="grandchild",
        kernels={
            "\tleaf_kernel\n": KernelStats(
                launches=2,
                total_duration_ns=4000.0,
                min_duration_ns=1000.0,
                max_duration_ns=3000.0,
                kernel_id=92,
                pmc_group_keys=[("guid", 2, 256, 64, 0)],
            ),
            "unmapped_leaf": KernelStats(kernel_id=93),
            "unmapped_without_id": KernelStats(),
        },
        invocation_ids={"grandchild:1"},
        args_invocations={"(self=float32[2x2])": {"grandchild:1"}},
        kernel_launches=2,
        total_duration_ms=0.004,
    )
    child = CallTreeNode(
        name="child",
        children=[grandchild],
        kernels={"shared_kernel": KernelStats(kernel_id=91)},
    )
    return CallTreeNode(
        name="root",
        children=[child],
        kernels={
            "  shared_kernel  ": KernelStats(kernel_id=90),
            "unmapped_root": KernelStats(kernel_id=94),
        },
    )


@pytest.mark.parametrize(
    ("name_to_id", "mapped_ids", "shared_id", "leaf_id"),
    [
        pytest.param(
            {"shared_kernel": 0, "leaf_kernel": 7, "absent_kernel": 100},
            {0, 7},
            0,
            7,
            id="whitespace-shared-zero-id-and-descendants",
        ),
        pytest.param(
            {"shared_kernel": 4},
            {4},
            4,
            92,
            id="partial-map-retains-unmapped-descendant-id",
        ),
        pytest.param({}, set(), 90, 92, id="empty-map-retains-all-ids"),
    ],
)
def test_assign_kernel_ids_from_top_changes_only_mapped_ids(
    name_to_id, mapped_ids, shared_id, leaf_id
) -> None:
    node = build_numbering_node()
    expected = deepcopy(node)
    root_stats = node.kernels["  shared_kernel  "]
    child_stats = node.children[0].kernels["shared_kernel"]
    leaf_stats = node.children[0].children[0].kernels["\tleaf_kernel\n"]
    mapping_before = dict(name_to_id)
    expected.kernels["  shared_kernel  "].kernel_id = shared_id
    expected.children[0].kernels["shared_kernel"].kernel_id = (
        shared_id if name_to_id else 91
    )
    expected.children[0].children[0].kernels["\tleaf_kernel\n"].kernel_id = leaf_id

    assert assign_kernel_ids_from_top(node, name_to_id) == mapped_ids
    assert node == expected
    assert node.kernels["  shared_kernel  "] is root_stats
    assert node.children[0].kernels["shared_kernel"] is child_stats
    assert node.children[0].children[0].kernels["\tleaf_kernel\n"] is leaf_stats
    assert name_to_id == mapping_before
    assert assign_kernel_ids_from_top(node, name_to_id) == mapped_ids
    assert node == expected


def test_assign_kernel_ids_from_top_empty_node_returns_no_ids() -> None:
    node = CallTreeNode(name="empty", children=[CallTreeNode(name="empty-child")])
    original = deepcopy(node)
    assert assign_kernel_ids_from_top(node, {"absent_kernel": 0}) == set()
    assert node == original


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
    previous = deepcopy(forest)

    subtree, pmc_group_keys = select_operators(forest, {"torch": ["*"]})

    assert forest_node_names(subtree) == ["nn.Module.Linear.forward"]
    assert pmc_group_keys == [(3, "torch_gemm_kernel", 0, 0, 0)]
    assert forest == previous


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
    previous = deepcopy(forest)

    assert select_operators(forest, patterns) == ({}, [])
    assert forest == previous


def test_select_operators_cpu_only_match_returns_copied_forest_without_keys():
    node = CallTreeNode(
        name="aten::cpu",
        backend="torch",
        invocation_ids={"10"},
        args_invocations={"(value=1)": {"10"}},
        file_name="model.py",
        line_number=17,
        thread_id="1",
    )
    forest = {"1": [node]}

    subtree, pmc_group_keys = select_operators(forest, {"torch": ["aten::cpu"]})

    assert subtree == forest
    assert subtree is not forest
    assert subtree["1"][0] is not node
    assert subtree["1"][0].invocation_ids is not node.invocation_ids
    assert subtree["1"][0].args_invocations is not node.args_invocations
    assert pmc_group_keys == []


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
    stats.kernel_id = 7
    stats.min_duration_ns = 50.0
    stats.max_duration_ns = 50.0
    stats.pmc_group_keys.append(ADDMM_PMC_GROUP_KEY)
    previous = deepcopy(forest)

    subtree, pmc_group_keys = select_operators(
        forest,
        {"torch": ["*Linear.forward", "*/aten::addmm", "*/aten::addmm"]},
    )

    assert pmc_group_keys == [ADDMM_PMC_GROUP_KEY, ADDMM_PMC_GROUP_KEY]
    assert forest == previous
    assert subtree["1"][0].kernels == {}
    copied_addmm = subtree["1"][0].children[0].children[0]
    assert copied_addmm == addmm
    assert copied_addmm is not addmm
    assert copied_addmm.kernels["addmm_kernel"] is not stats
    assert (
        copied_addmm.kernels["addmm_kernel"].pmc_group_keys is not stats.pmc_group_keys
    )
    copied_addmm.args_invocations["(self=float32[2x2])"].add("30")
    copied_addmm.kernels["addmm_kernel"].kernel_id = 99
    assert forest == previous


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
    previous = dict(slot_to_id)

    assert analyze_dispatch_ids_from_pmc_keys(keys, slot_to_id) == expected_ids
    assert slot_to_id == previous


def test_analyze_dispatch_ids_from_pmc_keys_raises_for_first_missing_key():
    first_missing = ("first-missing",)
    keys = [RELU_PMC_GROUP_KEY, first_missing, ("later-missing",)]
    slot_to_id = {RELU_PMC_GROUP_KEY: 8}

    with pytest.raises(KeyError) as failure:
        analyze_dispatch_ids_from_pmc_keys(keys, slot_to_id)

    assert failure.value.args == (first_missing,)
    assert slot_to_id == {RELU_PMC_GROUP_KEY: 8}


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
    previous_ids = list(operator_ids)

    assert (
        intersect_dispatch_ids(
            operator_ids,
            requested_dispatch_ids=requested_ids,
            minimum_dispatch_id=minimum_id,
        )
        == expected_ids
    )
    assert operator_ids == previous_ids


def test_intersect_dispatch_ids_by_kernel_uses_labels_and_last_raw_row():
    raw_pmc = pd.DataFrame({
        "Dispatch_ID": ["1", "2", "3", 3],
        "Kernel_Name": [" target ", "other", "other", "\ttarget\n"],
    })
    kernel_top = pd.DataFrame({"Kernel_Name": ["other", " target "]}, index=[11, 7])
    operator_ids = [3, 1, 3, 99, 2]
    previous_raw = raw_pmc.copy(deep=True)
    previous_top = kernel_top.copy(deep=True)

    assert intersect_dispatch_ids_by_kernel(operator_ids, raw_pmc, kernel_top, [7]) == [
        3,
        1,
        3,
    ]
    assert operator_ids == [3, 1, 3, 99, 2]
    pd.testing.assert_frame_equal(raw_pmc, previous_raw)
    pd.testing.assert_frame_equal(kernel_top, previous_top)


def test_intersect_dispatch_ids_by_kernel_empty_selection_returns_no_dispatches():
    raw_pmc = pd.DataFrame({"Dispatch_ID": [1], "Kernel_Name": ["target"]})
    kernel_top = pd.DataFrame({"Kernel_Name": ["target"]})

    assert intersect_dispatch_ids_by_kernel([1], raw_pmc, kernel_top, []) == []


def test_intersect_dispatch_ids_by_kernel_missing_label_raises():
    raw_pmc = pd.DataFrame({"Dispatch_ID": [1], "Kernel_Name": ["target"]})
    kernel_top = pd.DataFrame({"Kernel_Name": ["target"]}, index=[7])

    with pytest.raises(KeyError) as failure:
        intersect_dispatch_ids_by_kernel([1], raw_pmc, kernel_top, [0])

    assert failure.value.args == (0,)
