# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML operator selection and shared kernel numbering."""

from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ml_api_tracing.call_tree import (
    copy_matched_operator_subtree,
    filter_forest_by_backends,
    iter_call_tree_paths,
)
from ml_api_tracing.models import CallTreeForest, CallTreeNode
from utils.pattern_matching import fnmatch_glob_matches


def assign_kernel_ids_from_top(
    node: CallTreeNode, name_to_id: Dict[str, int]
) -> Set[int]:
    """Set KernelStats.kernel_id on this node and descendants; return those ids."""
    kernel_ids: Set[int] = set()
    for kernel_name, stats in node.kernels.items():
        kernel_id = name_to_id.get(str(kernel_name).strip())
        if kernel_id is not None:
            stats.kernel_id = kernel_id
            kernel_ids.add(kernel_id)
    for child in node.children:
        kernel_ids.update(assign_kernel_ids_from_top(child, name_to_id))
    return kernel_ids


def select_operators(
    forest: CallTreeForest, patterns_by_backend: Dict[str, List[str]]
) -> Tuple[CallTreeForest, List[Tuple]]:
    """Return an owned matched forest and ordered, possibly repeated PMC keys.

    No match returns an empty forest. A matched operator without dispatches
    retains its copied forest and has an empty key list.
    """
    confined_forest = filter_forest_by_backends(forest, list(patterns_by_backend))
    matched_nodes = _collect_glob_matched_nodes(confined_forest, patterns_by_backend)
    if not matched_nodes:
        return {}, []
    selected_forest = copy_matched_operator_subtree(confined_forest, matched_nodes)
    pmc_group_keys: List[Tuple] = []
    for roots in selected_forest.values():
        for root in roots:
            pmc_group_keys.extend(_pmc_group_keys_from_node(root))
    return selected_forest, pmc_group_keys


def analyze_dispatch_ids_from_pmc_keys(
    pmc_group_keys: List[Tuple], slot_to_id: Dict[Tuple, int]
) -> List[int]:
    """Map PMC keys in encounter order, raising at the first absent key."""
    analyze_ids: List[int] = []
    for group_key in pmc_group_keys:
        analyze_ids.append(slot_to_id[group_key])
    return analyze_ids


def intersect_dispatch_ids(
    operator_dispatch_ids: List[int],
    requested_dispatch_ids: Optional[List[int]] = None,
    minimum_dispatch_id: Optional[int] = None,
) -> List[int]:
    """Return sorted unique operator IDs under normalized user restrictions.

    minimum_dispatch_id is an exclusive lower bound. None means unrestricted;
    an explicit empty requested_dispatch_ids list selects no dispatches.
    """
    selected_ids = set(operator_dispatch_ids)
    if minimum_dispatch_id is not None:
        selected_ids = {
            dispatch_id
            for dispatch_id in selected_ids
            if dispatch_id > minimum_dispatch_id
        }
    elif requested_dispatch_ids is not None:
        selected_ids &= set(requested_dispatch_ids)
    return sorted(selected_ids)


def intersect_dispatch_ids_by_kernel(
    operator_dispatch_ids: List[int],
    raw_pmc: pd.DataFrame,
    kernel_top_df: pd.DataFrame,
    filter_kernel_ids: List[int],
) -> List[int]:
    """Keep dispatch order and repeats for kernels selected from the sliced top.

    Kernel IDs address the existing DataFrame labels. Duplicate raw dispatch
    rows retain the last kernel-name mapping, as in the analyze adapter.
    """
    selected_names = {
        str(kernel_top_df.loc[kernel_id, "Kernel_Name"]).strip()
        for kernel_id in filter_kernel_ids
    }
    dispatch_names = {
        int(dispatch_id): str(kernel_name).strip()
        for dispatch_id, kernel_name in zip(
            raw_pmc["Dispatch_ID"].tolist(), raw_pmc["Kernel_Name"].tolist()
        )
    }
    return [
        dispatch_id
        for dispatch_id in operator_dispatch_ids
        if dispatch_names.get(int(dispatch_id)) in selected_names
    ]


def _collect_glob_matched_nodes(
    forest: CallTreeForest,
    framewise_patterns: Dict[str, List[str]],
) -> List[CallTreeNode]:
    """Return nodes whose backend matches and whose path or name glob-matches."""
    return [
        node
        for roots in forest.values()
        for node, path in iter_call_tree_paths(roots)
        if any(
            fnmatch_glob_matches(pattern, "/".join(path))
            or fnmatch_glob_matches(pattern, node.name)
            for pattern in framewise_patterns.get(node.backend or "", [])
        )
    ]


def _pmc_group_keys_from_node(node: CallTreeNode) -> List[Tuple]:
    pmc_group_keys: List[Tuple] = []
    for stats in node.kernels.values():
        pmc_group_keys.extend(stats.pmc_group_keys)
    for child in node.children:
        pmc_group_keys.extend(_pmc_group_keys_from_node(child))
    return pmc_group_keys
