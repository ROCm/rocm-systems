# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API call-tree processing, inclusive statistics, and copied views."""

from collections.abc import Sequence
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS
from ml_api_tracing.errors import (
    KernelSequenceLengthMismatchError,
    LauncherThreadNotFoundError,
    MarkerNotNestedError,
    MissingSourceLocationError,
    MlApiTraceError,
    OverlappingMarkerRangeError,
    UncorrelatedLauncherIntervalError,
    handle_fatal_ml_api_trace_error,
)
from ml_api_tracing.marker_format import split_operator_args
from ml_api_tracing.models import CallTreeForest, CallTreeNode, KernelStats, NodeRollup
from utils.utils_common import NS_TO_MS


def rollup_node_stats(node: CallTreeNode) -> NodeRollup:
    """Bottom-up rollup over this node and all descendants.

    Sets inclusive fields on node: kernel_launches, total_duration_ms,
    min_dispatch_ns, max_dispatch_ns, mean_dispatch_ns.

    Subtrees with no positive-duration dispatch leave min/max as None.
    Mean is None only when there are no launches; zero-duration launches
    contribute to the mean and can produce 0.
    """
    launches = 0
    total_duration_ns = 0.0
    mins: List[float] = []
    maxes: List[float] = []

    for stats in node.kernels.values():
        launches += stats.launches
        total_duration_ns += stats.total_duration_ns
        if stats.min_duration_ns is not None:
            mins.append(stats.min_duration_ns)
        if stats.max_duration_ns is not None:
            maxes.append(stats.max_duration_ns)

    for child in node.children:
        child_rollup = rollup_node_stats(child)
        launches += child_rollup.launches
        total_duration_ns += child_rollup.total_duration_ns
        if child_rollup.min_dispatch_ns is not None:
            mins.append(child_rollup.min_dispatch_ns)
        if child_rollup.max_dispatch_ns is not None:
            maxes.append(child_rollup.max_dispatch_ns)

    node.kernel_launches = launches
    node.total_duration_ms = total_duration_ns * NS_TO_MS
    node.min_dispatch_ns = min(mins, default=None)
    node.max_dispatch_ns = max(maxes, default=None)
    node.mean_dispatch_ns = total_duration_ns / launches if launches > 0 else None

    return NodeRollup(
        launches=launches,
        total_duration_ns=total_duration_ns,
        min_dispatch_ns=node.min_dispatch_ns,
        max_dispatch_ns=node.max_dispatch_ns,
    )


def iter_call_tree_paths(
    nodes: List[CallTreeNode], parent_path: Tuple[str, ...] = ()
) -> Iterator[Tuple[CallTreeNode, Tuple[str, ...]]]:
    """Yield each node depth-first with its operator path from the root."""
    for node in nodes:
        path = parent_path + (node.name,)
        yield node, path
        yield from iter_call_tree_paths(node.children, path)


def nest_marker_intervals(
    trace_df: pd.DataFrame,
    errors: Optional[List[MlApiTraceError]] = None,
    skipped_keys: Optional[Set[Tuple[str, str]]] = None,
) -> CallTreeForest:
    """Nest marker intervals per Thread_Id using timestamp containment.

    Rows are sorted by thread, start time, and descending end time so
    parents are visited before children. The input frame is not mutated.
    Overlapping rows are skipped and recorded in skipped_keys so later nest
    validation does not report them again.
    """
    forest: CallTreeForest = {}
    if trace_df.empty:
        return forest
    ordered = trace_df.sort_values(
        by=["Thread_Id", "Start_Timestamp", "End_Timestamp"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    for thread_id, group in ordered.groupby("Thread_Id", sort=False):
        roots: List[CallTreeNode] = []
        open_ranges: List[Tuple[CallTreeNode, float, float]] = []
        thread_key = str(thread_id)
        for row in group.itertuples(index=False):
            start = float(row.Start_Timestamp)
            end = float(row.End_Timestamp)
            while open_ranges and open_ranges[-1][2] <= start:
                open_ranges.pop()
            if open_ranges and end > open_ranges[-1][2]:
                _record_overlapping_marker(
                    errors,
                    skipped_keys,
                    thread_key,
                    open_ranges[-1],
                    row,
                    start,
                    end,
                )
                continue
            node = _call_tree_node_from_marker_row(row, errors)
            if open_ranges:
                open_ranges[-1][0].children.append(node)
            else:
                roots.append(node)
            open_ranges.append((node, start, end))
        forest[thread_key] = roots
    return forest


def attach_unlocated_trees_by_launcher_thread(
    forest: CallTreeForest,
    errors: List[MlApiTraceError],
) -> None:
    """Stack torch/triton trees with no source onto the autograd launcher thread.

    Roots that cannot be placed stay in the forest. Placement issues are
    appended to errors in encounter order; this function does not log.
    """
    pending: List[Tuple[str, CallTreeNode]] = []
    for thread_id, roots in forest.items():
        for root in roots:
            if root.file_name is not None:
                continue
            if root.backend not in KNOWN_ML_API_BACKENDS:
                continue
            pending.append((thread_id, root))
    for thread_id, root in pending:
        start = float(root.start_timestamp or 0.0)
        end = float(root.end_timestamp or 0.0)
        launcher_thread_id = _launcher_thread_id_from_tree(root)
        if launcher_thread_id is None:
            continue
        launcher_key = str(launcher_thread_id)
        if launcher_key not in forest:
            errors.append(
                LauncherThreadNotFoundError(
                    operator_name=root.name,
                    thread_id=thread_id,
                    start_timestamp=start,
                    launcher_thread_id=launcher_key,
                )
            )
            continue
        dest_roots = [node for node in forest[launcher_key] if node is not root]
        parent = _deepest_containing_node(dest_roots, start, end)
        if parent is None:
            errors.append(
                UncorrelatedLauncherIntervalError(
                    operator_name=root.name,
                    thread_id=thread_id,
                    start_timestamp=start,
                    end_timestamp=end,
                    launcher_thread_id=launcher_key,
                )
            )
            continue
        parent.children.append(root)
        forest[thread_id] = [node for node in forest[thread_id] if node is not root]
    for thread_id in [tid for tid, roots in forest.items() if not roots]:
        del forest[thread_id]


def prune_cpu_only_call_trees(forest: CallTreeForest) -> None:
    """Drop subtrees with no kernel dispatches."""
    for thread_id, roots in list(forest.items()):
        kept_roots = [root for root in roots if _prune_cpu_only_call_tree_node(root)]
        if kept_roots:
            forest[thread_id] = kept_roots
            continue
        del forest[thread_id]


def record_missing_source_location_errors(
    forest: CallTreeForest,
    errors: List[MlApiTraceError],
) -> None:
    """Append missing-source issues for backend roots without a launcher thread."""
    for thread_id, roots in forest.items():
        for root in roots:
            if root.file_name is not None:
                continue
            if root.backend not in KNOWN_ML_API_BACKENDS:
                continue
            if root.launcher_thread_id is not None:
                continue
            errors.append(
                MissingSourceLocationError(
                    operator_name=root.name,
                    thread_id=thread_id,
                    start_timestamp=float(root.start_timestamp or 0.0),
                )
            )


def validate_all_markers_nested(
    trace_df: pd.DataFrame,
    forest: CallTreeForest,
    errors: Optional[List[MlApiTraceError]] = None,
    skipped_keys: Optional[Set[Tuple[str, str]]] = None,
) -> None:
    """Record consolidated marker rows missing from the nested forest."""
    if trace_df.empty:
        return
    nested_keys = _nested_invocation_keys(forest)
    ignored_keys = skipped_keys or set()
    for row in trace_df.itertuples(index=False):
        thread_id = str(row.Thread_Id)
        start_key = str(row.Start_Timestamp)
        if (thread_id, start_key) in ignored_keys:
            continue
        if (thread_id, start_key) not in nested_keys:
            handle_fatal_ml_api_trace_error(
                MarkerNotNestedError(
                    operator_name=str(row.Operator_Name),
                    thread_id=thread_id,
                    start_timestamp=float(row.Start_Timestamp),
                ),
                errors,
            )


def clone_call_tree_node(node: CallTreeNode) -> CallTreeNode:
    """Copy a call-tree node's payload; children start empty.

    Kernel stats, invocation ids, and argument variants are new objects.
    Callers attach already-copied children after the clone.
    """
    copied = CallTreeNode(
        name=node.name,
        kernels={
            kernel_name: KernelStats(
                launches=stats.launches,
                total_duration_ns=stats.total_duration_ns,
                min_duration_ns=stats.min_duration_ns,
                max_duration_ns=stats.max_duration_ns,
                kernel_id=stats.kernel_id,
                pmc_group_keys=list(stats.pmc_group_keys),
            )
            for kernel_name, stats in node.kernels.items()
        },
        kernel_launches=node.kernel_launches,
        total_duration_ms=node.total_duration_ms,
        min_dispatch_ns=node.min_dispatch_ns,
        max_dispatch_ns=node.max_dispatch_ns,
        mean_dispatch_ns=node.mean_dispatch_ns,
        file_name=node.file_name,
        line_number=node.line_number,
        backend=node.backend,
        start_timestamp=node.start_timestamp,
        end_timestamp=node.end_timestamp,
        thread_id=node.thread_id,
        launcher_thread_id=node.launcher_thread_id,
    )
    copied.invocation_ids = set(node.invocation_ids)
    copied.args_invocations = {
        args_blob: set(recorded_invocation_ids)
        for args_blob, recorded_invocation_ids in node.args_invocations.items()
    }
    return copied


def clone_call_tree_forest(forest: CallTreeForest) -> CallTreeForest:
    """Copy every node and mutable payload without recursive descent.

    Retains thread, root, and child order and existing inclusive statistics.
    Callers can mutate the copy without changing the supplied forest.
    """
    copied_forest: CallTreeForest = {}
    for thread_id, roots in forest.items():
        copied_roots: List[CallTreeNode] = []
        copied_forest[thread_id] = copied_roots
        pending_nodes: List[Tuple[CallTreeNode, List[CallTreeNode]]] = [
            (node, copied_roots) for node in reversed(roots)
        ]
        while pending_nodes:
            node, destination = pending_nodes.pop()
            copied_node = clone_call_tree_node(node)
            destination.append(copied_node)
            pending_nodes.extend(
                (child, copied_node.children) for child in reversed(node.children)
            )
    return copied_forest


def fold_identical_sibling_subtrees(
    nodes: List[CallTreeNode],
) -> List[CallTreeNode]:
    """Merge siblings that share operator, kernel names, and child shape.

    Returns new nodes sorted by total GPU duration. Does not mutate nodes.
    """
    if not nodes:
        return []
    groups: Dict[tuple, List[CallTreeNode]] = {}
    for node in nodes:
        groups.setdefault(_subtree_shape_key(node), []).append(node)
    folded = [_merge_identical_sibling_group(group) for group in groups.values()]
    folded.sort(key=lambda node: node.total_duration_ms, reverse=True)
    return folded


def filter_forest_by_backends(
    forest: CallTreeForest,
    backends: List[str],
) -> CallTreeForest:
    """Copy a forest, keeping nodes for the requested backends and their ancestors.

    An empty backends list returns the original forest, including user.
    Ancestors kept only as scaffolding have their own kernels cleared, so a torch parent
    of a triton child does not contribute GPU time or kernel lines to a
    triton listing. Inclusive stats on each copy are recomputed from the
    kernels and children that remain.
    """
    if not backends:
        return forest

    filtered: CallTreeForest = {}
    for thread_id, roots in forest.items():
        kept_roots: List[CallTreeNode] = []
        for root in roots:
            copied_root = _copy_backend_view(root, backends)
            if copied_root is not None:
                kept_roots.append(copied_root)
        if kept_roots:
            filtered[thread_id] = kept_roots
    return filtered


def copy_matched_operator_subtree(
    forest: CallTreeForest,
    matched_nodes: List[CallTreeNode],
) -> CallTreeForest:
    """Copy matches, in-scope descendants, and ancestors needed to reach them."""
    matched_node_ids = {id(node) for node in matched_nodes}
    subtree: CallTreeForest = {}
    for thread_id, roots in forest.items():
        kept_roots: List[CallTreeNode] = []
        for root in roots:
            copied_root = _copy_scoped_operator_node(root, matched_node_ids, set())
            if copied_root is not None:
                kept_roots.append(copied_root)
        if kept_roots:
            subtree[thread_id] = kept_roots
    return subtree


def _optional_marker_file_name(value: object) -> Optional[str]:
    if value is None or value == "" or pd.isna(value):
        return None
    return str(value)


def _optional_marker_line_number(value: object) -> Optional[int]:
    if value is None or value == "" or pd.isna(value):
        return None
    return int(value)


def _optional_launcher_thread_id(value: object) -> Optional[str]:
    if value is None or value == "" or pd.isna(value):
        return None
    text = str(value)
    if text == "n/a":
        return None
    return text


def _optional_operator_args_blob(value: object) -> str:
    """Return the operator-args blob, or empty when none were recorded."""
    if value is None or value == "" or pd.isna(value):
        return ""
    args_text = str(value).strip()
    if args_text in ("n/a", "()"):
        return ""
    if not split_operator_args(args_text):
        return ""
    return args_text


def _record_operator_args(
    node: CallTreeNode, args_value: object, invocation_id: str
) -> None:
    """Record the operator-args blob on node for this invocation."""
    args_blob = _optional_operator_args_blob(args_value)
    if not args_blob:
        return
    node.args_invocations.setdefault(args_blob, set()).add(invocation_id)


def _sequence_from_cell(value: object) -> List[object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _kernel_stats_from_marker_row(
    row: object,
    errors: Optional[List[MlApiTraceError]] = None,
) -> Dict[str, KernelStats]:
    kernel_names = _sequence_from_cell(getattr(row, "Kernel_Names", []))
    kernel_starts = _sequence_from_cell(getattr(row, "Kernel_Start_Timestamps", []))
    kernel_ends = _sequence_from_cell(getattr(row, "Kernel_End_Timestamps", []))
    pmc_group_keys = _sequence_from_cell(getattr(row, "Kernel_Pmc_Group_Keys", []))
    if not (
        len(kernel_names)
        == len(kernel_starts)
        == len(kernel_ends)
        == len(pmc_group_keys)
    ):
        handle_fatal_ml_api_trace_error(
            KernelSequenceLengthMismatchError(
                operator_name=str(getattr(row, "Operator_Name", "")),
                name_count=len(kernel_names),
                start_count=len(kernel_starts),
                end_count=len(kernel_ends),
            ),
            errors,
        )
        return {}
    kernels: Dict[str, KernelStats] = {}
    for kernel_name, kernel_start, kernel_end, pmc_group_key in zip(
        kernel_names, kernel_starts, kernel_ends, pmc_group_keys
    ):
        duration_ns = 0.0
        try:
            duration_ns = float(kernel_end) - float(kernel_start)
        except (TypeError, ValueError):
            duration_ns = 0.0
        name = str(kernel_name)
        if name not in kernels:
            kernels[name] = KernelStats()
        stats = kernels[name]
        stats.launches += 1
        stats.total_duration_ns += duration_ns
        stats.pmc_group_keys.append(tuple(pmc_group_key))
        if duration_ns > 0:
            if stats.min_duration_ns is None or duration_ns < stats.min_duration_ns:
                stats.min_duration_ns = duration_ns
            if stats.max_duration_ns is None or duration_ns > stats.max_duration_ns:
                stats.max_duration_ns = duration_ns
    return kernels


def _call_tree_node_from_marker_row(
    row: object,
    errors: Optional[List[MlApiTraceError]] = None,
) -> CallTreeNode:
    backend_value = getattr(row, "Backend", None)
    if backend_value is None or (
        isinstance(backend_value, float) and pd.isna(backend_value)
    ):
        backend = None
    else:
        backend = str(backend_value)
    node = CallTreeNode(
        name=str(row.Operator_Name),
        kernels=_kernel_stats_from_marker_row(row, errors),
        file_name=_optional_marker_file_name(getattr(row, "File_Name", "")),
        line_number=_optional_marker_line_number(getattr(row, "Line_Number", "")),
        backend=backend,
        start_timestamp=float(row.Start_Timestamp),
        end_timestamp=float(row.End_Timestamp),
        thread_id=str(getattr(row, "Thread_Id", "")),
        launcher_thread_id=_optional_launcher_thread_id(
            getattr(row, "launcher_thread_id", "")
        ),
    )
    invocation_id = str(row.Start_Timestamp)
    node.invocation_ids.add(invocation_id)
    _record_operator_args(node, getattr(row, "args", ""), invocation_id)
    return node


def _record_overlapping_marker(
    errors: Optional[List[MlApiTraceError]],
    skipped_keys: Optional[Set[Tuple[str, str]]],
    thread_key: str,
    parent: Tuple[CallTreeNode, float, float],
    row: object,
    start: float,
    end: float,
) -> None:
    """Record an overlapping range and add its skip key when collecting skips."""
    parent_node, parent_start, parent_end = parent
    handle_fatal_ml_api_trace_error(
        OverlappingMarkerRangeError(
            thread_id=thread_key,
            first_name=parent_node.name,
            first_start=parent_start,
            first_end=parent_end,
            second_name=str(row.Operator_Name),
            second_start=start,
            second_end=end,
        ),
        errors,
    )
    if skipped_keys is not None:
        skipped_keys.add((thread_key, str(row.Start_Timestamp)))


def _launcher_thread_id_from_tree(node: CallTreeNode) -> Optional[str]:
    if node.launcher_thread_id is not None:
        return node.launcher_thread_id
    for child in node.children:
        found = _launcher_thread_id_from_tree(child)
        if found is not None:
            return found
    return None


def _interval_contains(node: CallTreeNode, start: float, end: float) -> bool:
    if node.start_timestamp is None or node.end_timestamp is None:
        return False
    return node.start_timestamp <= start and end <= node.end_timestamp


def _deepest_containing_node(
    nodes: List[CallTreeNode], start: float, end: float
) -> Optional[CallTreeNode]:
    for node in nodes:
        if not _interval_contains(node, start, end):
            continue
        nested = _deepest_containing_node(node.children, start, end)
        return nested if nested is not None else node
    return None


def _prune_cpu_only_call_tree_node(node: CallTreeNode) -> bool:
    """Drop CPU-only children. True if this node has kernel work."""
    node.children = [
        child for child in node.children if _prune_cpu_only_call_tree_node(child)
    ]
    return bool(node.kernels) or bool(node.children)


def _nested_invocation_keys(
    forest: CallTreeForest,
) -> Set[Tuple[str, str]]:
    """Return (Thread_Id, marker-start) pairs present in the forest."""
    return {
        (node.thread_id or "", invocation_id)
        for roots in forest.values()
        for node, _path in iter_call_tree_paths(roots)
        for invocation_id in node.invocation_ids
    }


def _subtree_shape_key(node: CallTreeNode) -> tuple:
    line_number = "" if node.line_number is None else str(node.line_number)
    child_keys = tuple(sorted(_subtree_shape_key(child) for child in node.children))
    kernel_names = tuple(sorted(node.kernels))
    return (
        node.name,
        node.file_name or "",
        line_number,
        node.backend or "",
        kernel_names,
        child_keys,
    )


def _add_kernel_stats(
    destination: Dict[str, KernelStats], name: str, stats: KernelStats
) -> None:
    existing = destination.get(name)
    if existing is None:
        destination[name] = KernelStats(
            launches=stats.launches,
            total_duration_ns=stats.total_duration_ns,
            min_duration_ns=stats.min_duration_ns,
            max_duration_ns=stats.max_duration_ns,
            kernel_id=stats.kernel_id,
            pmc_group_keys=list(stats.pmc_group_keys),
        )
        return
    minimum_durations = [
        duration
        for duration in (existing.min_duration_ns, stats.min_duration_ns)
        if duration is not None
    ]
    maximum_durations = [
        duration
        for duration in (existing.max_duration_ns, stats.max_duration_ns)
        if duration is not None
    ]
    existing.launches += stats.launches
    existing.total_duration_ns += stats.total_duration_ns
    existing.min_duration_ns = min(minimum_durations) if minimum_durations else None
    existing.max_duration_ns = max(maximum_durations) if maximum_durations else None
    if existing.kernel_id is None:
        existing.kernel_id = stats.kernel_id
    existing.pmc_group_keys.extend(stats.pmc_group_keys)


def _merge_args_invocations(
    destination: Dict[str, Set[str]], source: Dict[str, Set[str]]
) -> None:
    for args_blob, recorded_invocation_ids in source.items():
        destination.setdefault(args_blob, set()).update(recorded_invocation_ids)


def _merge_identical_sibling_group(group: List[CallTreeNode]) -> CallTreeNode:
    first = group[0]
    kernels: Dict[str, KernelStats] = {}
    for node in group:
        for kernel_name, stats in node.kernels.items():
            _add_kernel_stats(kernels, kernel_name, stats)
    merged = CallTreeNode(
        name=first.name,
        children=fold_identical_sibling_subtrees([
            child for node in group for child in node.children
        ]),
        kernels=kernels,
        file_name=first.file_name,
        line_number=first.line_number,
        backend=first.backend,
        thread_id=first.thread_id,
        launcher_thread_id=first.launcher_thread_id,
    )
    for node in group:
        merged.invocation_ids.update(node.invocation_ids)
        _merge_args_invocations(merged.args_invocations, node.args_invocations)
    rollup_node_stats(merged)
    return merged


def _copy_backend_view(
    node: CallTreeNode, backends: List[str]
) -> Optional[CallTreeNode]:
    """Copy one node for filter_forest_by_backends, or None when nothing is kept."""
    kept_children: List[CallTreeNode] = []
    for child in node.children:
        copied_child = _copy_backend_view(child, backends)
        if copied_child is not None:
            kept_children.append(copied_child)
    if node.backend not in backends and not kept_children:
        return None
    copied_node = clone_call_tree_node(node)
    copied_node.children = kept_children
    if node.backend not in backends:
        copied_node.kernels = {}
    rollup_node_stats(copied_node)
    return copied_node


def _copy_scoped_operator_node(
    node: CallTreeNode,
    matched_node_ids: Set[int],
    inherited_backends: Set[str],
) -> Optional[CallTreeNode]:
    """Copy one node, keeping match kernels and in-scope descendant kernels."""
    node_is_match = id(node) in matched_node_ids
    descendant_backends = inherited_backends
    if node_is_match and node.backend is not None:
        descendant_backends = set(inherited_backends)
        descendant_backends.add(node.backend)
    kept_children: List[CallTreeNode] = []
    for child in node.children:
        copied_child = _copy_scoped_operator_node(
            child,
            matched_node_ids,
            descendant_backends,
        )
        if copied_child is not None:
            kept_children.append(copied_child)
    backend_in_scope = node.backend is not None and node.backend in inherited_backends
    keep_kernels = node_is_match or backend_in_scope
    if not keep_kernels and not kept_children:
        return None
    copied_node = clone_call_tree_node(node)
    copied_node.children = kept_children
    if not keep_kernels:
        copied_node.kernels = {}
    rollup_node_stats(copied_node)
    return copied_node
