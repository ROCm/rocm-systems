# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API call-tree processing, inclusive statistics, and copied views."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ml_api_tracing.analyze.definitions import (
    CallTreeForest,
    CallTreeNode,
    KernelStats,
    NodeRollup,
)
from ml_api_tracing.analyze.errors import (
    KernelSequenceLengthMismatchError,
    LauncherThreadNotFoundError,
    MarkerNotNestedError,
    MissingSourceLocationError,
    MlApiTraceError,
    OverlappingMarkerRangeError,
    UncorrelatedLauncherIntervalError,
)
from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS, USER_REGION_BACKEND
from ml_api_tracing.marker_format import split_operator_args
from utils.utils_common import NS_TO_MS


def _iter_post_order(root: CallTreeNode) -> Iterator[CallTreeNode]:
    """Yield every node under root, children before their parent.

    Uses an explicit stack, so deep trees do not hit the recursion limit.
    A parent's child list is read before the parent is yielded, so the caller
    may replace it while iterating. Each node must appear once in the tree.
    """
    stack: List[Tuple[CallTreeNode, bool]] = [(root, False)]
    while stack:
        node, children_done = stack.pop()
        if children_done:
            yield node
            continue
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(node.children))


def rollup_node_stats(node: CallTreeNode) -> NodeRollup:
    """Bottom-up rollup over this node and all descendants.

    Sets inclusive fields on node: kernel_launches, total_duration_ms,
    min_dispatch_ns, max_dispatch_ns, mean_dispatch_ns.

    Subtrees with no positive-duration dispatch leave min/max as None.
    Mean is None only when there are no launches; zero-duration launches
    contribute to the mean and can produce 0.
    """
    rollups: Dict[int, NodeRollup] = {}
    for current in _iter_post_order(node):
        # pop frees each child's rollup once its parent has used it.
        rollups[id(current)] = _roll_up_from_child_rollups(
            current, [rollups.pop(id(child)) for child in current.children]
        )
    return rollups[id(node)]


def _roll_up_from_child_rollups(
    node: CallTreeNode, child_rollups: Sequence[NodeRollup]
) -> NodeRollup:
    """Set node's inclusive fields from its kernels and its children's rollups.

    The children's rollups are already complete, so the node does not walk its
    descendants again. Kernels are added first, then each child in order.
    """
    launches = 0
    total_duration_ns = 0.0
    minimum_durations: List[float] = []
    maximum_durations: List[float] = []

    for stats in node.kernels.values():
        launches += stats.launches
        total_duration_ns += stats.total_duration_ns
        if stats.min_duration_ns is not None:
            minimum_durations.append(stats.min_duration_ns)
        if stats.max_duration_ns is not None:
            maximum_durations.append(stats.max_duration_ns)

    for child_rollup in child_rollups:
        launches += child_rollup.launches
        total_duration_ns += child_rollup.total_duration_ns
        if child_rollup.min_dispatch_ns is not None:
            minimum_durations.append(child_rollup.min_dispatch_ns)
        if child_rollup.max_dispatch_ns is not None:
            maximum_durations.append(child_rollup.max_dispatch_ns)

    node.kernel_launches = launches
    node.total_duration_ms = total_duration_ns * NS_TO_MS
    node.min_dispatch_ns = min(minimum_durations, default=None)
    node.max_dispatch_ns = max(maximum_durations, default=None)
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
    stack = [(node, parent_path) for node in reversed(nodes)]
    while stack:
        node, node_parent_path = stack.pop()
        path = node_parent_path + (node.name,)
        yield node, path
        stack.extend((child, path) for child in reversed(node.children))


def nest_marker_intervals(
    trace_df: pd.DataFrame,
) -> CallTreeForest:
    """Nest marker intervals per Thread_Id using timestamp containment.

    Rows are sorted by thread, start time, and descending end time so
    parents are visited before children. Overlapping rows raise
    OverlappingMarkerRangeError.
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
                raise _overlapping_marker_error(
                    thread_key, open_ranges[-1], row, start, end
                )
            node = _call_tree_node_from_marker_row(row)
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
        launcher_roots = [node for node in forest[launcher_key] if node is not root]
        parent = _deepest_containing_node(launcher_roots, start, end)
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
) -> None:
    """Raise MarkerNotNestedError for a marker row missing from the forest."""
    if trace_df.empty:
        return
    nested_keys = _nested_invocation_keys(forest)
    for row in trace_df.itertuples(index=False):
        thread_id = str(row.Thread_Id)
        start_key = str(row.Start_Timestamp)
        if (thread_id, start_key) not in nested_keys:
            raise MarkerNotNestedError(
                operator_name=str(row.Operator_Name),
                thread_id=thread_id,
                start_timestamp=float(row.Start_Timestamp),
            )


def fold_identical_sibling_subtrees(
    nodes: List[CallTreeNode],
) -> List[CallTreeNode]:
    """Merge siblings with matching operator, source location, backend,
    kernel names, and child shape.

    Returns new nodes sorted by total GPU duration. Does not mutate nodes.
    """
    return [node for node, _rollup in _fold_siblings(nodes, _ShapeKeys())]


def copy_matched_operator_subtree(
    forest: CallTreeForest,
    matched_nodes: List[CallTreeNode],
) -> CallTreeForest:
    """Copy matching operators with the ancestors needed to reach them.

    A framework match keeps its framework's descendant kernels. An independently
    matched backend adds its own descendants. Ancestors keep their metadata and
    child paths, but not their own kernels. A matched user region keeps every
    descendant, including Torch, Triton and nested user regions.
    """
    matched_node_ids = {id(node) for node in matched_nodes}
    subtree: CallTreeForest = {}
    for thread_id, roots in forest.items():
        kept_roots: List[CallTreeNode] = []
        for root in roots:
            scoped_root = _copy_scoped_operator_node(root, matched_node_ids, set())
            if scoped_root is not None:
                kept_roots.append(scoped_root[0])
        if kept_roots:
            subtree[thread_id] = kept_roots
    return subtree


def _optional_marker_file_name(value: object) -> Optional[str]:
    """Return the source file name, using None for missing marker values."""
    if value is None or value == "" or pd.isna(value):
        return None
    return str(value)


def _optional_marker_line_number(value: object) -> Optional[int]:
    """Return the source line number, using None for missing marker values."""
    if value is None or value == "" or pd.isna(value):
        return None
    return int(value)


def _optional_launcher_thread_id(value: object) -> Optional[str]:
    """Return a recorded launcher thread ID, treating 'n/a' as missing."""
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
    """Normalize a dispatch-list cell while keeping strings as single values."""
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


def _kernel_stats_from_marker_row(row: object) -> Dict[str, KernelStats]:
    """Aggregate a marker's dispatches by kernel name in their recorded order.

    The parallel name, timestamp and PMC key lists must have equal lengths.
    Timestamp conversion failures use zero duration. Numeric differences remain
    in the total even when negative or nonfinite. Every dispatch counts as a
    launch; only positive durations contribute to minimum and maximum values.
    """
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
        raise KernelSequenceLengthMismatchError(
            operator_name=str(getattr(row, "Operator_Name", "")),
            name_count=len(kernel_names),
            start_count=len(kernel_starts),
            end_count=len(kernel_ends),
        )
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


def _call_tree_node_from_marker_row(row: object) -> CallTreeNode:
    """Build one call node with its dispatches and available marker metadata."""
    backend_value = getattr(row, "Backend", None)
    if backend_value is None or (
        isinstance(backend_value, float) and pd.isna(backend_value)
    ):
        backend = None
    else:
        backend = str(backend_value)
    node = CallTreeNode(
        name=str(row.Operator_Name),
        kernels=_kernel_stats_from_marker_row(row),
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


def _overlapping_marker_error(
    thread_key: str,
    parent: Tuple[CallTreeNode, float, float],
    row: object,
    start: float,
    end: float,
) -> OverlappingMarkerRangeError:
    """Describe a marker that overlaps its open parent without being contained."""
    parent_node, parent_start, parent_end = parent
    return OverlappingMarkerRangeError(
        thread_id=thread_key,
        first_name=parent_node.name,
        first_start=parent_start,
        first_end=parent_end,
        second_name=str(row.Operator_Name),
        second_start=start,
        second_end=end,
    )


def _launcher_thread_id_from_tree(node: CallTreeNode) -> Optional[str]:
    """Find the first launcher thread ID on this node or its descendants."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.launcher_thread_id is not None:
            return current.launcher_thread_id
        stack.extend(reversed(current.children))
    return None


def _interval_contains(node: CallTreeNode, start: float, end: float) -> bool:
    """Return whether a complete marker interval contains the supplied interval."""
    if node.start_timestamp is None or node.end_timestamp is None:
        return False
    return node.start_timestamp <= start and end <= node.end_timestamp


def _deepest_containing_node(
    nodes: List[CallTreeNode], start: float, end: float
) -> Optional[CallTreeNode]:
    """Find the innermost marker that contains the worker's start and end times."""
    deepest: Optional[CallTreeNode] = None
    candidates = nodes
    while True:
        container = next(
            (node for node in candidates if _interval_contains(node, start, end)),
            None,
        )
        if container is None:
            return deepest
        deepest = container
        candidates = container.children


def _prune_cpu_only_call_tree_node(node: CallTreeNode) -> bool:
    """Drop CPU-only children. True if this node has kernel work."""
    has_kernel_work: Dict[int, bool] = {}
    for current in _iter_post_order(node):
        current.children = [
            child for child in current.children if has_kernel_work[id(child)]
        ]
        has_kernel_work[id(current)] = bool(current.kernels) or bool(current.children)
    return has_kernel_work[id(node)]


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


class _ShapeKeys:
    """Subtree shape keys, cached by node id.

    Two subtrees have the same key when their operator, location, backend,
    kernel names and children match, in any order. Each distinct shape gets a
    small integer, so comparing keys never walks a deep tree. A node is keyed
    once even when its ancestors are folded at several levels. key() does not
    reuse _iter_post_order because it skips subtrees that are already keyed.
    """

    def __init__(self) -> None:
        self._by_node: Dict[int, int] = {}
        self._by_shape: Dict[tuple, int] = {}

    def key(self, root: CallTreeNode) -> int:
        """Return the shape key of root, keying its unkeyed descendants first."""
        stack: List[Tuple[CallTreeNode, bool]] = [(root, False)]
        while stack:
            node, children_done = stack.pop()
            if id(node) in self._by_node:
                continue
            if not children_done:
                stack.append((node, True))
                stack.extend((child, False) for child in node.children)
                continue
            line_number = "" if node.line_number is None else str(node.line_number)
            shape = (
                node.name,
                node.file_name or "",
                line_number,
                node.backend or "",
                tuple(sorted(node.kernels)),
                tuple(sorted(self._by_node[id(child)] for child in node.children)),
            )
            self._by_node[id(node)] = self._by_shape.setdefault(
                shape, len(self._by_shape)
            )
        return self._by_node[id(root)]


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
    existing.pmc_group_keys.extend(stats.pmc_group_keys)


def _merge_args_invocations(
    destination: Dict[str, Set[str]], source: Dict[str, Set[str]]
) -> None:
    for args_blob, recorded_invocation_ids in source.items():
        destination.setdefault(args_blob, set()).update(recorded_invocation_ids)


class _FoldLevel:
    """One sibling list being folded, and the folded nodes it produces."""

    def __init__(self, groups: List[List[CallTreeNode]]) -> None:
        self.groups = groups
        self.merged: List[CallTreeNode] = []
        self.child_levels: List[_FoldLevel] = []
        self.folded: List[Tuple[CallTreeNode, NodeRollup]] = []


def _fold_siblings(
    nodes: List[CallTreeNode], shape_keys: _ShapeKeys
) -> List[Tuple[CallTreeNode, NodeRollup]]:
    """Fold identical siblings; return each new node with its completed rollup.

    Merged nodes are created top-down. Their rollups and sort order are set
    bottom-up, so the folded children of each node are ordered by their
    completed totals.
    """
    root_level = _FoldLevel(_group_by_shape(nodes, shape_keys))
    levels = [root_level]
    # Top-down: the list grows while it is walked, because each level adds its
    # child levels to the end.
    for level in levels:
        for group in level.groups:
            level.merged.append(_merge_group_node(group))
            level.child_levels.append(
                _FoldLevel(
                    _group_by_shape(
                        [child for node in group for child in node.children],
                        shape_keys,
                    )
                )
            )
        levels.extend(level.child_levels)
    # Bottom-up: a child level always comes after its parent in the list, so
    # popping from the end finishes the children first.
    while levels:
        level = levels.pop()
        folded: List[Tuple[CallTreeNode, NodeRollup]] = []
        for merged, child_level in zip(level.merged, level.child_levels):
            merged.children = [child for child, _rollup in child_level.folded]
            rollup = _roll_up_from_child_rollups(
                merged, [child_rollup for _child, child_rollup in child_level.folded]
            )
            folded.append((merged, rollup))
        folded.sort(
            key=lambda folded_node: folded_node[0].total_duration_ms, reverse=True
        )
        level.folded = folded
        level.child_levels = []  # frees the finished children
    return root_level.folded


def _group_by_shape(
    nodes: List[CallTreeNode], shape_keys: _ShapeKeys
) -> List[List[CallTreeNode]]:
    """Group nodes with the same shape, in order of first appearance."""
    groups: Dict[int, List[CallTreeNode]] = {}
    for node in nodes:
        groups.setdefault(shape_keys.key(node), []).append(node)
    return list(groups.values())


def _merge_group_node(group: List[CallTreeNode]) -> CallTreeNode:
    """Merge identical siblings into one node without children."""
    first = group[0]
    kernels: Dict[str, KernelStats] = {}
    for node in group:
        for kernel_name, stats in node.kernels.items():
            _add_kernel_stats(kernels, kernel_name, stats)
    merged = CallTreeNode(
        name=first.name,
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
    return merged


def _copy_subtree(node: CallTreeNode) -> CallTreeNode:
    """Copy a node and its descendants; the copies share the kernel stats."""
    copies: Dict[int, CallTreeNode] = {}
    for current in _iter_post_order(node):
        # pop frees each child's copy once its parent has used it.
        copies[id(current)] = replace(
            current, children=[copies.pop(id(child)) for child in current.children]
        )
    return copies[id(node)]


def _copy_scoped_operator_node(
    node: CallTreeNode,
    matched_node_ids: Set[int],
    inherited_backends: Set[str],
) -> Optional[Tuple[CallTreeNode, NodeRollup]]:
    """Copy one node, keeping match kernels and in-scope descendant kernels.

    Returns the copy with its completed rollup, or None when nothing under the
    node is kept. Ancestors use the rollups of their kept children instead of
    walking the copied descendants again.
    """
    scoped: Dict[int, Optional[Tuple[CallTreeNode, NodeRollup]]] = {}
    stack: List[Tuple[CallTreeNode, Set[str], bool]] = [
        (node, inherited_backends, False)
    ]
    while stack:
        current, inherited, children_done = stack.pop()
        node_is_match = id(current) in matched_node_ids
        if not children_done:
            if node_is_match and current.backend == USER_REGION_BACKEND:
                copied_subtree = _copy_subtree(current)
                scoped[id(current)] = (
                    copied_subtree,
                    rollup_node_stats(copied_subtree),
                )
                continue
            descendant_backends = inherited
            if node_is_match and current.backend is not None:
                descendant_backends = set(inherited)
                descendant_backends.add(current.backend)
            stack.append((current, inherited, True))
            stack.extend(
                (child, descendant_backends, False)
                for child in reversed(current.children)
            )
            continue
        # pop frees each child's result once its parent has used it.
        kept_children = [
            scoped_child
            for child in current.children
            if (scoped_child := scoped.pop(id(child))) is not None
        ]
        backend_in_scope = current.backend is not None and current.backend in inherited
        keep_kernels = node_is_match or backend_in_scope
        if not keep_kernels and not kept_children:
            scoped[id(current)] = None
            continue
        copied_node = replace(
            current,
            children=[child for child, _child_rollup in kept_children],
            kernels=current.kernels if keep_kernels else {},
        )
        rollup = _roll_up_from_child_rollups(
            copied_node, [child_rollup for _child, child_rollup in kept_children]
        )
        scoped[id(current)] = (copied_node, rollup)
    return scoped[id(node)]
