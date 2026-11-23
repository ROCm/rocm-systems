# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML API marker call-tree processing and inclusive kernel statistics."""

from collections.abc import Sequence
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
from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS
from ml_api_tracing.marker_format import split_operator_args
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
    minimum_durations: List[float] = []
    maximum_durations: List[float] = []

    for stats in node.kernels.values():
        launches += stats.launches
        total_duration_ns += stats.total_duration_ns
        if stats.min_duration_ns is not None:
            minimum_durations.append(stats.min_duration_ns)
        if stats.max_duration_ns is not None:
            maximum_durations.append(stats.max_duration_ns)

    for child in node.children:
        child_rollup = rollup_node_stats(child)
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
    for node in nodes:
        path = parent_path + (node.name,)
        yield node, path
        yield from iter_call_tree_paths(node.children, path)


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
    if node.launcher_thread_id is not None:
        return node.launcher_thread_id
    for child in node.children:
        found = _launcher_thread_id_from_tree(child)
        if found is not None:
            return found
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
