# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from utils import csv_compression, schema
from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)
from utils.ml_api_trace_errors import (
    ForwardThreadNotFoundError,
    MarkerNotNestedError,
    MissingSourceLocationError,
    OverlappingMarkerRangeError,
    PassMarkerMismatchError,
    UnaccountedKernelError,
    UncorrelatedForwardIntervalError,
)
from utils.utils_counter_defs import UNIT_COUNTER

NS_TO_MS = 1.0 / 1_000_000.0

# Canonical column-name preference order for Percent of Peak lookups
VALUE_COL_PREFERENCE: tuple[str, ...] = ("Avg", "Value")
PEAK_COL_PREFERENCE: tuple[str, ...] = ("Peak", "Peak (Empirical)")


def get_bw_scale_and_unit(value: float) -> tuple[float, str]:
    """Return the divisor and suffix for a bandwidth value in Bytes/s."""
    if value >= 1e12:
        return 1e12, "TB/s"
    if value >= 1e9:
        return 1e9, "GB/s"
    if value >= 1e6:
        return 1e6, "MB/s"
    if value >= 1e3:
        return 1e3, "KB/s"
    return 1.0, "B/s"


def format_bw_human_readable(
    value: Union[int, float, str, None], unit: str = "Bytes/s", precision: int = 2
) -> str:
    """Format bandwidth to human-readable string (e.g. 1.5 TB/s).

    Accepts Bytes/s (default) or legacy GB/s input.
    Returns 'NaN' for NaN, 'N/A' for None/invalid.
    """
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (ValueError, TypeError):
        return "N/A"

    if math.isnan(numeric_value):
        return "NaN"

    bytes_per_sec = numeric_value * 1e9 if unit == "GB/s" else numeric_value
    divisor, output_unit = get_bw_scale_and_unit(bytes_per_sec)
    return f"{bytes_per_sec / divisor:.{precision}f} {output_unit}"


@dataclass
class KernelStats:
    """Aggregated kernel launch stats for one kernel name.

    min_duration_ns and max_duration_ns are None until a dispatch with a
    non-zero duration is observed.
    """

    launches: int = 0
    total_duration_ns: float = 0.0
    min_duration_ns: Optional[float] = None
    max_duration_ns: Optional[float] = None
    kernel_id: Optional[int] = None


@dataclass
class CallTreeNode:
    """A node in the operator call tree.

    children is the list of child operator nodes. file_name, line_number,
    backend, start_timestamp, end_timestamp, t_tid, f_tid, and thread_id
    are optional fields copied from a marker row when set. invocation_ids
    stores marker-start strings for this node.

    Inclusive over this node plus all descendants:
      kernel_launches, total_duration_ms, min/max/mean dispatch stats.
    """

    name: str
    children: list["CallTreeNode"] = field(default_factory=list)
    kernels: dict[str, KernelStats] = field(default_factory=dict)
    kernel_launches: int = 0
    total_duration_ms: float = 0.0
    invocation_ids: set[str] = field(default_factory=set)
    min_dispatch_ns: Optional[float] = None
    max_dispatch_ns: Optional[float] = None
    mean_dispatch_ns: Optional[float] = None
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    backend: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    t_tid: Optional[str] = None
    f_tid: Optional[str] = None
    thread_id: Optional[str] = None

    @property
    def call_count(self) -> int:
        return len(self.invocation_ids)


@dataclass
class NodeRollup:
    """Inclusive subtree aggregate returned by rollup_node_stats."""

    launches: int
    total_duration_ns: float
    min_dispatch_ns: Optional[float]
    max_dispatch_ns: Optional[float]


def simplify_kernel_name(full_kernel_name: str) -> str:
    """Simplify a kernel name for display by stripping templates and namespaces.

    Strips ``void`` prefix, template parameters, and namespace qualifiers so
    that e.g. ``void at::native::vectorized_elementwise_kernel<4, ...>``
    becomes ``vectorized_elementwise_kernel``.
    """
    name = full_kernel_name.strip()
    if name.startswith("void "):
        name = name[5:]

    if "<" in name:
        main_part = name.split("<")[0]
    elif "(" in name:
        main_part = name.split("(")[0]
    else:
        main_part = name

    if "::" in main_part:
        function_name = main_part.split("::")[-1].strip()
        return function_name if function_name else name.strip()

    return main_part.strip()


def rollup_node_stats(node: CallTreeNode) -> NodeRollup:
    """Bottom-up rollup over this node and all descendants.

    Sets inclusive fields on node: kernel_launches, total_duration_ms,
    min_dispatch_ns, max_dispatch_ns, mean_dispatch_ns.

    Subtrees with no non-zero-duration dispatch leave min/max/mean as None so
    callers can render N/A rather than a misleading 0.
    """
    launches = 0
    total_duration_ns = 0.0
    mins: list[float] = []
    maxes: list[float] = []

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


def decode_marker_name(name: str) -> str:
    """Decode a percent-encoded marker segment ('%2F' -> '/', '%25' -> '%')."""
    return name.replace("%2F", "/").replace("%25", "%")


def _split_name_and_location(first_token: str) -> tuple[str, str, object]:
    """Split `{name}:{location}` from the right into name, file, and line."""
    if first_token.endswith(":n/a"):
        return first_token[: -len(":n/a")], "", ""
    parts = first_token.rsplit(":", 2)
    if len(parts) == 3:
        operator_name, file_part, line_part = parts
        if file_part and line_part.isdigit():
            return operator_name, Path(file_part).name, int(line_part)
    return first_token, "", ""


def parse_marker_function(function_value: object) -> dict[str, Any]:
    """Parse one Function cell into operator, location, backend, and wire keys."""
    if function_value is None or (
        isinstance(function_value, float) and pd.isna(function_value)
    ):
        raw = ""
    else:
        raw = str(function_value)
    tokens = raw.split("|") if raw else [""]
    first_token = tokens[0]
    if ":#" in first_token or "@" in first_token:
        console_error(
            "analysis",
            f"Stacked marker wire is not supported: {raw}",
        )
    operator_name, file_name, line_number = _split_name_and_location(first_token)
    parsed_keys = {
        "seqNr": "n/a",
        "tid": "n/a",
        "ftid": "n/a",
        "scope": "n/a",
        "args": "n/a",
    }
    backend = "user"
    for token in tokens[1:]:
        if token in KNOWN_ML_API_BACKENDS:
            backend = token
            continue
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key == "args":
            parsed_keys["args"] = value.replace("%7C", "|")
        elif key in parsed_keys:
            parsed_keys[key] = value
    t_tid = "" if parsed_keys["tid"] in ("", "n/a") else parsed_keys["tid"]
    f_tid = "" if parsed_keys["ftid"] in ("", "n/a") else parsed_keys["ftid"]
    return {
        "Operator_Name": decode_marker_name(operator_name),
        "File_Name": file_name,
        "Line_Number": line_number,
        "Backend": backend,
        "seqNr": parsed_keys["seqNr"],
        "T_Tid": t_tid,
        "F_Tid": f_tid,
        "scope": parsed_keys["scope"],
        "args": parsed_keys["args"],
    }


def _apply_parsed_function_columns(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Add parse_marker_function columns, keeping Function and Thread_Id."""
    if trace_df.empty:
        return trace_df
    parsed_rows = trace_df["Function"].map(parse_marker_function)
    parsed_df = pd.DataFrame(list(parsed_rows), index=trace_df.index)
    return pd.concat([trace_df, parsed_df], axis=1)


def _optional_marker_file_name(value: object) -> Optional[str]:
    if value is None or value == "" or pd.isna(value):
        return None
    return str(value)


def _optional_marker_line_number(value: object) -> Optional[int]:
    if value is None or value == "" or pd.isna(value):
        return None
    return int(value)


def _optional_pytorch_tid(value: object) -> Optional[str]:
    if value is None or value == "" or pd.isna(value):
        return None
    text = str(value)
    if text in ("n/a", "0"):
        return None
    return text


def _sequence_from_cell(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _kernel_stats_from_marker_row(row: object) -> dict[str, KernelStats]:
    kernel_names = _sequence_from_cell(getattr(row, "Kernel_Names", []))
    kernel_starts = _sequence_from_cell(getattr(row, "Kernel_Start_Timestamps", []))
    kernel_ends = _sequence_from_cell(getattr(row, "Kernel_End_Timestamps", []))
    kernels: dict[str, KernelStats] = {}
    for kernel_name, kernel_start, kernel_end in zip(
        kernel_names, kernel_starts, kernel_ends
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
        if duration_ns > 0:
            if stats.min_duration_ns is None or duration_ns < stats.min_duration_ns:
                stats.min_duration_ns = duration_ns
            if stats.max_duration_ns is None or duration_ns > stats.max_duration_ns:
                stats.max_duration_ns = duration_ns
    return kernels


def _call_tree_node_from_marker_row(row: object) -> CallTreeNode:
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
        t_tid=_optional_pytorch_tid(getattr(row, "T_Tid", "")),
        f_tid=_optional_pytorch_tid(getattr(row, "F_Tid", "")),
        thread_id=str(getattr(row, "Thread_Id", "")),
    )
    node.invocation_ids.add(str(row.Start_Timestamp))
    return node


def nest_marker_intervals(
    trace_df: pd.DataFrame,
) -> dict[str, list[CallTreeNode]]:
    """Nest marker intervals per Thread_Id using timestamp containment."""
    forest: dict[str, list[CallTreeNode]] = {}
    if trace_df.empty:
        return forest
    for thread_id, group in trace_df.groupby("Thread_Id", sort=False):
        roots: list[CallTreeNode] = []
        open_ranges: list[tuple[CallTreeNode, float, float]] = []
        thread_key = str(thread_id)
        for row in group.itertuples(index=False):
            start = float(row.Start_Timestamp)
            end = float(row.End_Timestamp)
            while open_ranges and open_ranges[-1][2] <= start:
                open_ranges.pop()
            if open_ranges and end > open_ranges[-1][2]:
                parent_node, parent_start, parent_end = open_ranges[-1]
                raise OverlappingMarkerRangeError(
                    thread_id=thread_key,
                    first_name=parent_node.name,
                    first_start=parent_start,
                    first_end=parent_end,
                    second_name=str(row.Operator_Name),
                    second_start=start,
                    second_end=end,
                )
            node = _call_tree_node_from_marker_row(row)
            if open_ranges:
                open_ranges[-1][0].children.append(node)
            else:
                roots.append(node)
            open_ranges.append((node, start, end))
        for root in roots:
            rollup_node_stats(root)
        forest[thread_key] = roots
    return forest


def _forward_tid_from_tree(node: CallTreeNode) -> Optional[str]:
    if node.f_tid is not None:
        return node.f_tid
    for child in node.children:
        found = _forward_tid_from_tree(child)
        if found is not None:
            return found
    return None


def _tree_has_t_tid(node: CallTreeNode, pytorch_tid: str) -> bool:
    if node.t_tid == pytorch_tid:
        return True
    return any(_tree_has_t_tid(child, pytorch_tid) for child in node.children)


def _thread_ids_with_pytorch_tid(
    forest: dict[str, list[CallTreeNode]], pytorch_tid: str
) -> list[str]:
    return [
        thread_id
        for thread_id, roots in forest.items()
        if any(_tree_has_t_tid(root, pytorch_tid) for root in roots)
    ]


def _interval_contains(node: CallTreeNode, start: float, end: float) -> bool:
    if node.start_timestamp is None or node.end_timestamp is None:
        return False
    return node.start_timestamp <= start and end <= node.end_timestamp


def _deepest_containing_node(
    nodes: list[CallTreeNode], start: float, end: float
) -> Optional[CallTreeNode]:
    for node in nodes:
        if not _interval_contains(node, start, end):
            continue
        nested = _deepest_containing_node(node.children, start, end)
        return nested if nested is not None else node
    return None


def attach_unlocated_trees_by_forward_thread(
    forest: dict[str, list[CallTreeNode]],
) -> None:
    """Stack torch/triton trees with no source onto the matching forward thread."""
    pending: list[tuple[str, CallTreeNode]] = []
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
        f_tid = _forward_tid_from_tree(root)
        if f_tid is None:
            raise MissingSourceLocationError(
                operator_name=root.name,
                thread_id=thread_id,
                start_timestamp=start,
            )
        matches = _thread_ids_with_pytorch_tid(forest, f_tid)
        if len(matches) != 1:
            raise ForwardThreadNotFoundError(
                operator_name=root.name,
                thread_id=thread_id,
                start_timestamp=start,
                f_tid=f_tid,
            )
        forward_thread_id = matches[0]
        dest_roots = [node for node in forest[forward_thread_id] if node is not root]
        parent = _deepest_containing_node(dest_roots, start, end)
        if parent is None:
            raise UncorrelatedForwardIntervalError(
                operator_name=root.name,
                thread_id=thread_id,
                start_timestamp=start,
                end_timestamp=end,
                f_tid=f_tid,
                forward_thread_id=forward_thread_id,
            )
        parent.children.append(root)
        forest[thread_id] = [node for node in forest[thread_id] if node is not root]
    for thread_id in [tid for tid, roots in forest.items() if not roots]:
        del forest[thread_id]
    for roots in forest.values():
        for node in roots:
            rollup_node_stats(node)


def _nested_invocation_keys(
    forest: dict[str, list[CallTreeNode]],
) -> set[tuple[str, str]]:
    """Return (Thread_Id, marker-start) pairs present in the forest."""
    keys: set[tuple[str, str]] = set()

    def walk(node: CallTreeNode) -> None:
        thread_id = node.thread_id or ""
        for invocation_id in node.invocation_ids:
            keys.add((thread_id, invocation_id))
        for child in node.children:
            walk(child)

    for roots in forest.values():
        for root in roots:
            walk(root)
    return keys


def _validate_all_markers_nested(
    trace_df: pd.DataFrame, forest: dict[str, list[CallTreeNode]]
) -> None:
    """Raise if a consolidated marker row is missing from the nested forest."""
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
                start_timestamp=row.Start_Timestamp,
            )


def clone_call_tree_node(node: CallTreeNode) -> CallTreeNode:
    """Deep-copy a call-tree node, including descendants and kernel stats."""
    copied = CallTreeNode(
        name=node.name,
        kernels={
            kernel_name: KernelStats(
                launches=stats.launches,
                total_duration_ns=stats.total_duration_ns,
                min_duration_ns=stats.min_duration_ns,
                max_duration_ns=stats.max_duration_ns,
                kernel_id=stats.kernel_id,
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
        t_tid=node.t_tid,
        f_tid=node.f_tid,
        thread_id=node.thread_id,
    )
    copied.invocation_ids = set(node.invocation_ids)
    copied.children = [clone_call_tree_node(child) for child in node.children]
    return copied


def filter_forest_by_backend(
    forest: dict[str, list[CallTreeNode]],
    backend: Optional[str],
) -> dict[str, list[CallTreeNode]]:
    """Copy a forest, keeping ``backend`` nodes and their ancestors.

    ``backend=None`` keeps every node, including ``user``.
    """

    def copy_matching_view(node: CallTreeNode) -> Optional[CallTreeNode]:
        kept_children: list[CallTreeNode] = []
        for child in node.children:
            copied_child = copy_matching_view(child)
            if copied_child is not None:
                kept_children.append(copied_child)
        if backend is not None and node.backend != backend and not kept_children:
            return None
        copied_node = clone_call_tree_node(node)
        copied_node.children = kept_children
        return copied_node

    filtered: dict[str, list[CallTreeNode]] = {}
    for thread_id, roots in forest.items():
        kept_roots: list[CallTreeNode] = []
        for root in roots:
            copied_root = copy_matching_view(root)
            if copied_root is not None:
                kept_roots.append(copied_root)
        if kept_roots:
            filtered[thread_id] = kept_roots
    return filtered


def copy_matched_operator_subtree(
    forest: dict[str, list[CallTreeNode]],
    matched_nodes: list[CallTreeNode],
) -> dict[str, list[CallTreeNode]]:
    """Copy each matched node with its ancestors and descendants."""
    match_ids = {id(node) for node in matched_nodes}
    keep_ids: set[int] = set()

    def walk(node: CallTreeNode, ancestor_matched: bool) -> bool:
        is_match = id(node) in match_ids
        keep_below = ancestor_matched or is_match
        descendant_kept = False
        for child in node.children:
            if walk(child, keep_below):
                descendant_kept = True
        keep_this = keep_below or descendant_kept
        if keep_this:
            keep_ids.add(id(node))
        return keep_this

    for roots in forest.values():
        for root in roots:
            walk(root, False)

    def copy_kept(node: CallTreeNode) -> Optional[CallTreeNode]:
        if id(node) not in keep_ids:
            return None
        copied = clone_call_tree_node(node)
        kept_children: list[CallTreeNode] = []
        for child in node.children:
            copied_child = copy_kept(child)
            if copied_child is not None:
                kept_children.append(copied_child)
        copied.children = kept_children
        return copied

    subtree: dict[str, list[CallTreeNode]] = {}
    for thread_id, roots in forest.items():
        kept_roots: list[CallTreeNode] = []
        for root in roots:
            copied_root = copy_kept(root)
            if copied_root is not None:
                kept_roots.append(copied_root)
        if kept_roots:
            subtree[thread_id] = kept_roots
    return subtree


def _node_source_location(node: CallTreeNode) -> str:
    """file:line from the node, or empty when file_name is unset."""
    if not node.file_name:
        return ""
    if node.line_number is None:
        return node.file_name
    return f"{node.file_name}:{node.line_number}"


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _aggregate_operator_summary_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine per-node rows that share an Operator path across locations."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Operator"]), []).append(row)

    aggregated: list[dict[str, Any]] = []
    for operator, group in grouped.items():
        locations = {row["Location"] for row in group}
        location = next(iter(locations)) if len(locations) == 1 else ""
        call_values = [row["Calls"] for row in group]
        has_calls = not any(_is_nan(value) for value in call_values)
        calls = float(sum(call_values)) if has_calls else float("nan")
        dispatches = float(sum(row["Dispatches"] for row in group))
        total_gpu = float(sum(row["Total_GPU"] for row in group))
        min_values = [
            row["Min_Dispatch"] for row in group if not _is_nan(row["Min_Dispatch"])
        ]
        max_values = [
            row["Max_Dispatch"] for row in group if not _is_nan(row["Max_Dispatch"])
        ]
        mean_weights = [
            (row["Mean_Per_Dispatch"], row["Dispatches"])
            for row in group
            if not _is_nan(row["Mean_Per_Dispatch"])
        ]
        weight_sum = sum(weight for _, weight in mean_weights)
        if mean_weights and weight_sum:
            mean_dispatch = (
                sum(mean * weight for mean, weight in mean_weights) / weight_sum
            )
        else:
            mean_dispatch = float("nan")
        aggregated.append({
            "Operator": operator,
            "Location": location,
            "Calls": calls,
            "Dispatches": dispatches,
            "Dispatches_Per_Call": (
                dispatches / calls if has_calls and calls else float("nan")
            ),
            "Total_GPU": total_gpu,
            "Pct_Total_GPU": float("nan"),
            "Mean_Per_Call": (
                total_gpu / calls if has_calls and calls else float("nan")
            ),
            "Mean_Per_Dispatch": mean_dispatch,
            "Min_Dispatch": min(min_values) if min_values else float("nan"),
            "Max_Dispatch": max(max_values) if max_values else float("nan"),
        })
    return aggregated


def build_operator_summary(
    call_trees: dict[str, list[CallTreeNode]],
) -> pd.DataFrame:
    """Build a one-row-per-operator summary table from the call trees.

    Each row describes one operator path (e.g. aten::matmul) that ran at
    least one GPU kernel. Calls of that path at different locations are
    combined. All time values are in milliseconds.

    Columns:

    - Operator: full path of the operator (e.g. "aten::matmul/aten::mm").

    - Location: file:line when every combined node shares one; empty when
      locations differ or file_name is unset.

    - Calls: how many times this operator was invoked. NaN when the trace
      did not include marker-start invocation ids.

    - Dispatches: how many GPU kernels ran while this operator was on the
      call stack (kernels launched by operators it called also count).

    - Dispatches_Per_Call: Dispatches divided by Calls. NaN when Calls is
      unknown.

    - Total_GPU: total GPU time spent while this operator was on the call
      stack.

    - Pct_Total_GPU: how much of the workload's total GPU time fell while
      this operator was on the call stack. The same kernel time gets
      counted for an operator and for any operator that called it, so the
      column can add up to more than 100%. NaN when no GPU time was
      recorded at all.

    - Mean_Per_Call: average GPU time per call to this operator.

    - Mean_Per_Dispatch, Min_Dispatch, Max_Dispatch: per-kernel timings
      across kernels launched while this operator was on the call stack.

    Operators that ran no GPU kernels are skipped. Empty input returns an
    empty DataFrame with the full column list.

    Sorted by Total_GPU descending, then Operator ascending.
    """
    columns = [
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
    rows: list[dict[str, Any]] = []

    def walk(node: CallTreeNode, location: str, path_parts: list[str]) -> None:
        if node.kernel_launches > 0:
            has_calls = len(node.invocation_ids) > 0
            calls = len(node.invocation_ids) if has_calls else float("nan")
            dispatches = node.kernel_launches
            total_gpu_ms = node.total_duration_ms
            rows.append({
                "Operator": "/".join(path_parts),
                "Location": location,
                "Calls": calls,
                "Dispatches": dispatches,
                "Dispatches_Per_Call": (
                    dispatches / calls if has_calls else float("nan")
                ),
                "Total_GPU": total_gpu_ms,
                "Pct_Total_GPU": float("nan"),
                "Mean_Per_Call": (total_gpu_ms / calls if has_calls else float("nan")),
                "Mean_Per_Dispatch": (
                    node.mean_dispatch_ns * NS_TO_MS
                    if node.mean_dispatch_ns is not None
                    else float("nan")
                ),
                "Min_Dispatch": (
                    node.min_dispatch_ns * NS_TO_MS
                    if node.min_dispatch_ns is not None
                    else float("nan")
                ),
                "Max_Dispatch": (
                    node.max_dispatch_ns * NS_TO_MS
                    if node.max_dispatch_ns is not None
                    else float("nan")
                ),
            })
        for child in node.children:
            walk(child, _node_source_location(child), path_parts + [child.name])

    all_roots: list[CallTreeNode] = []
    for roots in call_trees.values():
        all_roots.extend(roots)
        for root in roots:
            walk(root, _node_source_location(root), [root.name])

    if not rows:
        return pd.DataFrame(columns=columns)

    rows = _aggregate_operator_summary_rows(rows)
    grand_total_ms = sum(root.total_duration_ms for root in all_roots)
    if grand_total_ms > 0:
        for row in rows:
            row["Pct_Total_GPU"] = 100.0 * row["Total_GPU"] / grand_total_ms

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(
        by=["Total_GPU", "Operator"],
        ascending=[False, True],
        ignore_index=True,
    )


_REQUIRED_MARKER_COLUMNS = (
    "Function",
    "Thread_Id",
    "Correlation_ID",
    "Start_Timestamp",
    "End_Timestamp",
)


def _find_ml_api_trace_csv_pairs(workload_dir: Path) -> list[tuple[Path, Path]]:
    """Return (marker, counter) paths for each profiling pass."""
    marker_glob = f"**/ml_api_trace*_marker_api_trace.csv{csv_compression.GZIP_SUFFIX}"
    pairs: list[tuple[Path, Path]] = []
    for marker_path in sorted(workload_dir.glob(marker_glob)):
        marker_csv_name = marker_path.name
        if marker_csv_name.endswith(csv_compression.GZIP_SUFFIX):
            marker_csv_name = marker_csv_name[: -len(csv_compression.GZIP_SUFFIX)]
        counter_csv_name = marker_csv_name.replace(
            "_marker_api_trace.csv", "_counter_collection.csv"
        )
        counter_path = csv_compression.compressed_name(
            marker_path.parent / counter_csv_name
        )
        if counter_path.is_file():
            pairs.append((marker_path, counter_path))
    return pairs


def _rename_column_alias(
    frame: pd.DataFrame, canonical_name: str, alias_name: str
) -> pd.DataFrame:
    """Rename alias_name to canonical_name, or drop the alias when both exist."""
    if alias_name in frame.columns and canonical_name not in frame.columns:
        return frame.rename(columns={alias_name: canonical_name})
    if alias_name in frame.columns:
        return frame.drop(columns=[alias_name])
    return frame


def _load_marker_trace_dataframe(marker_path: Path) -> pd.DataFrame:
    """Load one marker CSV and keep the columns used by later analyze steps."""
    marker_df = _rename_column_alias(
        pd.read_csv(marker_path), "Correlation_ID", "Correlation_Id"
    )
    missing_columns = [
        column for column in _REQUIRED_MARKER_COLUMNS if column not in marker_df.columns
    ]
    if missing_columns:
        console_error(
            "analysis",
            f"Marker CSV {marker_path} is missing required columns {missing_columns}",
        )
    kept_columns = list(_REQUIRED_MARKER_COLUMNS)
    if "GUID" in marker_df.columns:
        kept_columns.append("GUID")
    null_columns = [column for column in kept_columns if marker_df[column].isna().any()]
    if null_columns:
        console_error(
            "analysis",
            f"Marker CSV {marker_path} has null values in {null_columns}",
        )
    return marker_df[kept_columns].copy()


_REQUIRED_COUNTER_COLUMNS = (
    "Correlation_ID",
    "Dispatch_ID",
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
)
_DISPATCH_KEEP_COLUMNS = (
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "Correlation_ID",
    "Dispatch_ID",
    "GUID",
)


def _collapse_counter_dispatches(
    counter_df: pd.DataFrame, counter_path: Path
) -> pd.DataFrame:
    """Collapse long counter rows to one row per GPU dispatch."""
    missing_columns = [
        column
        for column in _REQUIRED_COUNTER_COLUMNS
        if column not in counter_df.columns
    ]
    if missing_columns:
        console_error(
            "analysis",
            f"Counter CSV {counter_path} is missing required columns {missing_columns}",
        )
    null_columns = [
        column
        for column in _REQUIRED_COUNTER_COLUMNS
        if counter_df[column].isna().any()
    ]
    if null_columns:
        console_error(
            "analysis",
            f"Counter CSV {counter_path} has null values in {null_columns}",
        )
    group_keys = ["Dispatch_ID"]
    if "GUID" in counter_df.columns:
        group_keys.append("GUID")
    keep_columns = [
        column for column in _DISPATCH_KEEP_COLUMNS if column in counter_df.columns
    ]
    for group_key in group_keys:
        if group_key not in keep_columns:
            keep_columns.append(group_key)
    collapsed = (
        counter_df[keep_columns]
        .groupby(group_keys, sort=False, dropna=False)
        .first()
        .reset_index()
    )
    return collapsed.rename(
        columns={
            "Start_Timestamp": "Kernel_Start_Timestamp",
            "End_Timestamp": "Kernel_End_Timestamp",
        }
    )


def _join_keys_for_marker_and_dispatch(
    marker_df: pd.DataFrame,
    dispatch_df: pd.DataFrame,
) -> list[str]:
    """Correlation_ID, plus GUID when both frames have that column."""
    join_keys = ["Correlation_ID"]
    if "GUID" in marker_df.columns and "GUID" in dispatch_df.columns:
        join_keys.append("GUID")
    return join_keys


def _outer_join_dispatches_and_markers(
    dispatch_df: pd.DataFrame,
    marker_df: pd.DataFrame,
) -> pd.DataFrame:
    """Full-outer-join unique dispatches with one pass of marker rows."""
    marker_ordered = marker_df.copy()
    marker_ordered["_marker_order"] = range(len(marker_ordered))
    return pd.merge(
        dispatch_df,
        marker_ordered,
        on=_join_keys_for_marker_and_dispatch(marker_df, dispatch_df),
        how="outer",
    )


def _join_pass_marker_and_counter(pair: schema.MlApiTracePair) -> pd.DataFrame:
    """Load one counter CSV, collapse dispatches, and outer-join markers."""
    counter_df = _rename_column_alias(
        pd.read_csv(pair.counter_path), "Correlation_ID", "Correlation_Id"
    )
    counter_df = _rename_column_alias(counter_df, "Dispatch_ID", "Dispatch_Id")
    dispatch_df = _collapse_counter_dispatches(counter_df, pair.counter_path)
    return _outer_join_dispatches_and_markers(dispatch_df, pair.marker_df)


def _add_stitch_key_and_ordinal(pass_frame: pd.DataFrame) -> pd.DataFrame:
    """Add stitch_key from Function (seqNr, tid, ftid kept) and function_ordinal."""
    if pass_frame.empty:
        result = pass_frame.copy()
        result["stitch_key"] = pd.Series(dtype=str)
        result["function_ordinal"] = pd.Series(dtype=int)
        return result
    ordered = pass_frame
    if "_marker_order" in pass_frame.columns:
        ordered = pass_frame.sort_values("_marker_order", kind="mergesort")
    result = ordered.copy()
    result["stitch_key"] = result["Function"].map(str)
    result["function_ordinal"] = result.groupby("stitch_key", sort=False).cumcount()
    return result


_PASS_DROP_COLUMNS = (
    "Correlation_ID",
    "GUID",
    "stitch_key",
    "function_ordinal",
    "_pass_id",
    "_marker_order",
)
_COLLAPSED_MARKER_COLUMNS = (
    "Function",
    "Thread_Id",
    "Start_Timestamp",
    "End_Timestamp",
    "Kernel_Names",
    "Kernel_Start_Timestamps",
    "Kernel_End_Timestamps",
)


def _kernel_names_as_set(names: object) -> frozenset[str]:
    if names is None or (isinstance(names, float) and pd.isna(names)):
        return frozenset()
    return frozenset(str(name) for name in names)


def _collapse_matching_markers_across_passes(
    pass_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Match operator occurrences across passes and keep the first pass row."""
    if not pass_frames:
        return pd.DataFrame(columns=list(_COLLAPSED_MARKER_COLUMNS))
    labeled_frames = []
    for pass_id, frame in enumerate(pass_frames):
        labeled = frame.copy()
        labeled["_pass_id"] = pass_id
        labeled_frames.append(labeled)
    concatenated = pd.concat(labeled_frames, ignore_index=True)
    if concatenated.empty:
        collapsed = concatenated
    elif len(pass_frames) == 1:
        collapsed = concatenated
    else:
        pass_counts = [len(frame) for frame in pass_frames]
        if len(set(pass_counts)) != 1:
            raise PassMarkerMismatchError(
                stitch_key="",
                function_ordinal=None,
                disagreeing_values=f"per-pass marker counts {pass_counts}",
            )
        expected_passes = set(range(len(pass_frames)))
        records: list[dict[str, Any]] = []
        grouped = concatenated.groupby(
            ["stitch_key", "function_ordinal"], sort=False, dropna=False
        )
        for (stitch_key, ordinal), group in grouped:
            present_passes = set(group["_pass_id"].tolist())
            missing_passes = sorted(expected_passes - present_passes)
            if missing_passes:
                raise PassMarkerMismatchError(
                    stitch_key=str(stitch_key),
                    function_ordinal=int(ordinal),
                    disagreeing_values=f"missing passes {missing_passes}",
                )
            kernel_name_sets = [
                _kernel_names_as_set(
                    group[group["_pass_id"] == pass_id].iloc[0]["Kernel_Names"]
                )
                for pass_id in range(len(pass_frames))
            ]
            if len(set(kernel_name_sets)) != 1:
                raise PassMarkerMismatchError(
                    stitch_key=str(stitch_key),
                    function_ordinal=int(ordinal),
                    disagreeing_values=f"Kernel_Names {list(kernel_name_sets)}",
                )
            first = group[group["_pass_id"] == 0].iloc[0]
            records.append({
                "Function": first["Function"],
                "Thread_Id": first["Thread_Id"],
                "Start_Timestamp": first["Start_Timestamp"],
                "End_Timestamp": first["End_Timestamp"],
                "Kernel_Names": first["Kernel_Names"],
                "Kernel_Start_Timestamps": first["Kernel_Start_Timestamps"],
                "Kernel_End_Timestamps": first["Kernel_End_Timestamps"],
            })
        collapsed = pd.DataFrame.from_records(records)
    drop_columns = [
        column for column in _PASS_DROP_COLUMNS if column in collapsed.columns
    ]
    if drop_columns:
        collapsed = collapsed.drop(columns=drop_columns)
    if collapsed.empty:
        return collapsed
    return collapsed.sort_values(
        by=["Thread_Id", "Start_Timestamp", "End_Timestamp"],
        kind="mergesort",
    )


def _unmatched_kernel_rows(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Dispatches whose Correlation_ID is not in that pass's marker CSV."""
    return joined_df[
        joined_df["Kernel_Name"].notna() & joined_df["Function"].isna()
    ].copy()


def _group_kernels_onto_markers(joined_df: pd.DataFrame) -> pd.DataFrame:
    """Collect Kernel_Names for each marker identity in one pass row."""
    if joined_df.empty:
        return joined_df
    ordered = joined_df
    if "_marker_order" in joined_df.columns:
        ordered = joined_df.sort_values("_marker_order", kind="mergesort")
    group_keys = ["Function", "Thread_Id", "Start_Timestamp", "End_Timestamp"]
    records: list[dict[str, Any]] = []
    for _, group in ordered.groupby(group_keys, sort=False, dropna=False):
        first = group.iloc[0]
        kernel_names: list[str] = []
        kernel_starts: list[object] = []
        kernel_ends: list[object] = []
        for row in group.itertuples(index=False):
            kernel_name = row.Kernel_Name
            if pd.isna(kernel_name):
                continue
            kernel_names.append(str(kernel_name))
            kernel_starts.append(row.Kernel_Start_Timestamp)
            kernel_ends.append(row.Kernel_End_Timestamp)
        record: dict[str, Any] = {
            "Function": first["Function"],
            "Thread_Id": first["Thread_Id"],
            "Start_Timestamp": first["Start_Timestamp"],
            "End_Timestamp": first["End_Timestamp"],
            "Correlation_ID": first["Correlation_ID"],
            "Kernel_Names": kernel_names,
            "Kernel_Start_Timestamps": kernel_starts,
            "Kernel_End_Timestamps": kernel_ends,
        }
        if "GUID" in group.columns:
            record["GUID"] = first["GUID"]
        if "_marker_order" in group.columns:
            record["_marker_order"] = first["_marker_order"]
        records.append(record)
    grouped = pd.DataFrame.from_records(records)
    if "_marker_order" in grouped.columns:
        grouped = grouped.sort_values("_marker_order", kind="mergesort")
    return grouped


@demarcate
def process_ml_api_trace_output(
    workload: schema.Workload,
    workload_dir: str,
) -> None:
    """Load, join, nest, and validate ML API marker rows for operator analyze."""
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    csv_pairs = _find_ml_api_trace_csv_pairs(Path(workload_dir))
    if not csv_pairs:
        console_warning(
            "No marker files with corresponding counter files found. "
            "Ensure profiling was done with ML API tracing enabled "
            "(e.g., via '--torch-trace')."
        )
        workload.ml_api_trace_pairs = []
        return

    workload.ml_api_trace_pairs = [
        schema.MlApiTracePair(
            marker_df=_load_marker_trace_dataframe(marker_path),
            counter_path=counter_path,
        )
        for marker_path, counter_path in csv_pairs
    ]
    unmatched_kernel_frames: list[pd.DataFrame] = []
    for pair in workload.ml_api_trace_pairs:
        pair.joined_df = _join_pass_marker_and_counter(pair)
        unmatched_kernel_frames.append(_unmatched_kernel_rows(pair.joined_df))
    workload.unmatched_kernel_frames = unmatched_kernel_frames
    nonempty_unmatched = [frame for frame in unmatched_kernel_frames if not frame.empty]
    if nonempty_unmatched:
        raise UnaccountedKernelError(pd.concat(nonempty_unmatched, ignore_index=True))
    for pair in workload.ml_api_trace_pairs:
        marker_rows = pair.joined_df[pair.joined_df["Function"].notna()].copy()
        pair.joined_df = _add_stitch_key_and_ordinal(
            _group_kernels_onto_markers(marker_rows)
        )
    workload.ml_api_trace_df = _apply_parsed_function_columns(
        _collapse_matching_markers_across_passes([
            pair.joined_df for pair in workload.ml_api_trace_pairs
        ])
    )
    workload.ml_api_call_trees = nest_marker_intervals(workload.ml_api_trace_df)
    attach_unlocated_trees_by_forward_thread(workload.ml_api_call_trees)
    _validate_all_markers_nested(workload.ml_api_trace_df, workload.ml_api_call_trees)


def validate_workload(path: str) -> None:
    """Validate workload directory contains readable, non-empty profiling output."""
    workload_dir = Path(path)
    pmc_perf_path = csv_compression.compressed_name(
        workload_dir / f"{schema.PMC_PERF_FILE_PREFIX}.csv"
    )

    # Find PMC data files (merged or separate)
    if pmc_perf_path.is_file():
        files_to_check = [pmc_perf_path]
    else:
        files_to_check = sorted(
            workload_dir.glob(f"results_*.csv{csv_compression.GZIP_SUFFIX}")
        )

    if not files_to_check:
        console_error("analysis", "No profiling data found.")
        return

    # Validate files are not empty
    for file_path in files_to_check:
        try:
            # read_csv infers gzip from the .gz suffix.
            temp_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            console_error(
                "profiling",
                f"No counter data in {file_path}.\nProfiling data could be corrupt.",
            )
            return
        except csv_compression.CORRUPT_CSV_ERRORS as e:
            console_error(
                "profiling",
                f"Could not read {file_path}: {e}\n"
                "The file is truncated or corrupt, which a profile run that was "
                "killed mid-write leaves behind.\n"
                "Please re-run 'rocprof-compute profile'.",
            )
            return
        if temp_df.dropna().empty:
            console_error(
                "profiling",
                f"Found empty cells in {file_path}.\nProfiling data could be corrupt.",
            )
            break


def add_unit_counter(df: pd.DataFrame) -> None:
    """Add the UNIT_COUNTER column in place: 1 per dispatch, so SUM == N."""
    df[UNIT_COUNTER] = 1


def impute_counters_iteration_multiplex(
    df: pd.DataFrame,
    policy: str,
    workload_dir: Path,
) -> pd.DataFrame:
    """
    Perform data imputation for missing counter values due to iteration multiplexing.
    """
    # Counter buckets configured for the workload. A kernel needs at least
    # this many dispatches to cover every bucket.
    num_perfmon_files = len(list(workload_dir.glob("perfmon/*.txt"))) + len(
        list(workload_dir.glob("perfmon/pmc_perf_*.yaml"))
    )

    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Kernel_ID",
    ]

    # Group by unique kernel configurations
    unique_occurences = (
        df.groupby("Kernel_Name")
        if policy == "kernel"
        else df.groupby(
            [
                "Kernel_Name",
                "Grid_Size",
                "Workgroup_Size",
                "LDS_Per_Workgroup",
            ],
            as_index=False,
        )
    )

    counter_columns = [col for col in df.columns if col not in non_counter_column_index]
    # Collect imputed groups as dataframes
    group_dfs: list[pd.DataFrame] = []

    # Log imputation task summary before processing
    console_debug(
        f"Performing data imputation on {len(df)} dispatches "
        f"across {unique_occurences.ngroups} unique kernel configurations"
    )

    incomplete_kernel_names: set[str] = set()

    for _, group in unique_occurences:
        # Skip imputation entirely for undersampled kernels: nullify
        # counters so metric evaluation excludes them. Non-counter columns
        # are preserved for Top Stats (Block 1) timing.
        if len(group) < num_perfmon_files:
            group_copy = group.copy()
            group_copy[counter_columns] = np.nan
            incomplete_kernel_names.add(group_copy["Kernel_Name"].iloc[0])
            group_dfs.append(group_copy)
            continue

        # Identify counter buckets
        counter_groups: set[frozenset[str]] = set()
        for _, row in group.iterrows():
            # Set of counter column names with non empty values
            cols_frozenset = frozenset(row[counter_columns].dropna().index)
            # If no counters found for this dispatch, continue
            if not cols_frozenset:
                continue
            # Since counter buckets are repeated in round robin fashion,
            # we can stop once we see a repeated bucket
            if cols_frozenset in counter_groups:
                break
            counter_groups.add(cols_frozenset)

        # If no counters found for this group, continue
        if not counter_groups:
            continue

        # Iterate over subgroups of dispatches containing
        # all counters and impute missing values
        # Create subgroup_id column for groupby: 0,0,0,...,1,1,1,...,2,2,2,...
        # Use numpy for vectorized operation
        group_copy = group.copy()
        group_copy["__subgroup_id"] = np.arange(len(group_copy)) // len(counter_groups)
        # groupby().bfill() automatically excludes the grouping column from result
        group_copy[counter_columns] = (
            group_copy[[*counter_columns, "__subgroup_id"]]
            .groupby("__subgroup_id", group_keys=False)
            .bfill()  # Propagate first valid value backward to start of subgroup
            .ffill()  # Propagate forward to end of subgroup
        )
        group_copy = group_copy.drop(columns=["__subgroup_id"])
        group_dfs.append(group_copy)

    if incomplete_kernel_names:
        _warn_kernels_with_incomplete_coverage(incomplete_kernel_names)

    if not group_dfs:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(group_dfs, ignore_index=True)


def _warn_kernels_with_incomplete_coverage(incomplete_kernel_names: set[str]) -> None:
    """
    Emit a warning listing kernels excluded from metrics due to missing counter data.
    """
    kernel_list = "\n\n".join(
        f"  Kernel {i}: {name}"
        for i, name in enumerate(sorted(incomplete_kernel_names), start=1)
    )
    console_warning(
        "imputation",
        (
            f"Some kernels have missing counter data after imputation and "
            f"have been excluded from metrics calculations:\n\n"
            f"{kernel_list}\n\n"
            f"Execution times for these kernels are still shown in Top Stats.\n"
            f"To get more complete kernel coverage for metrics calculations, "
            f"you may consider:\n"
            f"  - disabling iteration multiplexing to use application replay\n"
            f"  - increasing the number of iterations for these kernels in "
            f"the workload"
        ),
    )


def process_rocpd_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge counters across unique dispatches from the
    input dataframe and return processed dataframe.
    """
    if df.empty:
        return df

    data: list[dict[str, Any]] = []

    # Group by unique kernel and merge into a single row
    for _, group_df in df.groupby([
        "Dispatch_ID",
        "Kernel_Name",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
    ]):
        row = {
            "GPU_ID": group_df["GPU_ID"].iloc[0],
            "Grid_Size": group_df["Grid_Size"].iloc[0],
            "Workgroup_Size": group_df["Workgroup_Size"].iloc[0],
            "LDS_Per_Workgroup": group_df["LDS_Per_Workgroup"].iloc[0],
            "Scratch_Per_Workitem": group_df["Scratch_Per_Workitem"].iloc[0],
            "Arch_VGPR": group_df["Arch_VGPR"].iloc[0],
            "Accum_VGPR": group_df["Accum_VGPR"].iloc[0],
            "SGPR": group_df["SGPR"].iloc[0],
            "Kernel_Name": group_df["Kernel_Name"].iloc[0],
            "Kernel_ID": group_df["Kernel_ID"].iloc[0],
            "Start_Timestamp": group_df["Start_Timestamp"].iloc[0],
            "End_Timestamp": group_df["End_Timestamp"].iloc[0],
        }
        # Each counter will become its own column
        row.update(dict(zip(group_df["Counter_Name"], group_df["Counter_Value"])))
        data.append(row)
    df = pd.DataFrame(data)
    # Rank GPU IDs, map lowest number to 0, next to 1, etc.
    df["GPU_ID"] = df["GPU_ID"].rank(method="dense").astype(int) - 1
    # Reset dispatch IDs
    df["Dispatch_ID"] = range(1, len(df) + 1)
    return df


def get_matrix_ops_type(gpu_series: str) -> str:
    """
    Get the matrix operation type supported by the profiled hardware in roofline
    run_parameters.
    For the supported architecture of this tool, only CDNA2/3/4 supports Matrix
    Fused Multiply-Add instructions; all other architectures support Warp
    Matrix Multiply-Accumulate operations.
    """
    if gpu_series.upper() in ["MI200", "MI300", "MI350"]:
        return "MFMA"
    return "WMMA"
