# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Operator summary construction and aggregation across call-tree paths."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from ml_api_tracing.call_tree import iter_call_tree_paths
from ml_api_tracing.models import CallTreeForest, CallTreeNode
from utils.utils_common import NS_TO_MS


def build_operator_summary(
    call_trees: CallTreeForest,
) -> pd.DataFrame:
    """One row per operator path that launched GPU work.

    Rows combine the same path across locations. Times are milliseconds.
    Dispatches and GPU time include callees, so Pct_Total_GPU can exceed 100%.
    Empty input returns an empty frame with the summary columns. Sorted by
    Total_GPU descending, then Operator.
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
    all_roots = [root for roots in call_trees.values() for root in roots]
    rows = [
        _operator_summary_row(node, path)
        for node, path in iter_call_tree_paths(all_roots)
        if node.kernel_launches > 0
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    rows = _aggregate_operator_summary_rows(rows)
    grand_total_ms = sum(root.total_duration_ms for root in all_roots)
    if grand_total_ms > 0:
        for row in rows:
            row["Pct_Total_GPU"] = 100.0 * row["Total_GPU"] / grand_total_ms

    summary_df = pd.DataFrame(rows, columns=columns)
    return summary_df.sort_values(
        by=["Total_GPU", "Operator"],
        ascending=[False, True],
        ignore_index=True,
    )


def _node_source_location(node: CallTreeNode) -> str:
    """file:line from the node, or empty when file_name is unset."""
    if not node.file_name:
        return ""
    if node.line_number is None:
        return node.file_name
    return f"{node.file_name}:{node.line_number}"


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and pd.isna(value)


def _aggregate_operator_summary_rows(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine per-node rows that share an Operator path across locations."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Operator"]), []).append(row)

    aggregated: List[Dict[str, Any]] = []
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


def _operator_summary_row(node: CallTreeNode, path: Tuple[str, ...]) -> Dict[str, Any]:
    """Build the build_operator_summary row for one node that launched GPU work."""
    has_calls = len(node.invocation_ids) > 0
    calls = len(node.invocation_ids) if has_calls else float("nan")
    dispatches = node.kernel_launches
    total_gpu_ms = node.total_duration_ms
    return {
        "Operator": "/".join(path),
        "Location": _node_source_location(node),
        "Calls": calls,
        "Dispatches": dispatches,
        "Dispatches_Per_Call": (dispatches / calls if has_calls else float("nan")),
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
    }
