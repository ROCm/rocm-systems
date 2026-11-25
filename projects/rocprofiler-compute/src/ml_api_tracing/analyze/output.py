# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ML operator call-tree reports, summary tables and terminal formatting."""

import math
import shutil
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tabulate import tabulate

from ml_api_tracing.analyze.call_tree import (
    copy_matched_operator_subtree,
    fold_identical_sibling_subtrees,
    iter_call_tree_paths,
)
from ml_api_tracing.analyze.definitions import CallTreeForest, CallTreeNode
from ml_api_tracing.analyze.operator_patterns import operator_backend_labels
from ml_api_tracing.analyze.selection import kernel_ids_from_top
from ml_api_tracing.marker_format import split_operator_args
from utils import schema
from utils.utils_common import NS_TO_MS

ARGS_DISPLAY_MAX_ITEMS = 8
ARGS_DISPLAY_MAX_CHARS = 160
ARGS_DISPLAY_MAX_VARIANTS = 5

_REPORT_WIDTH = 80
_DURATION_US_THRESHOLD_MS = 0.01
_MIN_KERNEL_CONTENT_WIDTH = 20
_MIN_WRAPPED_KERNEL_NAME_WIDTH = 8
_DEFAULT_TREE_MIN_WRAP_WIDTH = 72
_DEFAULT_TREE_MAX_WRAP_WIDTH = 120
_OPERATOR_SUMMARY_NAME_WRAP_WIDTH = 72


def list_operators(
    workload_path: str,
    workload: schema.Workload,
    kernel_top_df: pd.DataFrame,
    backends: List[str],
) -> None:
    """List the operators, numbering kernels as in the top-kernel table."""
    list_ml_operators(
        workload_path,
        workload.ml_api_call_trees,
        framework_labels=[operator_backend_labels(backends)],
        backends=backends,
        kernel_ids=kernel_ids_from_top(kernel_top_df),
    )


def show_matched_operator_report(workload: schema.Workload) -> None:
    """Print the report for the operators that matched the --operator patterns."""
    if not workload.ml_api_call_trees:
        return
    patterns_str = ", ".join(
        dict.fromkeys(
            pattern
            for patterns in workload.operator_patterns.values()
            for pattern in patterns
        )
    )
    show_matched_operators(
        workload.ml_api_call_trees,
        [operator_backend_labels(list(workload.operator_patterns))],
        patterns_str,
        workload.operator_kernel_ids,
    )


def list_ml_operators(
    workload_path: str,
    call_trees: CallTreeForest,
    framework_labels: List[str],
    backends: List[str],
    kernel_ids: Dict[str, int],
) -> None:
    """Display the operators sorted by GPU kernel duration.

    framework_labels sets the heading text, such as PyTorch, Triton, User, or ML API.
    """
    forest_view = _operator_listing_view(call_trees, backends)
    if not forest_view:
        print(f"\n{', '.join(framework_labels)} Operators in: {workload_path}")
        print("Total: 0 operators")
        return

    print(f"\n{'=' * _REPORT_WIDTH}")
    print(f"{', '.join(framework_labels)} Operator Call Tree: {workload_path}")
    print("Sorted by total GPU kernel duration.")
    print(f"{'=' * _REPORT_WIDTH}")
    show_call_tree(forest_view, kernel_ids)
    show_operator_summary(build_operator_summary(forest_view))
    print(f"\n{'=' * _REPORT_WIDTH}")


def show_matched_operators(
    call_trees: CallTreeForest,
    framework_labels: List[str],
    patterns_str: str,
    kernel_ids: Dict[str, int],
) -> None:
    """Print the matched operators sorted by GPU kernel duration."""
    if not call_trees:
        return
    labels = ", ".join(framework_labels)
    print(f"\n{'=' * _REPORT_WIDTH}")
    print(f"Matched {labels} Operators: {patterns_str}")
    print("Sorted by total GPU kernel duration.")
    print(f"{'=' * _REPORT_WIDTH}")
    show_call_tree(call_trees, kernel_ids)
    show_operator_summary(build_operator_summary(call_trees))
    print(f"{'=' * _REPORT_WIDTH}")


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


def simplify_kernel_name(full_kernel_name: str) -> str:
    """Simplify a kernel name for display by stripping templates and namespaces.

    Strips void prefix, template parameters, and namespace qualifiers so
    that e.g. void at::native::vectorized_elementwise_kernel<4, ...>
    becomes vectorized_elementwise_kernel.
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


def format_operator_args(
    args_blob: str,
    max_items: int = ARGS_DISPLAY_MAX_ITEMS,
    max_chars: int = ARGS_DISPLAY_MAX_CHARS,
) -> str:
    """Render an operator-args blob, truncated to max_items and max_chars."""
    tokens = split_operator_args(args_blob)
    if not tokens:
        return ""

    shown_tokens = tokens[:max_items]
    if len(tokens) > max_items:
        shown_tokens.append("...")
    formatted_args = "(" + ", ".join(shown_tokens) + ")"
    if len(formatted_args) > max_chars:
        kept_length = max(max_chars - 4, 0)
        formatted_args = (formatted_args[:kept_length].rstrip() + "...)")[:max_chars]
    return formatted_args


def format_duration(duration_ms: Optional[float]) -> str:
    """Format a duration in ms; switch to us below 0.01 ms; None/NaN render as N/A."""
    if duration_ms is None:
        return "N/A"
    if isinstance(duration_ms, float) and math.isnan(duration_ms):
        return "N/A"
    if duration_ms < _DURATION_US_THRESHOLD_MS:
        return f"{duration_ms * 1000:.2f} us"
    return f"{duration_ms:.2f} ms"


def format_node_stats(node: CallTreeNode) -> str:
    """Format operator-node stats (calls, dispatches, total, dispatch_mean/min/max).

    dispatch_mean / dispatch_min / dispatch_max are per kernel dispatch.
    """
    mean_ms = (
        node.mean_dispatch_ns * NS_TO_MS if node.mean_dispatch_ns is not None else None
    )
    min_ms = (
        node.min_dispatch_ns * NS_TO_MS if node.min_dispatch_ns is not None else None
    )
    max_ms = (
        node.max_dispatch_ns * NS_TO_MS if node.max_dispatch_ns is not None else None
    )
    calls_prefix = f"calls: {node.call_count}, " if len(node.invocation_ids) > 0 else ""
    return (
        f"({calls_prefix}"
        f"dispatches: {node.kernel_launches}, "
        f"total: {format_duration(node.total_duration_ms)}, "
        f"dispatch_mean: {format_duration(mean_ms)}, "
        f"dispatch_min: {format_duration(min_ms)}, "
        f"dispatch_max: {format_duration(max_ms)})"
    )


def format_node_args(node: CallTreeNode) -> str:
    """Return the leading-space args=... segment for a single blob."""
    variants = node.args_variants
    if len(variants) != 1:
        return ""
    args_blob, _call_count = variants[0]
    formatted_args = format_operator_args(args_blob)
    return f" args={formatted_args}" if formatted_args else ""


def format_args_variant_call_count(call_count: int) -> str:
    """Return the call-count label for one args variant."""
    if call_count <= 0:
        return ""
    return f"{call_count} call" if call_count == 1 else f"{call_count} calls"


def format_args_variant_lines(node: CallTreeNode) -> List[str]:
    """Return the args variants block, most frequent first."""
    variants = node.args_variants
    if len(variants) < 2:
        return []

    shown_variants = variants[:ARGS_DISPLAY_MAX_VARIANTS]
    call_count_labels = [
        format_args_variant_call_count(call_count) for _, call_count in shown_variants
    ]
    label_width = max(len(label) for label in call_count_labels)
    lines = ["args variants:"]
    lines.extend(
        f"  {label:<{label_width}}  {format_operator_args(args_blob)}"
        for label, (args_blob, _) in zip(call_count_labels, shown_variants)
    )
    hidden_variant_count = len(variants) - len(shown_variants)
    if hidden_variant_count:
        plural = "" if hidden_variant_count == 1 else "s"
        lines.append(f"  ... {hidden_variant_count} more variant{plural}")
    return lines


def get_tree_wrap_width(
    min_width: int = _DEFAULT_TREE_MIN_WRAP_WIDTH,
    max_width: int = _DEFAULT_TREE_MAX_WRAP_WIDTH,
) -> int:
    """Pick wrap width based on terminal size to avoid terminal hard-wrap artifacts."""
    terminal_cols = shutil.get_terminal_size((max_width, 20)).columns
    safe_width = max(terminal_cols - 2, min_width)
    return min(safe_width, max_width)


def build_continuation_indent(prefix: str, continuation_prefix: str) -> str:
    """Return the wrapped-line indent under prefix."""
    if not continuation_prefix:
        return " " * len(prefix)
    padding = max(len(prefix) - len(continuation_prefix), 0)
    return continuation_prefix + " " * padding


def print_wrapped_tree_line(
    prefix: str,
    body: str,
    width: Optional[int] = None,
    break_long_words: bool = False,
    continuation_prefix: str = "",
) -> None:
    """Print a tree line and wrap continuation lines to preserve indentation."""
    effective_width = get_tree_wrap_width() if width is None else width
    print(
        textwrap.fill(
            body,
            width=effective_width,
            initial_indent=prefix,
            subsequent_indent=build_continuation_indent(prefix, continuation_prefix),
            break_long_words=break_long_words,
            break_on_hyphens=False,
        )
    )


def print_wrapped_kernel_line(
    prefix: str,
    kernel_name: str,
    suffix: str,
    continuation_prefix: str = "",
) -> None:
    """Wrap long kernel names while keeping suffix attached to final name chunk."""
    content_width = max(get_tree_wrap_width() - len(prefix), _MIN_KERNEL_CONTENT_WIDTH)

    inline = f"{kernel_name} {suffix}"
    if len(inline) <= content_width:
        print(f"{prefix}{inline}")
        return

    # Reserve room so suffix is never detached on its own line.
    name_width = max(content_width - len(suffix) - 1, _MIN_WRAPPED_KERNEL_NAME_WIDTH)
    wrapped_name = textwrap.wrap(
        kernel_name,
        width=name_width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped_name:
        print(f"{prefix}{suffix}")
        return

    continuation = build_continuation_indent(prefix, continuation_prefix)

    for i, chunk in enumerate(wrapped_name):
        if i == 0:
            if len(wrapped_name) == 1:
                print(f"{prefix}{chunk} {suffix}")
            else:
                print(f"{prefix}{chunk}")
        elif i == len(wrapped_name) - 1:
            print(f"{continuation}{chunk} {suffix}")
        else:
            print(f"{continuation}{chunk}")


def show_call_tree(call_trees: CallTreeForest, kernel_ids: Dict[str, int]) -> None:
    """Print top-level marker nodes sorted by total GPU duration.

    Identical sibling subtrees are folded. kernel_ids maps kernel names to the
    ids shown next to them.
    """
    roots: List[CallTreeNode] = []
    for thread_roots in call_trees.values():
        roots.extend(fold_identical_sibling_subtrees(thread_roots))
    roots.sort(key=lambda node: node.total_duration_ms, reverse=True)
    for i, root in enumerate(roots):
        if i > 0:
            print(f"\n{'- ' * (_REPORT_WIDTH // 2)}")
        heading = (
            f"{_operator_display_name(root)}{format_node_args(root)} "
            f"{format_node_stats(root)}"
        )
        print()
        print_wrapped_tree_line("", heading)
        for variant_line in format_args_variant_lines(root):
            print_wrapped_tree_line("", variant_line)
        children = fold_identical_sibling_subtrees(root.children)
        for child_index, child in enumerate(children):
            child_is_last = (child_index == len(children) - 1) and not root.kernels
            print_operator_node(child, kernel_ids, is_last=child_is_last)
        _print_node_kernels(root, "", kernel_ids)


def show_operator_summary(summary_df: pd.DataFrame) -> None:
    """Print a flat per-operator summary table alongside the call tree.

    - Rendered as a fancy_grid bordered table via tabulate, matching the
      rest of the analyze CLI.

    - Operator column wraps long paths; numeric columns size to content.

    - A header line above the table explains the aggregation so column
      names can stay short.

    - Time cells are formatted per-cell via format_duration (auto-switching
      between ms and us). NaN renders as "N/A".
    """
    if summary_df is None or summary_df.empty:
        print("\nOperator summary: (no operators with recorded dispatches)")
        return

    operator_name_wrap_width = _OPERATOR_SUMMARY_NAME_WRAP_WIDTH

    # (DataFrame column, display label) pairs. DataFrame names match the
    # schema produced by build_operator_summary; display labels are short
    # because the header line below explains the semantics and time cells
    # self-label their unit.
    column_map = [
        ("Operator", "Operator"),
        ("Calls", "Calls"),
        ("Dispatches", "Dispatches"),
        ("Total_GPU", "Total"),
        ("Pct_Total_GPU", "% Total"),
        ("Mean_Per_Call", "Mean/Call"),
        ("Mean_Per_Dispatch", "Mean"),
        ("Min_Dispatch", "Min"),
        ("Max_Dispatch", "Max"),
    ]
    source_columns = [source for source, _ in column_map]
    headers = [h for _, h in column_map]
    time_columns = (
        "Total_GPU",
        "Mean_Per_Call",
        "Mean_Per_Dispatch",
        "Min_Dispatch",
        "Max_Dispatch",
    )

    display_df = summary_df[source_columns].copy()
    display_df["Operator"] = (
        display_df["Operator"]
        .astype(str)
        .apply(lambda s: textwrap.fill(s, width=operator_name_wrap_width))
    )
    for column in time_columns:
        display_df[column] = display_df[column].apply(format_duration)

    # Time columns are pre-formatted strings; only % Total still needs floatfmt.
    floatfmt = ("", "", "", "", ".2f", "", "", "", "")
    colalign = ("left",) + ("right",) * (len(headers) - 1)

    print(
        "\nOperator summary (Min/Max/Mean are per-dispatch over the subtree; "
        "sorted by Total):"
    )
    print(
        tabulate(
            display_df.values,
            headers=headers,
            tablefmt="fancy_grid",
            floatfmt=floatfmt,
            colalign=colalign,
            missingval="N/A",
        )
    )


def print_operator_node(
    node: CallTreeNode,
    kernel_ids: Dict[str, int],
    is_last: bool = True,
    parent_pipes: str = "",
) -> None:
    """Print a node and its descendants, folding identical sibling subtrees."""
    indent = parent_pipes
    is_branching = len(node.children) + len(node.kernels) > 1

    branch_char = "└─ " if is_last else "├─ "
    node_prefix = f"{indent}{branch_char}"

    if is_last:
        new_parent_pipes = parent_pipes + "   "
    else:
        new_parent_pipes = parent_pipes + "|  "

    args_segment = format_node_args(node)
    if is_branching:
        print_wrapped_tree_line(
            node_prefix,
            f"{_operator_display_name(node)}{args_segment} {format_node_stats(node)}",
            continuation_prefix=new_parent_pipes,
        )
    else:
        if len(node.invocation_ids) > 0:
            suffix = f" (calls: {node.call_count})"
        else:
            suffix = ""
        print_wrapped_tree_line(
            node_prefix,
            f"{_operator_display_name(node)}{args_segment}{suffix}",
            continuation_prefix=new_parent_pipes,
        )

    for variant_line in format_args_variant_lines(node):
        print_wrapped_tree_line(
            new_parent_pipes, variant_line, continuation_prefix=new_parent_pipes
        )

    children = fold_identical_sibling_subtrees(node.children)
    for i, child in enumerate(children):
        child_is_last = (i == len(children) - 1) and (len(node.kernels) == 0)
        print_operator_node(
            child, kernel_ids, is_last=child_is_last, parent_pipes=new_parent_pipes
        )

    _print_node_kernels(node, new_parent_pipes, kernel_ids)


def _operator_listing_view(
    call_trees: CallTreeForest, backends: List[str]
) -> CallTreeForest:
    """Return the operators of the listed backends with the ancestors above them.

    With no backends given, the whole forest is listed.
    """
    if not backends:
        return call_trees
    matched_nodes = [
        node
        for roots in call_trees.values()
        for node, _path in iter_call_tree_paths(roots)
        if node.backend in backends
    ]
    return copy_matched_operator_subtree(call_trees, matched_nodes)


def _operator_display_name(node: CallTreeNode) -> str:
    """Operator name with file:line when file_name is set."""
    if not node.file_name:
        return node.name
    if node.line_number is None:
        return f"{node.name} {node.file_name}"
    return f"{node.name} {node.file_name}:{node.line_number}"


def _print_node_kernels(
    node: CallTreeNode, parent_pipes: str, kernel_ids: Dict[str, int]
) -> None:
    """Print this node's GPU dispatches, longest total duration first."""
    for i, (kernel_name, kernel_stats) in enumerate(
        sorted(
            node.kernels.items(),
            key=lambda kv: kv[1].total_duration_ns,
            reverse=True,
        )
    ):
        launches = kernel_stats.launches
        duration_ns = kernel_stats.total_duration_ns
        kernel_id = kernel_ids.get(str(kernel_name).strip())
        id_suffix = f" (id {kernel_id})" if kernel_id is not None else ""
        display_name = simplify_kernel_name(kernel_name)
        total_ms = duration_ns * NS_TO_MS
        stats = f"(dispatches: {launches}, total: {format_duration(total_ms)})"

        kernel_is_last = i == len(node.kernels) - 1
        kernel_branch_char = "└─ " if kernel_is_last else "├─ "
        kernel_prefix = f"{parent_pipes}{kernel_branch_char}"

        print_wrapped_kernel_line(
            kernel_prefix,
            display_name,
            f"{id_suffix} {stats}".strip(),
            continuation_prefix=parent_pipes,
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
