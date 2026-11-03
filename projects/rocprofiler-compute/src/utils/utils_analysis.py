# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml_api_tracing.call_tree import iter_call_tree_paths, rollup_node_stats
from ml_api_tracing.marker_format import (
    split_operator_args,
)
from ml_api_tracing.models import CallTreeForest, CallTreeNode, KernelStats
from utils import csv_compression
from utils.logger import (
    console_debug,
    console_error,
    console_warning,
)
from utils.utils_common import (
    NATIVE_AGENTS_PREFIX,
    NATIVE_COUNTERS_PREFIX,
    NATIVE_DISPATCH_PREFIX,
    NATIVE_KERNEL_SYMBOLS_PREFIX,
    NS_TO_MS,
)
from utils.utils_counter_defs import UNIT_COUNTER

# Canonical column-name preference order for Percent of Peak lookups
VALUE_COL_PREFERENCE: tuple[str, ...] = ("Avg", "Value")
PEAK_COL_PREFERENCE: tuple[str, ...] = ("Peak", "Peak (Empirical)")

# The long counter frame both lanes hand to process_rocpd_csv, in the column
# order of results_*.csv.
COUNTER_RESULTS_COLUMNS: List[str] = [
    "GPU_ID",
    "GUID",
    "Correlation_Id",
    "Dispatch_ID",
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
    "Counter_Name",
    "Counter_Value",
]

_NATIVE_ARTIFACT_RE = re.compile(
    rf"^(?P<kind>{NATIVE_COUNTERS_PREFIX}|{NATIVE_DISPATCH_PREFIX}"
    rf"|{NATIVE_KERNEL_SYMBOLS_PREFIX}|{NATIVE_AGENTS_PREFIX})"
    rf"_(?P<fbase>.+)_(?P<pid>\d+)\.csv\.gz$"
)

# Native CSV column names mapped to their results_*.csv names.
_NATIVE_TO_RESULTS_COLUMNS = {
    "gpu_id": "GPU_ID",
    "correlation_id": "Correlation_Id",
    "grid_size": "Grid_Size",
    "workgroup_size": "Workgroup_Size",
    "lds_per_workgroup": "LDS_Per_Workgroup",
    "scratch_per_workitem": "Scratch_Per_Workitem",
    "arch_vgpr": "Arch_VGPR",
    "accum_vgpr": "Accum_VGPR",
    "sgpr": "SGPR",
    "kernel_name": "Kernel_Name",
    "start_timestamp": "Start_Timestamp",
    "end_timestamp": "End_Timestamp",
    "counter_name": "Counter_Name",
    "counter_value": "Counter_Value",
}

# The keys profile mode numbers results_*.csv dispatches and kernels by.
_DISPATCH_ID_KEYS = [
    "PID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Start_Timestamp",
    "End_Timestamp",
]
_KERNEL_ID_KEYS = ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"]
ARGS_DISPLAY_MAX_ITEMS = 8
ARGS_DISPLAY_MAX_CHARS = 160
ARGS_DISPLAY_MAX_VARIANTS = 5


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


@dataclass(frozen=True)
class NativeArtifacts:
    """The four CSVs one process wrote for one counter set."""

    fbase: str
    pid: int
    counters: Path
    dispatch: Path
    kernel_symbols: Path
    agents: Path


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
    destination: dict[str, KernelStats], name: str, stats: KernelStats
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
    mins = [
        duration
        for duration in (existing.min_duration_ns, stats.min_duration_ns)
        if duration is not None
    ]
    maxes = [
        duration
        for duration in (existing.max_duration_ns, stats.max_duration_ns)
        if duration is not None
    ]
    existing.launches += stats.launches
    existing.total_duration_ns += stats.total_duration_ns
    existing.min_duration_ns = min(mins) if mins else None
    existing.max_duration_ns = max(maxes) if maxes else None
    if existing.kernel_id is None:
        existing.kernel_id = stats.kernel_id
    existing.pmc_group_keys.extend(stats.pmc_group_keys)


def _merge_args_invocations(
    destination: dict[str, set[str]], source: dict[str, set[str]]
) -> None:
    for args_blob, recorded_invocation_ids in source.items():
        destination.setdefault(args_blob, set()).update(recorded_invocation_ids)


def _merge_identical_sibling_group(group: list[CallTreeNode]) -> CallTreeNode:
    first = group[0]
    kernels: dict[str, KernelStats] = {}
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


def fold_identical_sibling_subtrees(
    nodes: list[CallTreeNode],
) -> list[CallTreeNode]:
    """Merge siblings that share operator, kernel names, and child shape.

    Returns new nodes sorted by total GPU duration. Does not mutate nodes.
    """
    if not nodes:
        return []
    groups: dict[tuple, list[CallTreeNode]] = {}
    for node in nodes:
        groups.setdefault(_subtree_shape_key(node), []).append(node)
    folded = [_merge_identical_sibling_group(group) for group in groups.values()]
    folded.sort(key=lambda node: node.total_duration_ms, reverse=True)
    return folded


def filter_forest_by_backends(
    forest: CallTreeForest,
    backends: list[str],
) -> CallTreeForest:
    """Copy a forest, keeping nodes for the requested backends and their ancestors.

    An empty backends list keeps every node, including user. Ancestors kept
    only as scaffolding have their own kernels cleared, so a torch parent
    of a triton child does not contribute GPU time or kernel lines to a
    triton listing. Inclusive stats on each copy are recomputed from the
    kernels and children that remain.
    """
    if not backends:
        return forest

    filtered: CallTreeForest = {}
    for thread_id, roots in forest.items():
        kept_roots: list[CallTreeNode] = []
        for root in roots:
            copied_root = _copy_backend_view(root, backends)
            if copied_root is not None:
                kept_roots.append(copied_root)
        if kept_roots:
            filtered[thread_id] = kept_roots
    return filtered


def _copy_backend_view(
    node: CallTreeNode, backends: List[str]
) -> Optional[CallTreeNode]:
    """Copy one node for filter_forest_by_backends, or None when nothing is kept."""
    kept_children: list[CallTreeNode] = []
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
    matched_node_ids: set[int],
    inherited_backends: set[str],
) -> Optional[CallTreeNode]:
    """Copy one node, keeping match kernels and in-scope descendant kernels."""
    node_is_match = id(node) in matched_node_ids
    descendant_backends = inherited_backends
    if node_is_match and node.backend is not None:
        descendant_backends = set(inherited_backends)
        descendant_backends.add(node.backend)
    kept_children: list[CallTreeNode] = []
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


def copy_matched_operator_subtree(
    forest: CallTreeForest,
    matched_nodes: list[CallTreeNode],
) -> CallTreeForest:
    """Copy matches, in-scope descendants, and ancestors needed to reach them."""
    matched_node_ids = {id(node) for node in matched_nodes}
    subtree: CallTreeForest = {}
    for thread_id, roots in forest.items():
        kept_roots: list[CallTreeNode] = []
        for root in roots:
            copied_root = _copy_scoped_operator_node(root, matched_node_ids, set())
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
    return isinstance(value, float) and pd.isna(value)


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

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(
        by=["Total_GPU", "Operator"],
        ascending=[False, True],
        ignore_index=True,
    )


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


def validate_workload(path: str) -> None:
    """Validate workload directory contains readable, non-empty profiling output."""
    workload_dir = Path(path)
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


def process_rocpd_csv(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple, int]]:
    """Merge counters across unique dispatches into one row each.

    The dict maps each groupby key to the new 1..N Dispatch_ID, in the
    same order as those ids are assigned.
    """
    if df.empty:
        return df, {}

    data: list[dict[str, Any]] = []
    slot_to_id: dict[tuple, int] = {}

    # Group by unique kernel and merge into a single row
    for group_key, group_df in df.groupby([
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
        slot_to_id[group_key] = len(data)
    df = pd.DataFrame(data)
    # Reset dispatch IDs
    df["Dispatch_ID"] = range(1, len(df) + 1)
    return df, slot_to_id


def find_native_artifacts(workload_dir: Path) -> List[NativeArtifacts]:
    """Return the complete per-pid native artifact sets in workload_dir.

    Sorted by counter set then pid, since ids are numbered in read order. An
    incomplete set cannot be joined, so it is skipped.
    """
    found: Dict[Tuple[str, int], Dict[str, Path]] = {}
    for path in workload_dir.glob(f"*.csv{csv_compression.GZIP_SUFFIX}"):
        match = _NATIVE_ARTIFACT_RE.match(path.name)
        if match is None:
            continue
        key = (match["fbase"], int(match["pid"]))
        found.setdefault(key, {})[match["kind"]] = path

    kinds = {
        NATIVE_COUNTERS_PREFIX,
        NATIVE_DISPATCH_PREFIX,
        NATIVE_KERNEL_SYMBOLS_PREFIX,
        NATIVE_AGENTS_PREFIX,
    }
    artifacts = []
    for (fbase, pid), paths in sorted(found.items()):
        if not kinds <= set(paths):
            continue
        artifacts.append(
            NativeArtifacts(
                fbase=fbase,
                pid=pid,
                counters=paths[NATIVE_COUNTERS_PREFIX],
                dispatch=paths[NATIVE_DISPATCH_PREFIX],
                kernel_symbols=paths[NATIVE_KERNEL_SYMBOLS_PREFIX],
                agents=paths[NATIVE_AGENTS_PREFIX],
            )
        )
    return artifacts


def number_gpus_by_node(agents: pd.DataFrame) -> Dict[int, int]:
    """Map every GPU's node id to a GPU id from 0, in node order."""
    node_ids = sorted({int(node_id) for node_id in agents["node_id"]})
    return {node_id: gpu_id for gpu_id, node_id in enumerate(node_ids)}


def join_native_counters(
    counters: pd.DataFrame,
    dispatches: pd.DataFrame,
    symbols: pd.DataFrame,
    pid: int,
) -> pd.DataFrame:
    """Join one process's native counters to its dispatches and kernel symbols.

    counters holds one total per dispatch and counter. A row whose dispatch or
    kernel symbol is missing is dropped. The pid fills GUID, since the native
    tool has no session uuid and GUID only tells processes apart.
    """
    joined = counters.merge(dispatches, on="dispatch_id", how="inner").merge(
        symbols, on="kernel_id", how="inner"
    )
    return joined.rename(columns=_NATIVE_TO_RESULTS_COLUMNS).assign(GUID=pid, PID=pid)


def assign_native_ids(counter_set: pd.DataFrame) -> pd.DataFrame:
    """Number one counter set's dispatches from 1 and its kernels from 0.

    Rows are sorted by pid, then dispatch_id, so ids follow the order kernels
    were launched, not the order they finished.
    """
    launch_order = counter_set.sort_values(["PID", "dispatch_id"], kind="stable")
    dispatch_groups = launch_order.groupby(_DISPATCH_ID_KEYS, sort=False, dropna=False)
    kernel_groups = launch_order.groupby(_KERNEL_ID_KEYS, sort=False, dropna=False)
    numbered = launch_order.assign(
        Dispatch_ID=dispatch_groups.ngroup() + 1,
        Kernel_ID=kernel_groups.ngroup(),
    )
    return numbered.loc[:, COUNTER_RESULTS_COLUMNS]
