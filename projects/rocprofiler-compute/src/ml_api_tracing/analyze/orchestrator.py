# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run the ML API trace analysis steps in order for one workload."""

import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from ml_api_tracing.analyze.call_tree import (
    attach_unlocated_trees_by_launcher_thread,
    nest_marker_intervals,
    prune_cpu_only_call_trees,
    record_missing_source_location_errors,
    rollup_node_stats,
    validate_all_markers_nested,
)
from ml_api_tracing.analyze.definitions import CallTreeForest, MlApiTracePair
from ml_api_tracing.analyze.errors import (
    MlApiTraceError,
    UnaccountedKernelError,
)
from ml_api_tracing.analyze.operator_patterns import (
    bind_operator_patterns,
    check_backends_traced,
    operator_backend_labels,
    traced_frameworks_from_config,
)
from ml_api_tracing.analyze.pass_matching import (
    apply_parsed_function_columns,
    collapse_matching_markers_across_passes,
    group_marker_pass,
    join_marker_counter_pass,
    unmatched_kernel_rows,
)
from ml_api_tracing.analyze.selection import (
    analyze_dispatch_ids_from_pmc_keys,
    intersect_dispatch_ids,
    intersect_dispatch_ids_by_kernel,
    kernel_ids_from_top,
    select_operators,
)
from ml_api_tracing.analyze.trace_csv import (
    load_counter_trace_dataframe,
    load_ml_api_trace_pairs,
)
from utils import file_io, schema
from utils.logger import console_error, console_log, console_warning, demarcate


@demarcate
def process_ml_api_trace_output(
    workload: schema.Workload,
    workload_dir: str,
) -> None:
    """Populate a Workload with the parsed marker rows and the call trees.

    Integrity failures abort analyze; coverage issues are reported as warnings.
    """
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    try:
        pairs = load_ml_api_trace_pairs(Path(workload_dir))
        if not pairs:
            console_warning(
                "No marker files with corresponding counter files found. "
                "Ensure profiling was done with ML API tracing enabled "
                "(e.g., via '--trace-ml-api')."
            )
            return
        _build_ml_api_call_trees(workload, pairs)
    except MlApiTraceError as error:
        console_error("analysis", str(error))


def _build_ml_api_call_trees(
    workload: schema.Workload, pairs: List[MlApiTracePair]
) -> None:
    """Match the passes, nest the markers and roll up the kernel stats."""
    joined_passes = [
        join_marker_counter_pass(
            pair.marker_df, load_counter_trace_dataframe(pair.counter_path)
        )
        for pair in pairs
    ]
    _warn_unmatched_kernels(joined_passes)
    workload.ml_api_trace_df = apply_parsed_function_columns(
        collapse_matching_markers_across_passes([
            group_marker_pass(joined) for joined in joined_passes
        ])
    )
    forest = nest_marker_intervals(workload.ml_api_trace_df)
    workload.ml_api_call_trees = forest
    _warn_about(forest, attach_unlocated_trees_by_launcher_thread)
    validate_all_markers_nested(workload.ml_api_trace_df, forest)
    prune_cpu_only_call_trees(forest)
    for roots in forest.values():
        for node in roots:
            rollup_node_stats(node)
    _warn_about(forest, record_missing_source_location_errors)


def _warn_unmatched_kernels(joined_passes: List[pd.DataFrame]) -> None:
    """Warn once about dispatches with no marker in any pass."""
    unmatched = [unmatched_kernel_rows(joined) for joined in joined_passes]
    unmatched = [rows for rows in unmatched if not rows.empty]
    if not unmatched:
        return
    rows = pd.concat(unmatched, ignore_index=True)
    names = list(dict.fromkeys(str(name) for name in rows["Kernel_Name"]))
    console_warning("analysis", str(UnaccountedKernelError(len(rows), names)))


def _warn_about(
    forest: CallTreeForest,
    collect_issues: Callable[[CallTreeForest, List[MlApiTraceError]], None],
) -> None:
    """Run a call_tree step that collects coverage issues, then warn about them."""
    issues: List[MlApiTraceError] = []
    collect_issues(forest, issues)
    for issue in issues:
        console_warning("analysis", str(issue))


def validate_operator_trace_options(
    workload_path: str,
    list_operators: Optional[List[str]],
    operator_patterns: Optional[List[str]],
    profiling_config: Dict[str, Any],
) -> Optional[Dict[str, List[str]]]:
    """Bind raw operator patterns to this workload's recorded frameworks.

    Listing takes precedence, including a bare list-all request. Errors name
    the workload path.
    """
    if list_operators is None and operator_patterns is None:
        return None
    traced_frameworks = traced_frameworks_from_config(profiling_config)
    try:
        if list_operators is not None:
            check_backends_traced(list_operators, traced_frameworks)
            return None
        if operator_patterns is not None:
            return bind_operator_patterns(operator_patterns, traced_frameworks)
    except ValueError as error:
        console_error("ml api trace", f"{workload_path}: {error}")
    return None


def apply_operator_filter(workload: schema.Workload, workload_path: str) -> List[Tuple]:
    """Keep only the operators that match workload.operator_patterns.

    Exits with an error when nothing matches or the matches launched no
    dispatches. Returns the PMC keys in their recorded order, each once.
    """
    patterns_by_backend = workload.operator_patterns
    labels = operator_backend_labels(list(patterns_by_backend))
    selected_forest, pmc_group_keys = select_operators(
        workload.ml_api_call_trees, patterns_by_backend
    )
    if not selected_forest:
        pattern_text = ", ".join(
            dict.fromkeys(
                pattern
                for patterns in patterns_by_backend.values()
                for pattern in patterns
            )
        )
        console_error(
            "ml api trace",
            f"{workload_path}: No {labels} operators matched the "
            f"pattern(s): {pattern_text}",
        )
    workload.ml_api_call_trees = selected_forest
    if not pmc_group_keys:
        console_error(
            "ml api trace",
            f"{workload_path}: No {labels} kernel dispatches on the matched operators",
        )
    return pmc_group_keys


def select_operator_dispatches(
    workload: schema.Workload,
    pmc_group_keys: List[Tuple],
    slot_to_id: Dict[Tuple, int],
    backends: List[str],
) -> None:
    """Select the analyze dispatch IDs of the matched operators, within -d."""
    labels = operator_backend_labels(backends)
    try:
        operator_dispatch_ids = analyze_dispatch_ids_from_pmc_keys(
            pmc_group_keys, slot_to_id
        )
    except KeyError as error:
        console_error(
            "ml api trace",
            f"No {labels} analyze dispatch id for pmc group key {error.args[0]}",
        )
        return
    user_dispatch_filter = workload.filter_dispatch_ids
    requested_dispatch_ids, minimum_dispatch_id = _parse_operator_dispatch_filter(
        user_dispatch_filter
    )
    selected_ids = intersect_dispatch_ids(
        operator_dispatch_ids, requested_dispatch_ids, minimum_dispatch_id
    )
    if not selected_ids:
        console_error(
            "ml api trace",
            f"No {labels} dispatches matched the -d filter: {user_dispatch_filter}",
        )
    workload.filter_dispatch_ids = selected_ids
    console_log(
        "ml api trace",
        f"{labels} operator filter selected "
        f"{len(workload.filter_dispatch_ids)} dispatch(es) for metric analysis.",
    )


def apply_sliced_kernel_top(
    workload: schema.Workload,
    kernel_top_df: pd.DataFrame,
    args: argparse.Namespace,
    backends: List[str],
) -> None:
    """Record the kernel ids from the top-kernel table for the operator report.

    If -k is present, restrict workload.filter_dispatch_ids to those kernels.
    """
    labels = operator_backend_labels(backends)
    workload.operator_kernel_ids = kernel_ids_from_top(kernel_top_df)
    if not workload.filter_kernel_ids:
        return
    file_io.validate_kernel_filter_ids(
        workload.filter_kernel_ids, len(kernel_top_df["Kernel_Name"])
    )
    workload.filter_dispatch_ids = intersect_dispatch_ids_by_kernel(
        workload.filter_dispatch_ids,
        workload.raw_pmc,
        kernel_top_df,
        workload.filter_kernel_ids,
    )
    if not workload.filter_dispatch_ids:
        console_error(
            "ml api trace",
            f"No {labels} operators matched the -k filter: {args.gpu_kernel}",
        )


def _parse_operator_dispatch_filter(
    dispatch_filter: Optional[List[Union[str, int]]],
) -> Tuple[Optional[List[int]], Optional[int]]:
    """Interpret the existing explicit-ID or greater-than dispatch grammar."""
    if not dispatch_filter:
        return None, None
    first_filter = str(dispatch_filter[0]).strip()
    if first_filter.startswith(">"):
        return None, int(first_filter[1:].strip())
    return [int(value) for value in dispatch_filter], None
