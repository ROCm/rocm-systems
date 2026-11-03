# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload integration and diagnostics for ML API trace analysis."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ml_api_tracing import call_tree
from ml_api_tracing.call_tree import (
    nest_marker_intervals,
    prune_cpu_only_call_trees,
    rollup_node_stats,
    validate_all_markers_nested,
)
from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS
from ml_api_tracing.errors import MlApiTraceError, UnaccountedKernelError
from ml_api_tracing.input import (
    load_counter_trace_dataframe,
    load_ml_api_trace_pairs,
)
from ml_api_tracing.models import CallTreeForest
from ml_api_tracing.processing import (
    apply_parsed_function_columns,
    collapse_matching_markers_across_passes,
    group_marker_pass,
    join_marker_counter_pass,
    unmatched_kernel_rows,
)
from ml_api_tracing.selection import (
    analyze_dispatch_ids_from_pmc_keys,
    assign_kernel_ids_from_top,
    intersect_dispatch_ids,
    intersect_dispatch_ids_by_kernel,
    select_operators,
)
from utils import file_io, schema
from utils.logger import console_error, console_log, console_warning, demarcate

# Maps each ML API trace backend to its analyze CLI attributes and display label.
_ML_API_ANALYSIS_CLI_OPTIONS = {
    "torch": {
        "filter_attr": "torch_operator",
        "list_attr": "list_torch_operators",
        "label": "PyTorch",
    },
    "triton": {
        "filter_attr": "triton_operator",
        "list_attr": "list_triton_operators",
        "label": "Triton",
    },
}


@demarcate
def process_ml_api_trace_output(
    workload: schema.Workload,
    workload_dir: str,
) -> None:
    """Load ML API traces and update each Workload stage as it succeeds.

    Integrity failures abort analyze. Coverage issues warn and processing
    continues, preserving state completed before any failure.
    """
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    try:
        pairs = load_ml_api_trace_pairs(Path(workload_dir))
        if not pairs:
            console_warning(
                "No marker files with corresponding counter files found. "
                "Ensure profiling was done with ML API tracing enabled "
                "(e.g., via '--torch-trace')."
            )
            workload.ml_api_trace_pairs = []
            return
        workload.ml_api_trace_pairs = pairs
        _build_ml_api_call_trees(workload)
    except MlApiTraceError as error:
        console_error("analysis", str(error))


def parse_operator_patterns(
    args: argparse.Namespace, backends: List[str]
) -> Optional[Dict[str, List[str]]]:
    """Parse operator glob patterns from args for each backend in backends.

    Returns a {backend: patterns} dict, or None when no filter flag is
    set for any backend. An empty flag value maps to ["**"].
    """
    framewise_patterns: Dict[str, List[str]] = {}
    for backend in backends:
        raw = getattr(args, _ML_API_ANALYSIS_CLI_OPTIONS[backend]["filter_attr"], None)
        if raw is None:
            continue
        pattern_list: List[str] = []
        for operator_arg in raw:
            pattern_list.extend(
                pattern.strip()
                for pattern in str(operator_arg).split(",")
                if pattern.strip()
            )
        if not pattern_list:
            pattern_list = ["**"]
        framewise_patterns[backend] = pattern_list
    if not framewise_patterns:
        return None
    return framewise_patterns


def active_operator_backends(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    """Return filtering and listing backends in the canonical backend order."""
    filters = [
        backend
        for backend in KNOWN_ML_API_BACKENDS
        if getattr(args, _ML_API_ANALYSIS_CLI_OPTIONS[backend]["filter_attr"], None)
        is not None
    ]
    listings = [
        backend
        for backend in KNOWN_ML_API_BACKENDS
        if getattr(args, _ML_API_ANALYSIS_CLI_OPTIONS[backend]["list_attr"], False)
    ]
    return filters, listings


def validate_operator_trace_options(
    args: argparse.Namespace, profiling_config: Dict[str, Any]
) -> None:
    """Check trace prerequisites in backend order at the existing sanitize stage."""
    ml_api_trace = profiling_config.get("ml_api_trace", False)
    for backend in KNOWN_ML_API_BACKENDS:
        needs_trace = getattr(args, f"{backend}_operator", None) is not None or getattr(
            args, f"list_{backend}_operators", False
        )
        if needs_trace and not (
            profiling_config.get(f"{backend}_trace", False) or ml_api_trace
        ):
            console_error(
                "ml api trace",
                f'Workload was not profiled with "--{backend}-trace" or '
                '"--ml-api-trace". '
                f"Cannot use --{backend}-operator or "
                f"--list-{backend}-operators.",
            )


def operator_backend_labels(backends: List[str]) -> List[str]:
    """Translate backend names to their canonical CLI diagnostic/report labels."""
    return [_ML_API_ANALYSIS_CLI_OPTIONS[backend]["label"] for backend in backends]


def apply_operator_filter(
    args: argparse.Namespace, workload: schema.Workload, backends: List[str]
) -> List[Tuple]:
    """Publish the owned match before the no-dispatch fatal, preserving stages."""
    framewise_patterns = parse_operator_patterns(args, backends)
    if framewise_patterns is None:
        return []
    labels = ", ".join(operator_backend_labels(backends))
    selected_forest, pmc_group_keys = select_operators(
        workload.ml_api_call_trees, framewise_patterns
    )
    if not selected_forest:
        all_patterns = ", ".join(
            pattern for patterns in framewise_patterns.values() for pattern in patterns
        )
        console_error(
            "ml api trace",
            f"No {labels} operators matched the pattern(s): {all_patterns}",
        )
    workload.ml_api_call_trees = selected_forest
    if not pmc_group_keys:
        console_error(
            "ml api trace", f"No {labels} kernel dispatches on the matched operators"
        )
    return pmc_group_keys


def select_operator_dispatches(
    workload: schema.Workload,
    pmc_group_keys: List[Tuple],
    slot_to_id: Dict[Tuple, int],
    backends: List[str],
) -> None:
    """Publish selected dispatch IDs only after mapping and restriction succeed."""
    labels = ", ".join(operator_backend_labels(backends))
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
    """Publish sliced numbering and kernel-restricted dispatches at prior stages."""
    labels = ", ".join(operator_backend_labels(backends))
    name_to_id = {
        str(kernel_name).strip(): idx
        for idx, kernel_name in enumerate(kernel_top_df["Kernel_Name"].tolist())
    }
    for roots in workload.ml_api_call_trees.values():
        for root in roots:
            assign_kernel_ids_from_top(root, name_to_id)
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


def attach_unlocated_trees_by_launcher_thread(
    forest: CallTreeForest,
    errors: Optional[List[MlApiTraceError]] = None,
) -> None:
    """Attach worker roots with the adapter's warning policy.

    An explicit collector suppresses warnings. Otherwise, report collected
    issues before returning or propagating an error from attachment.
    """
    issues = errors if errors is not None else []
    try:
        call_tree.attach_unlocated_trees_by_launcher_thread(forest, issues)
    finally:
        if errors is None:
            for issue in issues:
                console_warning("analysis", str(issue))


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


def _build_ml_api_call_trees(workload: schema.Workload) -> None:
    """Advance joined frames, parsed markers and the forest in their order."""
    unmatched_frames: List[pd.DataFrame] = []
    for pair in workload.ml_api_trace_pairs:
        counters = load_counter_trace_dataframe(pair.counter_path)
        pair.joined_df = join_marker_counter_pass(pair.marker_df, counters)
        unmatched_frames.append(unmatched_kernel_rows(pair.joined_df))
    workload.unmatched_kernel_frames = unmatched_frames
    nonempty_unmatched = [frame for frame in unmatched_frames if not frame.empty]
    if nonempty_unmatched:
        console_warning(
            "analysis",
            str(
                UnaccountedKernelError(pd.concat(nonempty_unmatched, ignore_index=True))
            ),
        )
    for pair in workload.ml_api_trace_pairs:
        pair.joined_df = group_marker_pass(pair.joined_df)
    skipped_keys: Set[Tuple[str, str]] = set()
    workload.ml_api_trace_df = apply_parsed_function_columns(
        collapse_matching_markers_across_passes([
            pair.joined_df for pair in workload.ml_api_trace_pairs
        ])
    )
    workload.ml_api_call_trees = nest_marker_intervals(
        workload.ml_api_trace_df, skipped_keys=skipped_keys
    )
    attach_unlocated_trees_by_launcher_thread(workload.ml_api_call_trees)
    validate_all_markers_nested(
        workload.ml_api_trace_df,
        workload.ml_api_call_trees,
        skipped_keys=skipped_keys,
    )
    prune_cpu_only_call_trees(workload.ml_api_call_trees)
    for roots in workload.ml_api_call_trees.values():
        for node in roots:
            rollup_node_stats(node)
    _record_missing_source_location_errors(workload.ml_api_call_trees)


def _record_missing_source_location_errors(
    forest: CallTreeForest,
    errors: Optional[List[MlApiTraceError]] = None,
) -> None:
    """Report missing locations after rollup, or append to an explicit collector.

    Without a collector, report all collected issues even if detection fails.
    """
    issues = errors if errors is not None else []
    try:
        call_tree.record_missing_source_location_errors(forest, issues)
    finally:
        if errors is None:
            for issue in issues:
                console_warning("analysis", str(issue))
