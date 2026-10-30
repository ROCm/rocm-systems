# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Workload integration and diagnostics for ML API trace analysis."""

from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd

from ml_api_tracing import call_tree
from ml_api_tracing.call_tree import (
    nest_marker_intervals,
    prune_cpu_only_call_trees,
    rollup_node_stats,
    validate_all_markers_nested,
)
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
from utils import schema
from utils.logger import console_error, console_log, console_warning, demarcate


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
