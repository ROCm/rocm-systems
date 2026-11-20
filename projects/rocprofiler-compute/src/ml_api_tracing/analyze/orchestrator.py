# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run the ML API trace analysis steps in order for one workload."""

from pathlib import Path
from typing import Callable, List

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
from ml_api_tracing.analyze.pass_matching import (
    apply_parsed_function_columns,
    collapse_matching_markers_across_passes,
    group_marker_pass,
    join_marker_counter_pass,
    unmatched_kernel_rows,
)
from ml_api_tracing.analyze.trace_csv import (
    load_counter_trace_dataframe,
    load_ml_api_trace_pairs,
)
from utils import schema
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
