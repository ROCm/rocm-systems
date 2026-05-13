# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from utils.logger import (
    console_debug,
    console_error,
    console_log,
    console_warning,
    demarcate,
)

NS_TO_MS = 1.0 / 1_000_000.0


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

    Local to this frame:
      - invocation_ids: distinct Context_Id prefixes at this frame's depth.
      - call_count: derived as len(invocation_ids); see the property below.

    Inclusive over this node plus all descendants:
      - kernel_launches: kernel dispatches in the subtree.
      - total_duration_ms: cumulative GPU time in the subtree.
      - min_dispatch_ns / max_dispatch_ns / mean_dispatch_ns: per-kernel-dispatch
        duration stats. None when no non-zero-duration dispatch is in the subtree.
    """

    name: str
    children: dict[str, "CallTreeNode"] = field(default_factory=dict)
    kernels: dict[str, KernelStats] = field(default_factory=dict)
    kernel_launches: int = 0
    total_duration_ms: float = 0.0
    invocation_ids: set[str] = field(default_factory=set)
    min_dispatch_ns: Optional[float] = None
    max_dispatch_ns: Optional[float] = None
    mean_dispatch_ns: Optional[float] = None

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


def parse_top_level_location(context_id: object) -> str:
    """Extract 'file:line' from the first entry of a Context_Id string.

    Context_Id format: ``10@main.py:60/#10@main.py:21/...``
    Returns ``main.py:60`` or ``unknown:0`` on failure.
    """
    if pd.isna(context_id) or not str(context_id).strip():
        return "unknown:0"
    first_entry = str(context_id).split("/")[0]
    if "@" not in first_entry:
        return "unknown:0"
    _, location = first_entry.split("@", 1)
    return location if ":" in location else "unknown:0"


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

    for child in node.children.values():
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


def build_call_trees(
    df: pd.DataFrame,
) -> dict[str, CallTreeNode]:
    """Build per-source-location call trees from a consolidated torch trace DataFrame.

    Returns a dict mapping ``file:line`` to a CallTreeNode root whose
    children form the full operator/kernel hierarchy.

    Each kernel entry is stored as a ``KernelStats`` record.
    Full kernel names are used; shortening is left to the display layer.
    """
    required = {"Operator_Name", "Kernel_Name"}
    if df.empty or not required.issubset(df.columns):
        return {}

    has_kernel_timestamps = (
        "Start_Timestamp_kernel" in df.columns and "End_Timestamp_kernel" in df.columns
    )
    has_context_id = "Context_Id" in df.columns
    has_kernel_id = "Kernel_ID" in df.columns

    deduplication_columns = ["Operator_Name", "Kernel_Name"]
    if has_kernel_timestamps:
        deduplication_columns.append("Start_Timestamp_kernel")
    if has_context_id:
        deduplication_columns.append("Context_Id")
    dispatches = df.drop_duplicates(subset=deduplication_columns)

    call_trees: dict[str, CallTreeNode] = {}

    for row in dispatches.itertuples(index=False):
        op_path = str(row.Operator_Name).strip()
        kernel_name = str(row.Kernel_Name).strip()
        if not op_path or not kernel_name:
            continue

        context_id = getattr(row, "Context_Id", None) if has_context_id else None
        location = parse_top_level_location(context_id)

        duration_ns = 0.0
        if has_kernel_timestamps:
            try:
                duration_ns = float(row.End_Timestamp_kernel) - float(
                    row.Start_Timestamp_kernel
                )
            except (ValueError, TypeError):
                pass

        if location not in call_trees:
            call_trees[location] = CallTreeNode(name=location)
        location_root = call_trees[location]

        op_segments = op_path.split("/")
        ctx_segments = (
            str(context_id).split("/")
            if has_context_id and context_id is not None and pd.notna(context_id)
            else []
        )

        current_node = location_root
        for i, path_segment in enumerate(op_segments):
            if path_segment not in current_node.children:
                current_node.children[path_segment] = CallTreeNode(name=path_segment)
            current_node = current_node.children[path_segment]
            if i < len(ctx_segments):
                current_node.invocation_ids.add("/".join(ctx_segments[: i + 1]))

        if kernel_name not in current_node.kernels:
            kernel_id = None
            kernel_id_value = getattr(row, "Kernel_ID", None) if has_kernel_id else None
            if pd.notna(kernel_id_value):
                kernel_id = int(kernel_id_value)
            current_node.kernels[kernel_name] = KernelStats(kernel_id=kernel_id)
        kstats = current_node.kernels[kernel_name]
        kstats.launches += 1
        kstats.total_duration_ns += duration_ns
        if duration_ns > 0:
            if kstats.min_duration_ns is None or duration_ns < kstats.min_duration_ns:
                kstats.min_duration_ns = duration_ns
            if kstats.max_duration_ns is None or duration_ns > kstats.max_duration_ns:
                kstats.max_duration_ns = duration_ns

    for location_root in call_trees.values():
        rollup_node_stats(location_root)

    return call_trees


def write_torch_trace_consolidated_csv(
    consolidated_df: pd.DataFrame,
    torch_trace_path: Path,
) -> None:
    """Write the consolidated torch trace DataFrame to ``consolidated.csv``."""
    output_file = torch_trace_path / "consolidated.csv"
    consolidated_df.sort_values("Operator_Name", ignore_index=True).to_csv(
        output_file, index=False
    )
    console_log(f"Saved consolidated trace to {output_file}")


def build_call_trees_with_kernel_ids(
    consolidated_df: pd.DataFrame,
    kernel_top_df: pd.DataFrame,
) -> dict[str, CallTreeNode]:
    """Attach Kernel_ID values and build call trees from consolidated trace rows."""
    kernel_name_to_id = {
        str(row["Kernel_Name"]).strip(): idx for idx, row in kernel_top_df.iterrows()
    }
    consolidated_with_ids = consolidated_df.copy()
    consolidated_with_ids["Kernel_ID"] = (
        consolidated_with_ids["Kernel_Name"].str.strip().map(kernel_name_to_id)
    )
    return build_call_trees(consolidated_with_ids)


def build_operator_summary(
    call_trees: dict[str, CallTreeNode],
) -> pd.DataFrame:
    """Build a one-row-per-operator summary table from the call trees.

    Each row describes one operator (e.g. aten::matmul) that ran at least
    one GPU kernel. All time values are in milliseconds.

    Columns:

    - Operator: full path of the operator (e.g. "aten::matmul/aten::mm").

    - Location: Python file:line where the outermost caller lives.

    - Calls: how many times this operator was invoked. NaN when the trace
      did not include Context_Id information to count invocations.

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

    Operators that ran no GPU kernels and the synthetic location-root nodes
    are skipped. Empty input returns an empty DataFrame with the full
    column list.

    Sorted by Total_GPU descending, then Operator and Location ascending.
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
        for child_name, child in node.children.items():
            full_path = path_parts + [child_name]
            if child.kernel_launches > 0:
                has_calls = len(child.invocation_ids) > 0
                calls = len(child.invocation_ids) if has_calls else float("nan")
                dispatches = child.kernel_launches
                total_gpu_ms = child.total_duration_ms
                rows.append({
                    "Operator": "/".join(full_path),
                    "Location": location,
                    "Calls": calls,
                    "Dispatches": dispatches,
                    "Dispatches_Per_Call": (
                        dispatches / calls if has_calls else float("nan")
                    ),
                    "Total_GPU": total_gpu_ms,
                    "Pct_Total_GPU": float("nan"),  # filled in if grand total > 0
                    "Mean_Per_Call": (
                        total_gpu_ms / calls if has_calls else float("nan")
                    ),
                    "Mean_Per_Dispatch": (
                        child.mean_dispatch_ns * NS_TO_MS
                        if child.mean_dispatch_ns is not None
                        else float("nan")
                    ),
                    "Min_Dispatch": (
                        child.min_dispatch_ns * NS_TO_MS
                        if child.min_dispatch_ns is not None
                        else float("nan")
                    ),
                    "Max_Dispatch": (
                        child.max_dispatch_ns * NS_TO_MS
                        if child.max_dispatch_ns is not None
                        else float("nan")
                    ),
                })
            walk(child, location, full_path)

    for location, root in call_trees.items():
        walk(root, location, [])

    if not rows:
        return pd.DataFrame(columns=columns)

    grand_total_ms = sum(root.total_duration_ms for root in call_trees.values())
    if grand_total_ms > 0:
        for r in rows:
            r["Pct_Total_GPU"] = 100.0 * r["Total_GPU"] / grand_total_ms

    df = pd.DataFrame(rows, columns=columns)
    return df.sort_values(
        by=["Total_GPU", "Operator", "Location"],
        ascending=[False, True, True],
        ignore_index=True,
    )


@demarcate
def process_torch_trace_output(
    workload_dir: str,
) -> tuple[pd.DataFrame, Path]:
    """
    Build consolidated torch trace rows and prepare output directory.

    - Performs inner join on Correlation_ID, filtering out unmatched entries
    - Consolidates data across passes and normalizes required columns
    - Prepares a clean workload_dir/torch_trace/ directory for output files

    Returns (consolidated_df, torch_trace_path) on success.
    """
    console_log(f"Looking for marker and counter csv files in {workload_dir}")
    marker_api_trace_csvs = list(
        Path(workload_dir).glob("**/torch_trace*_marker_api_trace.csv")
    )
    counter_collection_csvs = [
        markers_file.parent
        / markers_file.name.replace("_marker_api_trace.", "_counter_collection.")
        for markers_file in marker_api_trace_csvs
    ]
    existing_csv_files = [
        [marker_api_trace_csvs[i], counter_collection_csvs[i]]
        for i in range(len(marker_api_trace_csvs))
        if counter_collection_csvs[i].is_file() and marker_api_trace_csvs[i].is_file()
    ]

    if not existing_csv_files:
        console_warning(
            "No marker files with corresponding counter files found. "
            "Ensure profiling was done with '--torch-trace'."
        )
        return pd.DataFrame(), Path(f"{workload_dir}/torch_trace")

    torch_trace_path = Path(f"{workload_dir}/torch_trace")
    if torch_trace_path.exists():
        shutil.rmtree(torch_trace_path)
        console_log(f"Removed previous torch_trace directory: {torch_trace_path}")
    torch_trace_path.mkdir(parents=True, exist_ok=True)

    # Join marker and counter data
    def _merge_pair(
        marker_path: Path,
        counter_path: Path,
        join_keys: tuple[str, ...] = ("Correlation_ID",),
    ) -> pd.DataFrame:
        """Merge a pair of marker and counter csv files on specified keys,
        return the merged dataframe.
        """
        marker_df = pd.read_csv(marker_path)
        counter_df = pd.read_csv(counter_path)
        # Normalize column names to handle case inconsistencies
        marker_df.columns = marker_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )
        counter_df.columns = counter_df.columns.str.replace(
            "Correlation_Id", "Correlation_ID"
        )

        return pd.merge(
            marker_df,
            counter_df,
            on=join_keys,
            how="inner",
            suffixes=("_function", "_kernel"),
        )

    # If rocpd format, pairs are present in workload_dir, one pair per fbase
    # If csv format, pairs are present in workload/{fbase}/ one pair per process
    # Extracting the output_format used in profiling from the path of a marker file
    if Path(workload_dir).resolve() == existing_csv_files[0][0].parent.resolve():
        join_keys = ("Correlation_ID", "GUID")  # output_format "rocpd"
    else:
        join_keys = ("Correlation_ID",)  # output_format "csv"
    consolidated_df = pd.concat(
        [_merge_pair(f[0], f[1], join_keys) for f in existing_csv_files],
        ignore_index=True,
    )
    required_columns = [
        "Function",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp_function",
        "End_Timestamp_function",
        "Start_Timestamp_kernel",
        "End_Timestamp_kernel",
    ]
    missing_columns = [
        col for col in required_columns if col not in consolidated_df.columns
    ]
    if missing_columns:
        console_error(
            f"Consolidated torch trace is missing required columns {missing_columns}"
        )
        raise ValueError(
            f"Consolidated torch trace is missing required columns {missing_columns}"
        )
    consolidated_df = consolidated_df[required_columns]
    if consolidated_df.isnull().values.any():
        console_warning("Consolidated torch trace contains missing values")
        raise ValueError("Consolidated torch trace contains missing values")
    consolidated_df = consolidated_df.sort_values(by=["Function", "Counter_Name"])
    split_columns = consolidated_df["Function"].str.split(":#", expand=True)
    consolidated_df["Operator_Name"] = (
        split_columns[0] if len(split_columns.columns) > 0 else None
    )
    consolidated_df["Context_Id"] = (
        split_columns[1] if len(split_columns.columns) > 1 else None
    )
    consolidated_df.drop(columns=["Function"], inplace=True)
    consolidated_df = consolidated_df[
        [
            "Operator_Name",
            "Context_Id",
            "Kernel_Name",
            "Counter_Name",
            "Counter_Value",
            "Start_Timestamp_function",
            "End_Timestamp_function",
            "Start_Timestamp_kernel",
            "End_Timestamp_kernel",
        ]
    ]
    if consolidated_df.isnull().values.any():
        console_error(
            "Missing values in consolidated torch trace after splitting ",
            "the Function name.",
        )
        raise ValueError("Missing values in consolidated torch trace after splitting")

    return consolidated_df, torch_trace_path


def is_workload_empty(path: str) -> None:
    """Peek workload directory to verify valid profiling output"""
    workload_dir = Path(path)
    pmc_perf_path = workload_dir / "pmc_perf.csv"

    # Find PMC data files (merged or separate)
    if pmc_perf_path.is_file():
        files_to_check = [pmc_perf_path]
    else:
        files_to_check = list(workload_dir.glob("results_*.csv"))

    if not files_to_check:
        console_error("analysis", "No profiling data found.")
        return

    # Validate files are not empty
    for file_path in files_to_check:
        temp_df = pd.read_csv(file_path)
        if temp_df.dropna().empty:
            console_error(
                "profiling",
                f"Found empty cells in {file_path}.\nProfiling data could be corrupt.",
            )
            break


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


def merge_counters_spatial_multiplex(df: pd.DataFrame) -> pd.DataFrame:
    """
    For spatial multiplexing, this merges counter values for the same kernel that
    runs on different devices. For time stamp, start time stamp will use median
    while for end time stamp, it will be equal to the summation between median
    start stamp and median delta time.
    """
    non_counter_column_index = [
        "Dispatch_ID",
        "GPU_ID",
        "Queue_ID",
        "PID",
        "TID",
        "Grid_Size",
        "Workgroup_Size",
        "LDS_Per_Workgroup",
        "Scratch_Per_Workitem",
        "Arch_VGPR",
        "Accum_VGPR",
        "SGPR",
        "Wave_Size",
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
        "Correlation_ID",
        "Kernel_ID",
        "Node",
    ]

    expired_column_index = [
        "Node",
        "PID",
        "TID",
        "Queue_ID",
    ]

    kernel_name_column_name = "Kernel_Name"
    if "Kernel_Name" not in df and "Name" in df:
        kernel_name_column_name = "Name"

    # Find the values in Kernel_Name that occur more than once
    kernel_single_occurances = df[kernel_name_column_name].value_counts().index

    # Define a list to store the merged rows
    result_data: list[dict[str, Any]] = []

    for kernel_name in kernel_single_occurances:
        # Get all rows for the current kernel_name
        group = df[df[kernel_name_column_name] == kernel_name]

        # Create a dictionary to store the merged row for the current group
        merged_row: dict[str, Any] = {}

        # Process non-counter columns
        for col in [
            col for col in non_counter_column_index if col not in expired_column_index
        ]:
            if col == "Start_Timestamp":
                # For Start_Timestamp, take the median
                merged_row[col] = group["Start_Timestamp"].median()
            elif col == "End_Timestamp":
                # For End_Timestamp, calculate the median delta time
                delta_time = group[col] - group["Start_Timestamp"]
                merged_row[col] = group["Start_Timestamp"] + delta_time.median()
            else:
                # For other non-counter columns, take the first occurrence (0th row)
                merged_row[col] = group.iloc[0][col]

        # Process counter columns (assumed to be all columns not in
        # non_counter_column_index)
        counter_columns = [
            col for col in group.columns if col not in non_counter_column_index
        ]
        for counter_col in counter_columns:
            # for counter columns, take the first non-none (or non-nan) value
            current_valid_counter_group = group[group[counter_col].notna()]
            first_valid_value = (
                current_valid_counter_group.iloc[0][counter_col]
                if len(current_valid_counter_group) > 0
                else None
            )
            merged_row[counter_col] = first_valid_value

        # Append the merged row to the result list
        result_data.append(merged_row)

    return pd.DataFrame(result_data)


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
    df["Dispatch_ID"] = range(len(df))
    return df
