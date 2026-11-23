# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import re
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import csv_compression
from utils.logger import (
    console_debug,
    console_error,
    console_warning,
    demarcate,
)
from utils.utils_common import (
    NATIVE_AGENTS_PREFIX,
    NATIVE_COUNTERS_PREFIX,
    NATIVE_DISPATCH_PREFIX,
    NATIVE_KERNEL_SYMBOLS_PREFIX,
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
# process_rocpd_csv merges the passes of one dispatch on these keys.
_MERGE_KEYS = ["Dispatch_ID", *_KERNEL_ID_KEYS]


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


@demarcate
def impute_counters_iteration_multiplex(
    df: pd.DataFrame,
    workload_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, ChainMap]]:
    """Fill in the counters each launch did not collect under iteration
    multiplexing.

    Each launch of a kernel collects one pass, in turn. A round is one launch
    of every pass. Returns two things:

    - The imputed frame. Each launch takes its missing counters from the
      other launches of its round.
    - One table per pass. Each launch takes that pass's counters from the one
      launch of its round that collected the pass, so a pass's counters always
      come from one launch.
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
    unique_occurences = df.groupby(
        [
            "Kernel_Name",
            "Grid_Size",
            "Workgroup_Size",
            "LDS_Per_Workgroup",
        ],
        as_index=False,
    )

    counter_columns = [col for col in df.columns if col not in non_counter_column_index]
    # Collect imputed groups as dataframes
    group_dfs: list[pd.DataFrame] = []
    # Each pass's counter values, one piece per kernel, keyed by the set of
    # counters the pass collects
    pass_pieces: dict[frozenset[str], list[pd.DataFrame]] = {}

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

        # Find the passes, in the order the launches collect them
        counter_groups: list[frozenset[str]] = []
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
            counter_groups.append(cols_frozenset)

        # If no counters found for this group, continue
        if not counter_groups:
            continue

        # Iterate over subgroups of dispatches containing
        # all counters and impute missing values
        # Create subgroup_id column for groupby: 0,0,0,...,1,1,1,...,2,2,2,...
        # Use numpy for vectorized operation
        round_ids = np.arange(len(group)) // len(counter_groups)
        # Which counters each launch collected
        collected = group[counter_columns].notna()
        for counter_group in counter_groups:
            pass_pieces.setdefault(counter_group, []).append(
                _pass_values_by_round(group, collected, counter_group, round_ids)
            )
        group_copy = group.copy()
        group_copy["__subgroup_id"] = round_ids
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
        return pd.DataFrame(columns=df.columns), {}
    # Pass tables join their pieces by the original row labels, so reset
    # the labels last
    imputed_df = pd.concat(group_dfs)
    merged_df = imputed_df.reset_index(drop=True)
    dispatches = merged_df.drop(columns=counter_columns)
    # A kernel that never collected a pass has empty counters in its table
    pass_tables = {
        f"pass_{number}": ChainMap(
            pd.concat(pieces).reindex(imputed_df.index).reset_index(drop=True),
            dispatches,
        )
        for number, pieces in enumerate(pass_pieces.values())
    }
    return merged_df, pass_tables


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


def _pass_values_by_round(
    group: pd.DataFrame,
    collected: pd.DataFrame,
    counter_group: frozenset[str],
    round_ids: np.ndarray,
) -> pd.DataFrame:
    """Give every launch of one kernel the counters of one pass.

    The values come from the launch of the same round that collected exactly
    that pass. A round without such a launch gets empty values.
    """
    columns = sorted(counter_group)
    # Launches whose collected counters are exactly this pass's counters
    collected_this_pass = collected[columns].all(axis=1) & ~collected.drop(
        columns=columns
    ).any(axis=1)
    source = (
        group
        .loc[collected_this_pass, columns]
        .assign(Round=round_ids[collected_this_pass.to_numpy()])
        .drop_duplicates("Round")
    )
    # Copy each round's source values onto every launch of that round
    values = (
        pd
        .DataFrame({"Round": round_ids})
        .merge(source, on="Round", how="left")
        .drop(columns="Round")
    )
    values.index = group.index
    return values


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
    for group_key, group_df in df.groupby(_MERGE_KEYS):
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
        # Each counter becomes its own column. A counter that several passes
        # collected keeps the value of the first pass.
        first_values = group_df.drop_duplicates("Counter_Name")
        row.update(
            dict(zip(first_values["Counter_Name"], first_values["Counter_Value"]))
        )
        data.append(row)
        slot_to_id[group_key] = len(data)
    df = pd.DataFrame(data)
    # Reset dispatch IDs
    df["Dispatch_ID"] = range(1, len(df) + 1)
    return df, slot_to_id


def split_counters_by_pass(
    counter_rows: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> Dict[str, ChainMap]:
    """Return one table per pass, keyed by Pass_Key.

    Each table has the rows of merged_df, its dispatch columns, and only the
    counters and timestamps that pass collected. A dispatch the pass did not see
    has empty counters there.

    A table is a ChainMap: table[column] reads the pass's own counters and
    timestamps first, then the dispatch columns, which all passes share.
    """
    if counter_rows.empty or "Pass_Key" not in counter_rows.columns:
        return {}
    counter_rows = counter_rows.assign(
        Merged_Row=counter_rows.groupby(_MERGE_KEYS).ngroup()
    )
    counter_names = set(counter_rows["Counter_Name"])
    dispatch_columns = [
        column
        for column in merged_df.columns
        if column not in counter_names
        and column not in ("Start_Timestamp", "End_Timestamp")
    ]
    dispatches = merged_df[dispatch_columns]
    pass_tables: Dict[str, ChainMap] = {}
    for pass_key, rows in counter_rows.groupby("Pass_Key", sort=False):
        counters = rows.pivot_table(
            index="Merged_Row",
            columns="Counter_Name",
            values="Counter_Value",
            aggfunc="first",
        )
        timestamps = rows.groupby("Merged_Row")[["Start_Timestamp", "End_Timestamp"]]
        pass_counters = timestamps.first().join(counters).reindex(merged_df.index)
        pass_tables[str(pass_key)] = ChainMap(pass_counters, dispatches)
    return pass_tables


def pass_tables_on_rows(
    pass_tables: Dict[str, ChainMap], rows: pd.DataFrame
) -> List[ChainMap]:
    """Return the pass tables cut to the rows of rows, part of the merged table.

    The pass tables share one dispatches table, so it is cut once.
    """
    if not pass_tables:
        return []
    _counters, dispatches = next(iter(pass_tables.values())).maps
    dispatches_on_rows = dispatches.loc[rows.index]
    tables_on_rows = []
    for table in pass_tables.values():
        counters = table.maps[0]
        tables_on_rows.append(ChainMap(counters.loc[rows.index], dispatches_on_rows))
    return tables_on_rows


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
