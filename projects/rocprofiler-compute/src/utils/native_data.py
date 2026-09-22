# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Build the analyze counter frame from the native tool's per-pid artifacts.

The native tool writes three files per process per counter set: the counters,
one row per dispatch per counter; the dispatches, one row each; and the kernel
symbols, one row per kernel. They are joined here rather than at profile time,
so profile mode only moves files and never reads them back.

Join keys are ``(pid, dispatch_id)`` for counters to dispatches and
``(pid, kernel_id)`` for dispatches to symbols. Both ids are only unique within
a process, and the pid comes from the filename.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from utils import csv_compression
from utils.logger import console_debug, console_warning
from utils.utils_profile_csv import GroupIdAssigner

COUNTERS_PREFIX = "counters"
DISPATCH_PREFIX = "dispatch"
KERNEL_SYMBOLS_PREFIX = "kernel_symbols"

# counters_<fbase>_<pid>.csv.gz, and the same shape for the other two kinds.
_ARTIFACT_RE = re.compile(
    rf"^(?P<kind>{COUNTERS_PREFIX}|{DISPATCH_PREFIX}|{KERNEL_SYMBOLS_PREFIX})"
    r"_(?P<fbase>.+)_(?P<pid>\d+)\.csv\.gz$"
)

# Column order of the frame analyze consumes, matching what the rocpd
# conversion produces so nothing downstream has to change.
RESULTS_COLUMNS = [
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


@dataclass(frozen=True)
class NativeArtifacts:
    """The three files one process wrote for one counter set."""

    fbase: str
    pid: int
    counters: Path
    dispatch: Path
    kernel_symbols: Path


def find_native_artifacts(workload_dir: Path) -> list[NativeArtifacts]:
    """Return the complete per-pid artifact sets in workload_dir.

    Sorted by counter set then pid, because the dispatch ids analyze hands out
    follow the order rows are read in. A set missing one of its three files is
    skipped with a warning: it cannot be joined, and dropping it is better than
    reporting a process with no counters.
    """
    found: dict[tuple[str, int], dict[str, Path]] = {}
    for path in workload_dir.glob(f"*.csv{csv_compression.GZIP_SUFFIX}"):
        match = _ARTIFACT_RE.match(path.name)
        if match is None:
            continue
        key = (match["fbase"], int(match["pid"]))
        found.setdefault(key, {})[match["kind"]] = path

    artifacts = []
    for (fbase, pid), paths in sorted(found.items()):
        missing = {COUNTERS_PREFIX, DISPATCH_PREFIX, KERNEL_SYMBOLS_PREFIX} - set(paths)
        if missing:
            console_warning(
                f"Incomplete native profiling data for pid {pid} of {fbase}: "
                f"missing {', '.join(sorted(missing))}. Skipping this process."
            )
            continue
        artifacts.append(
            NativeArtifacts(
                fbase=fbase,
                pid=pid,
                counters=paths[COUNTERS_PREFIX],
                dispatch=paths[DISPATCH_PREFIX],
                kernel_symbols=paths[KERNEL_SYMBOLS_PREFIX],
            )
        )
    return artifacts


def _read_keyed_rows(path: Path, key_column: str) -> dict[str, dict[str, str]]:
    """Read a small CSV into a dict keyed by one column."""
    with csv_compression.open_gzip_csv_read(path) as infile:
        return {row[key_column]: row for row in csv.DictReader(infile)}


def _sum_counter_instances(counters_path: Path) -> dict[tuple[str, str], float]:
    """Total each counter per dispatch, across the counter's instances.

    The tool records one row per hardware instance of a counter, which is what
    the rocpd view sums at read time. Rows for one dispatch are not guaranteed
    to be adjacent, so the totals are accumulated over the whole file. The
    result holds one entry per output row, not one per input row.
    """
    totals: dict[tuple[str, str], float] = {}
    with csv_compression.open_gzip_csv_read(counters_path) as infile:
        for row in csv.DictReader(infile):
            key = (row["dispatch_id"], row["counter_name"])
            totals[key] = totals.get(key, 0.0) + float(row["counter_value"])
    return totals


def _join_process(artifacts: NativeArtifacts) -> Iterator[dict]:
    """Yield one analyze row per dispatch and counter of a single process."""
    dispatches = _read_keyed_rows(artifacts.dispatch, "dispatch_id")
    symbols = _read_keyed_rows(artifacts.kernel_symbols, "kernel_id")

    for (dispatch_id, counter_name), counter_value in _sum_counter_instances(
        artifacts.counters
    ).items():
        dispatch = dispatches.get(dispatch_id)
        if dispatch is None:
            continue
        symbol = symbols.get(dispatch["kernel_id"])
        if symbol is None:
            continue

        yield {
            "GPU_ID": dispatch["gpu_id"],
            # The native tool has no session uuid. The pid identifies the
            # process the rows came from, which is what GUID is used for.
            "GUID": artifacts.pid,
            "Correlation_Id": dispatch["correlation_id"],
            "Grid_Size": dispatch["grid_size"],
            "Workgroup_Size": dispatch["workgroup_size"],
            "LDS_Per_Workgroup": dispatch["lds_per_workgroup"],
            "Scratch_Per_Workitem": dispatch["scratch_per_workitem"],
            "Arch_VGPR": symbol["arch_vgpr"],
            "Accum_VGPR": symbol["accum_vgpr"],
            "SGPR": symbol["sgpr"],
            "Kernel_Name": symbol["kernel_name"],
            "Start_Timestamp": dispatch["start_timestamp"],
            "End_Timestamp": dispatch["end_timestamp"],
            "Counter_Name": counter_name,
            "Counter_Value": counter_value,
            "PID": artifacts.pid,
        }


def write_counter_frame(workload_dir: Path, output_path: Path) -> int:
    """Join the native artifacts in workload_dir into output_path.

    Returns the number of counter rows written. Dispatch and kernel ids are
    reassigned per counter set, the way profile mode reassigns them for the
    rocpd path, so that the separate runs one counter set each can be lined up
    with one another afterwards.
    """
    artifacts = find_native_artifacts(workload_dir)
    if not artifacts:
        return 0

    rows_written = 0
    with csv_compression.open_gzip_csv_write(output_path) as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=RESULTS_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()

        for fbase in sorted({a.fbase for a in artifacts}):
            # Ids restart with each counter set, and are handed out in the
            # order rows first appear, so the sets line up with each other.
            dispatch_ids = GroupIdAssigner(
                [
                    "PID",
                    "Kernel_Name",
                    "Grid_Size",
                    "Workgroup_Size",
                    "LDS_Per_Workgroup",
                    "Start_Timestamp",
                    "End_Timestamp",
                ],
                "Dispatch_ID",
                start=1,
            )
            kernel_ids = GroupIdAssigner(
                ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
                "Kernel_ID",
            )

            for process in (a for a in artifacts if a.fbase == fbase):
                for row in _join_process(process):
                    writer.writerow(kernel_ids.apply(dispatch_ids.apply(row)))
                    rows_written += 1

    console_debug(f"Created file: {output_path} ({rows_written} counter rows)")
    return rows_written
