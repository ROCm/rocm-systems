# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import IO

import utils.utils_profile_csv as csv_ops
from utils import csv_compression
from utils.csv_compression import PathLike, compressed_name
from utils.logger import console_error

# From schema definition in source/share/rocprofiler-sdk-rocpd/data_views.sql
# in rocprofiler-sdk repository
COUNTERS_COLLECTION_QUERY = """
SELECT
    agent_id as GPU_ID,
    guid as GUID,
    stack_id as Correlation_Id,
    dispatch_id as Dispatch_ID,
    pid as PID,
    grid_size as Grid_Size,
    workgroup_size as Workgroup_Size,
    lds_block_size as LDS_Per_Workgroup,
    scratch_size as Scratch_Per_Workitem,
    vgpr_count as Arch_VGPR,
    accum_vgpr_count as Accum_VGPR,
    sgpr_count as SGPR,
    kernel_name as Kernel_Name,
    start as Start_Timestamp,
    end as End_Timestamp,
    kernel_id as Kernel_ID,
    counter_name as Counter_Name,
    value as Counter_Value
FROM counters_collection
"""
MARKER_API_TRACE_QUERY = """
SELECT
    category AS Domain,
    json_extract(extdata, '$.message') AS Function,
    pid AS Process_Id,
    tid AS Thread_Id,
    stack_id AS Correlation_Id,
    guid AS GUID,
    start AS Start_Timestamp,
    end AS End_Timestamp
FROM regions
ORDER BY start
"""
MARKER_COLUMNS = (
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "GUID",
    "Start_Timestamp",
    "End_Timestamp",
)
# Non-counter columns from counters_collection, read straight from the tables
# the view builds on. Native counter CSVs live in a separate lane, so a db
# the native tool collected has dispatches but no counters_collection rows.
KERNEL_INFO_QUERY = """
SELECT
    K.agent_id as GPU_ID,
    K.guid as GUID,
    E.stack_id as Correlation_Id,
    K.dispatch_id as Dispatch_ID,
    P.pid as PID,
    (K.grid_size_x * K.grid_size_y * K.grid_size_z) as Grid_Size,
    (K.workgroup_size_x * K.workgroup_size_y * K.workgroup_size_z) as Workgroup_Size,
    K.group_segment_size as LDS_Per_Workgroup,
    K.private_segment_size as Scratch_Per_Workitem,
    S.arch_vgpr_count as Arch_VGPR,
    S.accum_vgpr_count as Accum_VGPR,
    S.sgpr_count as SGPR,
    S.display_name as Kernel_Name,
    K.start as Start_Timestamp,
    K.end as End_Timestamp,
    K.kernel_id as Kernel_ID
FROM rocpd_kernel_dispatch K
    INNER JOIN rocpd_event E ON E.id = K.event_id AND E.guid = K.guid
    INNER JOIN rocpd_info_kernel_symbol S ON S.id = K.kernel_id AND S.guid = K.guid
    INNER JOIN rocpd_info_process P ON P.id = K.pid AND P.guid = K.guid
ORDER BY K.dispatch_id
"""
KERNEL_DISPATCH_EXISTS_QUERY = "SELECT EXISTS(SELECT 1 FROM rocpd_kernel_dispatch)"

OUT_DIR = "out"
NATIVE_COUNTERS_SUFFIX = "_native_counter_collection.csv"
PMC_COLUMNS = (
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
)


def pass_dir(workload_dir: PathLike, fbase: str) -> Path:
    """Return the directory holding one collection pass' per-process artifacts."""
    return Path(workload_dir) / OUT_DIR / fbase


def pass_dirs(workload_dir: PathLike) -> list[Path]:
    """Return a workload's collection pass directories, in pass order."""
    out_dir = Path(workload_dir) / OUT_DIR
    return sorted(path for path in out_dir.glob("*") if db_paths(path))


def db_paths(path: PathLike) -> list[Path]:
    """Return the per-process rocpd databases of one pass, in process order."""
    return sorted(Path(path).glob("*/*.db"))


def iter_counter_rows(db_path: str) -> Iterator[dict]:
    """Yield the long-form counter rows of a database, in dispatch order."""
    return _iter_query_rows(db_path, COUNTERS_COLLECTION_QUERY)


def read_kernel_info(db_path: str) -> dict[str, dict]:
    """Return kernel metadata for every dispatch, keyed by dispatch id as text."""
    return {
        str(row["Dispatch_ID"]): row
        for row in _iter_query_rows(db_path, KERNEL_INFO_QUERY)
    }


def has_kernel_dispatches(db_path: str) -> bool:
    """Whether a database recorded a kernel dispatch at all."""
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.execute(KERNEL_DISPATCH_EXISTS_QUERY)) as cursor:
            return bool(cursor.fetchone()[0])


def convert_dbs_to_marker_csv(
    db_paths_list: list[str], marker_trace_csv_path: str
) -> None:
    """Write the marker regions of every database into one compressed CSV."""
    with csv_compression.open_gzip_csv_write(marker_trace_csv_path) as output:
        writer = csv.DictWriter(
            output, fieldnames=MARKER_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for db_path in db_paths_list:
            try:
                writer.writerows(_iter_query_rows(db_path, MARKER_API_TRACE_QUERY))
            except sqlite3.Error as e:
                console_error(
                    f"Database error while extracting the marker trace "
                    f"from {db_path}: {e}"
                )


def iter_workload_rows(workload_dir: PathLike) -> Iterator[dict]:
    """Yield the counter rows of every collection pass of a workload."""
    for path in pass_dirs(workload_dir):
        yield from iter_pass_rows(path)


def iter_pass_rows(path: PathLike) -> Iterator[dict]:
    """Yield counter rows for one pass, renumbering dispatch and kernel ids."""
    dispatch_ids = csv_ops.GroupIdAssigner(
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
    )
    kernel_ids = csv_ops.GroupIdAssigner(
        ["Kernel_Name", "Grid_Size", "Workgroup_Size", "LDS_Per_Workgroup"],
        "Kernel_ID",
    )
    pass_path = Path(path)
    for db_path in db_paths(pass_path):
        for row in _iter_process_rows(pass_path, db_path):
            yield kernel_ids.apply(dispatch_ids.apply(row))


def write_pmc_rows(rows: Iterable[dict], output: IO[str]) -> int:
    """Write long-form counter rows to an open file; return the row count."""
    writer = csv.DictWriter(output, fieldnames=PMC_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    written = 0
    for row in rows:
        writer.writerow(row)
        written += 1
    return written


def _iter_process_rows(pass_path: Path, db_path: Path) -> Iterator[dict]:
    """Yield counter rows for one process from whichever tool collected them."""
    pid = db_path.stem.split("_")[0]
    counters_csv = compressed_name(pass_path / f"{pid}{NATIVE_COUNTERS_SUFFIX}")
    if not counters_csv.is_file():
        yield from iter_counter_rows(str(db_path))
        return

    counters = _total_counters_per_dispatch(counters_csv)
    kernel_info = read_kernel_info(str(db_path))
    for dispatch_id in sorted(counters, key=int):
        dispatch = kernel_info.get(dispatch_id)
        if dispatch is None:
            continue
        for counter_name, value in counters[dispatch_id].items():
            yield {**dispatch, "Counter_Name": counter_name, "Counter_Value": value}


def _total_counters_per_dispatch(counters_csv: Path) -> dict[str, dict[str, float]]:
    """Sum native counter rows per dispatch, matching counters_collection."""
    totals: dict[str, dict[str, float]] = {}
    for row in csv_ops.iter_csv_dicts(str(counters_csv)):
        per_counter = totals.setdefault(row["dispatch_id"], {})
        counter_name = row["counter_name"]
        per_counter[counter_name] = per_counter.get(counter_name, 0.0) + float(
            row["counter_value"]
        )
    return totals


def _iter_query_rows(db_path: str, query: str) -> Iterator[dict]:
    """Yield query rows as dicts keyed by the column names the query selects."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        with closing(conn.execute(query)) as cursor:
            for row in cursor:
                yield dict(row)
