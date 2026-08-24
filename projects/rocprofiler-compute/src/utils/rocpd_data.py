# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import IO, Dict, Iterable, Iterator, List, Optional, Tuple

import utils.utils_profile_csv as csv_ops
from utils import csv_compression
from utils.csv_compression import PathLike, compressed_name
from utils.logger import console_error, console_warning

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
KERNEL_INFO_BY_PID_QUERY = KERNEL_INFO_QUERY.replace(
    "ORDER BY K.dispatch_id", "WHERE P.pid = ?\nORDER BY K.dispatch_id"
)
KERNEL_DISPATCH_EXISTS_QUERY = "SELECT EXISTS(SELECT 1 FROM rocpd_kernel_dispatch)"

OUT_DIR = "out"
NATIVE_COUNTERS_SUFFIX = "_native_counter_collection.csv"
# rocpd suffixes its physical tables with a uuid; the plain names are views.
ROCPD_INFO_PMC_PREFIX = "rocpd_info_pmc"
ROCPD_PMC_EVENT_PREFIX = "rocpd_pmc_event"
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


def pass_dirs(workload_dir: PathLike) -> List[Path]:
    """Return a workload's collection pass directories, in pass order."""
    out_dir = Path(workload_dir) / OUT_DIR
    return sorted(path for path in out_dir.glob("*") if db_paths(path))


def db_paths(path: PathLike) -> List[Path]:
    """Return the per-process rocpd databases of one pass, in process order."""
    return sorted(Path(path).glob("*/*.db"))


def _process_id(db_path: Path) -> str:
    return db_path.stem.split("_")[0]


def native_counters_csv(pass_path: Path, db_path: Path) -> Optional[Path]:
    """Return the native counter CSV for a process when the native lane collected it."""
    path = compressed_name(
        pass_path / f"{_process_id(db_path)}{NATIVE_COUNTERS_SUFFIX}"
    )
    return path if path.is_file() else None


def _counters_csv_has_rows(counters_csv: Path) -> bool:
    """Return whether the native counter CSV decodes to at least one counter row."""
    try:
        for row in csv_ops.iter_csv_dicts(str(counters_csv)):
            return bool(row.get("counter_value"))
    except (OSError, ValueError, *csv_compression.CORRUPT_CSV_ERRORS):
        return False
    return False


def _pmc_catalog_tables(
    conn: sqlite3.Connection,
) -> Tuple[List[str], List[str]]:
    """Return physical PMC catalog and event tables, matched on an anchored prefix."""
    catalog, events = [], []
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        if name.startswith(ROCPD_INFO_PMC_PREFIX):
            catalog.append(name)
        elif name.startswith(ROCPD_PMC_EVENT_PREFIX):
            events.append(name)
    return catalog, events


def compact_rocpd_db(db_path: Path) -> int:
    """Clear redundant PMC catalog payload when counters live in native CSV."""
    before = db_path.stat().st_size
    with closing(sqlite3.connect(db_path)) as conn:
        catalog, events = _pmc_catalog_tables(conn)
        for name in catalog:
            conn.execute(f"UPDATE \"{name}\" SET extdata = ''")
        for name in events:
            conn.execute(f'DELETE FROM "{name}"')
        conn.commit()
        # VACUUM rewrites the file in place and cannot run inside a transaction.
        conn.execute("VACUUM")
        conn.commit()
    return before - db_path.stat().st_size


def compact_pass_rocpd_dbs(pass_path: Path) -> int:
    """Compact rocpd databases whose counters were captured in the native CSV lane."""
    bytes_removed = 0
    for db_path in db_paths(pass_path):
        counters_csv = native_counters_csv(pass_path, db_path)
        if counters_csv is None or not _counters_csv_has_rows(counters_csv):
            continue
        try:
            bytes_removed += compact_rocpd_db(db_path)
        except (sqlite3.Error, OSError) as e:
            console_warning(f"Skipped compacting {db_path}: {e}")
    return bytes_removed


def iter_counter_rows(db_path: str) -> Iterator[dict]:
    """Yield the long-form counter rows of a database, in dispatch order."""
    return _iter_query_rows(db_path, COUNTERS_COLLECTION_QUERY)


def read_kernel_info(db_path: str, pid: Optional[str] = None) -> Dict[str, dict]:
    """Return kernel metadata keyed by dispatch id, optionally scoped to one pid."""
    if pid is None:
        return {
            str(row["Dispatch_ID"]): row
            for row in _iter_query_rows(db_path, KERNEL_INFO_QUERY)
        }
    return {
        str(row["Dispatch_ID"]): row
        for row in _iter_query_rows(db_path, KERNEL_INFO_BY_PID_QUERY, (pid,))
    }


def has_kernel_dispatches(db_path: str) -> bool:
    """Return whether the database recorded any kernel dispatches."""
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            with closing(conn.execute(KERNEL_DISPATCH_EXISTS_QUERY)) as cursor:
                return bool(cursor.fetchone()[0])
    except sqlite3.Error:
        return False


def convert_dbs_to_marker_csv(
    db_paths_list: List[str], marker_trace_csv_path: str
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
    pid = _process_id(db_path)
    counters_csv = native_counters_csv(pass_path, db_path)
    if counters_csv is None:
        if has_kernel_dispatches(str(db_path)):
            console_warning(
                f"No native counter CSV for pid {pid} in {pass_path}; "
                f"reading counters from {db_path.name} instead."
            )
        yield from iter_counter_rows(str(db_path))
        return

    counters = _total_counters_per_dispatch(counters_csv)
    kernel_info = read_kernel_info(str(db_path), pid)
    for dispatch_id in sorted(counters, key=int):
        dispatch = kernel_info.get(dispatch_id)
        if dispatch is None:
            continue
        for counter_name, value in counters[dispatch_id].items():
            yield {**dispatch, "Counter_Name": counter_name, "Counter_Value": value}


def _total_counters_per_dispatch(counters_csv: Path) -> Dict[str, Dict[str, float]]:
    """Sum native counter rows per dispatch, matching counters_collection."""
    totals: Dict[str, Dict[str, float]] = {}
    for row in csv_ops.iter_csv_dicts(str(counters_csv)):
        per_counter = totals.setdefault(row["dispatch_id"], {})
        counter_name = row["counter_name"]
        per_counter[counter_name] = per_counter.get(counter_name, 0.0) + float(
            row["counter_value"]
        )
    return totals


def _iter_query_rows(db_path: str, query: str, params: Tuple = ()) -> Iterator[dict]:
    """Yield query rows as dicts keyed by the column names the query selects."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        with closing(conn.execute(query, params)) as cursor:
            for row in cursor:
                yield dict(row)
