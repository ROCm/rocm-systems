# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Union

from utils.logger import console_error


def find_workload_db_paths(workload_dir: Union[Path, str]) -> list[str]:
    """Return all rocpd .db files at the workload root: <workload>/<fbase>.db."""
    return [str(p) for p in sorted(Path(workload_dir).glob("*.db"))]


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
KERNEL_DISPATCH_QUERY = """
SELECT dispatch_id, event_id, guid
FROM rocpd_kernel_dispatch
WHERE guid = ?
"""
ROCPD_PMC_EVENT_TABLE_NAME_PREFIX = "rocpd_pmc_event_"
TABLE_NAME_PREFIX_QUERY = (
    "SELECT name FROM sqlite_master WHERE type='table' "
    "AND name LIKE '{table_name_prefix}%'"
)
INSERT_QUERY = "INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"


def _dump_query_to_csv(
    db_paths: list[str],
    query: str,
    csv_path: str,
) -> None:
    """Stream the rows of `query` across all `db_paths` into a single CSV.
    Rows are written out as the cursor produces them.
    """
    header_written = False
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for db_path in db_paths:
            with closing(sqlite3.connect(db_path)) as conn:
                try:
                    with closing(conn.execute(query)) as cursor:
                        if cursor.description is None:
                            continue
                        if not header_written:
                            writer.writerow([d[0] for d in cursor.description])
                            header_written = True
                        writer.writerows(cursor)
                except OSError as e:
                    console_error(
                        f"Database error while extracting {csv_path} "
                        f"from {db_path}: {e}"
                    )
                except Exception as e:
                    console_error(
                        f"Unexpected error while extracting {csv_path} "
                        f"from {db_path}: {e}"
                    )


def dump_counter_collection_csv(
    db_paths: list[str], counter_collection_csv_path: str
) -> None:
    """Write the counters_collection table from each db into one CSV."""
    _dump_query_to_csv(db_paths, COUNTERS_COLLECTION_QUERY, counter_collection_csv_path)


def dump_marker_trace_csv(db_paths: list[str], marker_trace_csv_path: str) -> None:
    """Write the regions (marker API) table from each db into one CSV."""
    _dump_query_to_csv(db_paths, MARKER_API_TRACE_QUERY, marker_trace_csv_path)


def merge_pass_dbs(src_db_paths: list[str], dst_db_path: str) -> None:
    """Merge multiple per-host rocpd ``.db`` files into a single ``.db``."""
    if not src_db_paths:
        console_error(f"merge_pass_dbs called with no source dbs (dst={dst_db_path})")
        return

    Path(dst_db_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(dst_db_path).exists():
        Path(dst_db_path).unlink()

    shutil.copyfile(src_db_paths[0], dst_db_path)
    if len(src_db_paths) == 1:
        return

    with closing(sqlite3.connect(dst_db_path)) as conn:
        with closing(
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ) as cursor:
            tables = [row[0] for row in cursor.fetchall()]
        for i, src in enumerate(src_db_paths[1:], start=1):
            alias = f"src{i}"
            conn.execute(f"ATTACH DATABASE ? AS {alias}", (src,))
            try:
                for table in tables:
                    conn.execute(
                        f'INSERT INTO main."{table}" SELECT * FROM {alias}."{table}"'
                    )
                conn.commit()
            finally:
                conn.execute(f"DETACH DATABASE {alias}")


def count_counter_rows(db_paths: list[str]) -> int:
    """Return total number of counters_collection rows across all dbs."""
    total = 0
    for db_path in db_paths:
        with closing(sqlite3.connect(db_path)) as conn:
            try:
                with closing(
                    conn.execute("SELECT COUNT(*) FROM counters_collection")
                ) as cursor:
                    row = cursor.fetchone()
                    if row:
                        total += int(row[0])
            except Exception as e:
                console_error(f"Unexpected error counting rows in {db_path}: {e}")
    return total


def update_rocpd_pmc_events(counter_info: list[dict], rocpd_db_path: str) -> None:
    """Updates pmc_event table in the given rocpd database path."""
    try:
        with closing(sqlite3.connect(rocpd_db_path)) as conn:
            # Get pmc_event table name
            with closing(
                conn.execute(
                    TABLE_NAME_PREFIX_QUERY.format(
                        table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
                    )
                )
            ) as cursor:
                table_name = cursor.fetchone()
            if table_name is None:
                console_error("No pmc_event table found in the rocpd database")
            table_name = table_name[0]

            # get pmc_event table data
            guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace(
                "_", "-"
            )
            # Map dispatch_id to event_id from rocpd_kernel_dispatch
            # Native counter collection CSV has dispatch_id, but schema needs event_id
            # event_id may differ from dispatch_id when marker API tracing is enabled
            with closing(conn.execute(KERNEL_DISPATCH_QUERY, (guid,))) as cursor:
                db_rows = cursor.fetchall()
            if not db_rows:
                console_error("No kernel dispatch data found.")
                return
            # DB output (numeric) converted to str to align with counter_info
            dispatch_to_event = {
                str(dispatch_id): str(event_id) for dispatch_id, event_id, _ in db_rows
            }

            # Map dispatch_id to event_id for each row
            # Create new event_id column without destroying dispatch_id
            for row in counter_info:
                dispatch_id = row.get("dispatch_id")
                row["event_id"] = dispatch_to_event.get(dispatch_id)

            columns = ("guid", "event_id", "pmc_id", "value")
            values = [
                (
                    guid,
                    row.get("event_id"),
                    row.get("counter_id"),
                    row.get("counter_value"),
                )
                for row in counter_info
            ]

            # insert into pmc_event table
            with conn:
                placeholders = ", ".join(["?"] * len(columns))
                conn.executemany(
                    INSERT_QUERY.format(
                        table_name=table_name,
                        columns=", ".join(columns),
                        placeholders=placeholders,
                    ),
                    values,
                )
    except OSError as e:
        console_error(f"Database error while updating pmc_event table: {e}")
    except Exception as e:
        console_error(f"Unexpected error updating pmc_event table: {e}")
