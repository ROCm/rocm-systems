# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import sqlite3
from contextlib import ExitStack, closing
from typing import Optional

import utils.utils_profile_csv as csv_ops
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

# Rows per executemany call. Must be >1 to avoid row-by-row overhead
# and less than the full file to preserve the streaming memory benefit.
PMC_EVENT_INSERT_BATCH_SIZE = 50_000


def convert_dbs_to_csv(
    db_paths: list[str],
    counter_collection_csv_path: str,
    marker_trace_csv_path: str,
) -> None:
    queries = {
        counter_collection_csv_path: COUNTERS_COLLECTION_QUERY,
        marker_trace_csv_path: MARKER_API_TRACE_QUERY,
    }
    header_written = {path: False for path in queries}

    with ExitStack() as stack:
        writers = {
            path: csv.writer(
                stack.enter_context(open(path, "w", newline="", encoding="utf-8"))
            )
            for path in queries
        }
        for db_path in db_paths:
            with closing(sqlite3.connect(db_path)) as conn:
                for file_path, query in queries.items():
                    try:
                        with closing(conn.execute(query)) as cursor:
                            if cursor.description is None:
                                continue
                            if not header_written[file_path]:
                                writers[file_path].writerow([
                                    desc[0] for desc in cursor.description
                                ])
                                header_written[file_path] = True
                            writers[file_path].writerows(cursor)
                    except OSError as e:
                        console_error(
                            f"Database error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )
                    except Exception as e:
                        console_error(
                            f"Unexpected error while extracting {file_path} "
                            f"from {db_path}: {e}"
                        )


def update_rocpd_pmc_events(
    counter_csv_path: str,
    rocpd_db_path: str,
    batch_size: int = PMC_EVENT_INSERT_BATCH_SIZE,
) -> None:
    """Stream counter CSV in batches into the rocpd pmc_event table."""
    try:
        with closing(sqlite3.connect(rocpd_db_path)) as conn:
            metadata = _resolve_pmc_event_metadata(conn)
            if metadata is None:
                return
            table_name, guid, dispatch_to_event = metadata

            _stream_csv_to_pmc_event_table(
                conn,
                counter_csv_path,
                table_name,
                guid,
                dispatch_to_event,
                batch_size,
            )
    except OSError as e:
        console_error(f"Error while updating pmc_event table: {e}")
    except Exception as e:
        console_error(f"Unexpected error updating pmc_event table: {e}")


def _resolve_pmc_event_metadata(
    conn: sqlite3.Connection,
) -> Optional[tuple[str, str, dict[str, str]]]:
    """Look up the pmc_event table and build dispatch-to-event mapping.

    Returns (table_name, guid, dispatch_to_event) or None on failure.
    """
    with closing(
        conn.execute(
            TABLE_NAME_PREFIX_QUERY.format(
                table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
            )
        )
    ) as cursor:
        table_name = cursor.fetchone()

    if table_name is None:
        console_error("No pmc_event table found in the rocpd database", exit=False)
        return None
    table_name = table_name[0]

    guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace("_", "-")

    # event_id may differ from dispatch_id when marker API tracing is enabled
    with closing(conn.execute(KERNEL_DISPATCH_QUERY, (guid,))) as cursor:
        db_rows = cursor.fetchall()

    if not db_rows:
        console_error("No kernel dispatch data found.", exit=False)
        return None

    dispatch_to_event = {
        str(dispatch_id): str(event_id) for dispatch_id, event_id, _ in db_rows
    }
    return table_name, guid, dispatch_to_event


def _stream_csv_to_pmc_event_table(
    conn: sqlite3.Connection,
    counter_csv_path: str,
    table_name: str,
    guid: str,
    dispatch_to_event: dict[str, str],
    batch_size: int,
) -> None:
    """Read counter CSV in batches and insert into the pmc_event table."""
    columns = ("guid", "event_id", "pmc_id", "value")
    placeholders = ", ".join(["?"] * len(columns))
    insert_sql = INSERT_QUERY.format(
        table_name=table_name,
        columns=", ".join(columns),
        placeholders=placeholders,
    )

    batch: list[tuple[Optional[str], ...]] = []
    with conn:
        for row in csv_ops.iter_csv_dicts(counter_csv_path):
            event_id = dispatch_to_event.get(row.get("dispatch_id", ""))
            batch.append((
                guid,
                event_id,
                row.get("counter_id"),
                row.get("counter_value"),
            ))
            if len(batch) >= batch_size:
                conn.executemany(insert_sql, batch)
                batch.clear()

        if batch:
            conn.executemany(insert_sql, batch)
