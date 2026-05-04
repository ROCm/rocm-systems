# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import csv
import sqlite3
from collections.abc import Generator, Iterable
from contextlib import ExitStack, closing

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


def convert_dbs_to_csvs(
    db_paths: list[str],
    counter_collection_csv_path: str,
    marker_trace_csv_path: str,
    results_csv_path: str,
) -> int:
    """Export rocpd DBs to the counter, marker, and results CSVs.

    Streams the counters_collection view from each DB, assigns final
    Dispatch_ID and Kernel_ID on the fly, drops PID, and writes directly
    to the output CSVs in a single pass per DB. Returns the total number
    of counter rows written across all DBs.
    """
    # IDs are assigned globally so the same dispatch/kernel keeps the same
    # ID across DBs in a multi-pass workload.
    dispatch_groups: dict[tuple, int] = {}
    kernel_groups: dict[tuple, int] = {}
    # Track header-written state across DBs.
    counter_header_written = False
    marker_header_written = False
    total_rows = 0

    with ExitStack() as stack:
        counter_writer = csv.writer(
            stack.enter_context(open(counter_collection_csv_path, "w", newline=""))
        )
        results_writer = csv.writer(
            stack.enter_context(open(results_csv_path, "w", newline=""))
        )
        marker_writer = csv.writer(
            stack.enter_context(open(marker_trace_csv_path, "w", newline=""))
        )
        counter_writers = [counter_writer, results_writer]

        for db_path in db_paths:
            with closing(sqlite3.connect(db_path)) as conn:
                marker_header_written = _stream_db_marker_trace(
                    conn,
                    marker_writer,
                    marker_header_written,
                    db_path,
                )
                rows_written, counter_header_written = _stream_db_counters(
                    conn,
                    counter_writers,
                    counter_header_written,
                    dispatch_groups,
                    kernel_groups,
                    db_path,
                )
                total_rows += rows_written

    return total_rows


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
    except sqlite3.Error as e:
        console_error(f"Database error while updating pmc_event table: {e}")
    except Exception as e:
        console_error(f"Unexpected error updating pmc_event table: {e}")


def _assign_counter_ids(
    row: tuple,
    column_positions: dict[str, int],
    dispatch_groups: dict[tuple, int],
    kernel_groups: dict[tuple, int],
) -> tuple[int, int]:
    """Return sequential (dispatch_id, kernel_id) for a counter row.

    IDs are assigned in first-seen order: the first unique combination
    of grouping columns gets ID 0, the next unseen combination gets 1,
    and so on.
    """
    # Dispatch uniqueness includes PID and timestamps because the same
    # kernel can be dispatched multiple times.
    dispatch_key = (
        row[column_positions["PID"]],
        row[column_positions["Kernel_Name"]],
        row[column_positions["Grid_Size"]],
        row[column_positions["Workgroup_Size"]],
        row[column_positions["LDS_Per_Workgroup"]],
        row[column_positions["Start_Timestamp"]],
        row[column_positions["End_Timestamp"]],
    )
    dispatch_id = dispatch_groups.setdefault(dispatch_key, len(dispatch_groups))
    kernel_key = (
        row[column_positions["Kernel_Name"]],
        row[column_positions["Grid_Size"]],
        row[column_positions["Workgroup_Size"]],
        row[column_positions["LDS_Per_Workgroup"]],
    )
    kernel_id = kernel_groups.setdefault(kernel_key, len(kernel_groups))
    return dispatch_id, kernel_id


def _compose_output_row(
    row: tuple,
    column_positions: dict[str, int],
    dispatch_id: int,
    kernel_id: int,
) -> tuple:
    """Return the output row tuple: PID dropped, IDs substituted in place."""
    pid_position = column_positions["PID"]
    overrides = {
        column_positions["Dispatch_ID"]: dispatch_id,
        column_positions["Kernel_ID"]: kernel_id,
    }
    return tuple(
        overrides.get(position, value)
        for position, value in enumerate(row)
        if position != pid_position
    )


def _regroup_counter_rows(
    cursor: sqlite3.Cursor,
    column_positions: dict[str, int],
    dispatch_groups: dict[tuple, int],
    kernel_groups: dict[tuple, int],
) -> Generator[tuple, None, None]:
    """Yield counter rows with final IDs assigned and PID removed."""
    for row in cursor:
        dispatch_id, kernel_id = _assign_counter_ids(
            row,
            column_positions,
            dispatch_groups,
            kernel_groups,
        )
        yield _compose_output_row(
            row,
            column_positions,
            dispatch_id,
            kernel_id,
        )


def _write_rows_to_csv_writers(
    row_iterator: Iterable[tuple],
    writers: list[csv.writer],
) -> int:
    """Write each row to all writers. Return total rows written."""
    count = 0
    for row in row_iterator:
        for writer in writers:
            writer.writerow(row)
        count += 1
    return count


def _stream_db_marker_trace(
    conn: sqlite3.Connection,
    writer: csv.writer,
    header_written: bool,
    db_path: str,
) -> bool:
    """Stream marker_api_trace results from one DB into a CSV writer.

    Writes the CSV header on the first call that produces a valid cursor.
    Returns the updated header_written state.
    """
    try:
        with closing(conn.execute(MARKER_API_TRACE_QUERY)) as cursor:
            if cursor.description is None:
                return header_written
            if not header_written:
                writer.writerow([desc[0] for desc in cursor.description])
                header_written = True
            writer.writerows(cursor)
    except sqlite3.Error as exc:
        console_error(f"Database error extracting marker trace from {db_path}: {exc}")
    except Exception as exc:
        console_error(f"Unexpected error extracting marker trace from {db_path}: {exc}")
    return header_written


def _stream_db_counters(
    conn: sqlite3.Connection,
    writers: list[csv.writer],
    header_written: bool,
    dispatch_groups: dict[tuple, int],
    kernel_groups: dict[tuple, int],
    db_path: str,
) -> tuple[int, bool]:
    """Stream regrouped counter rows from one DB into the writers.

    Writes the CSV header to each writer on the first call that produces
    a valid cursor. Returns (rows_written, updated header_written state).
    """
    try:
        with closing(conn.execute(COUNTERS_COLLECTION_QUERY)) as cursor:
            if cursor.description is None:
                return 0, header_written
            column_positions = {
                desc[0]: idx for idx, desc in enumerate(cursor.description)
            }
            if not header_written:
                # PID is used for dispatch grouping but excluded from output.
                output_header = [name for name in column_positions if name != "PID"]
                for writer in writers:
                    writer.writerow(output_header)
                header_written = True
            regrouped = _regroup_counter_rows(
                cursor,
                column_positions,
                dispatch_groups,
                kernel_groups,
            )
            return _write_rows_to_csv_writers(regrouped, writers), header_written
    except sqlite3.Error as exc:
        console_error(f"Database error extracting counters from {db_path}: {exc}")
    except Exception as exc:
        console_error(f"Unexpected error extracting counters from {db_path}: {exc}")
    return 0, header_written
