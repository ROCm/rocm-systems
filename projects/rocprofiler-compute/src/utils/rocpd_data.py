# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

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
ROCPD_QUERY_SURFACES = ("counters_collection", "regions")


def build_pass_db(db_paths: list[str], output_db_path: str) -> None:
    """Build one pass-level rocpd database from profiler-produced DBs."""
    sorted_db_paths = sorted(db_paths)
    if not sorted_db_paths:
        console_error("No rocpd database files found.")
        return

    output_path = Path(output_db_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    if len(sorted_db_paths) == 1:
        source_path = Path(sorted_db_paths[0])
        shutil.copyfile(source_path, output_path)
        return

    with closing(sqlite3.connect(output_path)) as conn:
        for surface_name in ROCPD_QUERY_SURFACES:
            _materialize_query_surface(conn, sorted_db_paths, surface_name)


def read_counter_collection_rows(db_paths: list[str]) -> list[dict[str, Any]]:
    """Read rocpd counter collection rows using the normalized query."""
    return _read_query_rows(db_paths, COUNTERS_COLLECTION_QUERY)


def read_counter_collection_df(db_paths: list[str]) -> pd.DataFrame:
    """Read rocpd counter collection rows into a DataFrame."""
    return _read_query_dataframe(db_paths, COUNTERS_COLLECTION_QUERY)


def read_marker_api_trace_rows(db_paths: list[str]) -> list[dict[str, Any]]:
    """Read rocpd marker API trace rows using the normalized query."""
    return _read_query_rows(db_paths, MARKER_API_TRACE_QUERY)


def read_marker_api_trace_df(db_paths: list[str]) -> pd.DataFrame:
    """Read rocpd marker API trace rows into a DataFrame."""
    return _read_query_dataframe(db_paths, MARKER_API_TRACE_QUERY)


def count_counter_collection_rows(db_path: str) -> int:
    """Count rows exposed by the rocpd counters collection query."""
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(
            conn.execute(f"SELECT COUNT(*) FROM ({COUNTERS_COLLECTION_QUERY})")
        ) as cursor:
            row = cursor.fetchone()
            return int(row[0]) if row is not None else 0


def get_rocpd_pass_db_paths(workload_dir: Path) -> list[Path]:
    """Return root pass DBs matching this workload's perfmon pass configs."""
    pass_db_paths: list[Path] = []
    perfmon_dir = workload_dir / "perfmon"

    for pass_config_path in sorted(perfmon_dir.glob("pmc_perf*.yaml")):
        pass_db_path = workload_dir / f"{pass_config_path.stem}.db"
        if pass_db_path.is_file():
            pass_db_paths.append(pass_db_path)

    return pass_db_paths


def has_counter_collection_rows(db_path: Path) -> bool:
    """Return whether a rocpd pass DB exposes counter collection rows.

    Uses ``LIMIT 1`` instead of ``COUNT(*)`` to avoid materializing the
    full counters_collection view (which is JOIN-heavy and can take
    seconds per pass DB on large workloads).
    """
    if not db_path.is_file():
        return False

    try:
        with closing(sqlite3.connect(str(db_path))) as conn:
            with closing(
                conn.execute(f"SELECT 1 FROM ({COUNTERS_COLLECTION_QUERY}) LIMIT 1")
            ) as cursor:
                return cursor.fetchone() is not None
    except sqlite3.Error:
        return False


def has_rocpd_pass_counter_data(workload_dir: Path) -> bool:
    """Return whether all discovered rocpd pass DBs expose counter rows."""
    pass_db_paths = get_rocpd_pass_db_paths(workload_dir)
    return bool(pass_db_paths) and all(
        has_counter_collection_rows(path) for path in pass_db_paths
    )


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


def _read_query_rows(db_paths: list[str], query: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for db_path in sorted(db_paths):
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.execute(query)) as cursor:
                rows.extend(dict(row) for row in cursor.fetchall())

    return rows


def _read_query_dataframe(db_paths: list[str], query: str) -> pd.DataFrame:
    dataframes: list[pd.DataFrame] = []

    for db_path in sorted(db_paths):
        with closing(sqlite3.connect(db_path)) as conn:
            dataframe = pd.read_sql_query(query, conn)
            if not dataframe.empty:
                dataframes.append(dataframe)

    if not dataframes:
        return pd.DataFrame()
    return pd.concat(dataframes, ignore_index=True)


def _materialize_query_surface(
    conn: sqlite3.Connection,
    db_paths: list[str],
    surface_name: str,
) -> None:
    created_surface = False

    for index, db_path in enumerate(db_paths):
        attached_name = f"source_{index}"
        conn.execute(f"ATTACH DATABASE ? AS {attached_name}", (db_path,))
        try:
            if not created_surface:
                conn.execute(
                    f"CREATE TABLE {surface_name} AS "
                    f"SELECT * FROM {attached_name}.{surface_name} WHERE 0"
                )
                created_surface = True
            conn.execute(
                f"INSERT INTO {surface_name} "
                f"SELECT * FROM {attached_name}.{surface_name}"
            )
        except sqlite3.Error as e:
            if surface_name == "counters_collection":
                raise
            if "no such table" in str(e).lower():
                console_warning(
                    "rocpd",
                    f"Skipping optional {surface_name} surface in {db_path}: {e}",
                )
            else:
                raise
        finally:
            conn.commit()
            conn.execute(f"DETACH DATABASE {attached_name}")

    conn.commit()
