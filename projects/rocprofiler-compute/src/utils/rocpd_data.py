# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import shutil
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from pathlib import Path
from typing import Optional, Union

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


def query_counter_collection(db_path: str) -> Iterator[tuple]:
    """Stream rows of the ``counters_collection`` table from one rocpd db.

    The first tuple yielded is the column header (names from the SELECT). All
    subsequent tuples are data rows. Yields nothing if the table is missing or
    the query returns no description.
    """
    yield from _stream_query(db_path, COUNTERS_COLLECTION_QUERY)


def query_marker_trace(db_path: str) -> Iterator[tuple]:
    """Stream rows of the marker-API ``regions`` table from one rocpd db.

    Same shape as :func:`query_counter_collection`: header tuple first, then
    data rows.
    """
    yield from _stream_query(db_path, MARKER_API_TRACE_QUERY)


def query_pmc_event_table(db_path: str) -> Optional[tuple[str, str]]:
    """Return ``(table_name, guid)`` for the ``rocpd_pmc_event_<guid>`` table.

   Returns ``None`` if no matching table exists in the db.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(
            conn.execute(
                TABLE_NAME_PREFIX_QUERY.format(
                    table_name_prefix=ROCPD_PMC_EVENT_TABLE_NAME_PREFIX
                )
            )
        ) as cursor:
            row = cursor.fetchone()
    if row is None:
        return None
    table_name = row[0]
    guid = table_name[len(ROCPD_PMC_EVENT_TABLE_NAME_PREFIX) :].replace("_", "-")
    return table_name, guid


def query_dispatch_to_event_map(db_path: str, guid: str) -> dict[str, str]:
    """Return ``{dispatch_id: event_id}`` from ``rocpd_kernel_dispatch`` for one guid.

    Both keys and values are stringified to align with CSV-derived inputs in the
    orchestrator. Returns an empty dict if the query yields no rows.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.execute(KERNEL_DISPATCH_QUERY, (guid,))) as cursor:
            rows = cursor.fetchall()
    return {str(dispatch_id): str(event_id) for dispatch_id, event_id, _ in rows}


def insert_pmc_events(
    db_path: str,
    table_name: str,
    rows: Iterable[tuple],
) -> None:
    """Bulk-INSERT pmc_event rows into the named table.

    ``rows`` must be an iterable of ``(guid, event_id, pmc_id, value)`` tuples.
    """
    columns = ("guid", "event_id", "pmc_id", "value")
    placeholders = ", ".join(["?"] * len(columns))
    statement = INSERT_QUERY.format(
        table_name=table_name,
        columns=", ".join(columns),
        placeholders=placeholders,
    )
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.executemany(statement, rows)


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


def _stream_query(db_path: str, query: str) -> Iterator[tuple]:
    """Yield header tuple followed by data rows for ``query`` against one db.

    Yields nothing if the query has no description (e.g. table missing). The
    sqlite connection and cursor are closed when the generator is exhausted or
    closed by the caller.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        with closing(conn.execute(query)) as cursor:
            if cursor.description is None:
                return
            yield tuple(d[0] for d in cursor.description)
            yield from cursor
