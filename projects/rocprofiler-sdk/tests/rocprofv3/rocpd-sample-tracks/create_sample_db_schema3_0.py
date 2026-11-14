#!/usr/bin/env python3

# MIT License
#
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Create a synthetic rocpd schema 3.0 SQLite database containing PMC sample
track data for two counters:

  - CPU: symbol="MemUsg"  (CPU Memory Usage, units=MB)
  - GPU: symbol="Temp"    (GPU Temperature, units=°C)

This script is tightly coupled to schema 3.0 (rocpd_tables.sql / rocpd_views.sql
/ data_views.sql).  The _UUID suffix used for physical table names must be
updated whenever the schema version is bumped.

The constant lists TIMESTAMPS_NS, CPU_MEMORY_MB, and GPU_TEMP_C are imported
directly by validate.py so that both files share a single source of truth.
"""

import argparse
import os
import sqlite3

# ---------------------------------------------------------------------------
# Deterministic test data — shared with validate.py via direct import
# ---------------------------------------------------------------------------

_BASE_TS_NS = 1_000_000_000  # 1 second expressed in nanoseconds
_PERIOD_NS = 100_000_000  # 100 ms sampling period in nanoseconds

TIMESTAMPS_NS = [_BASE_TS_NS + i * _PERIOD_NS for i in range(20)]

# CPU memory allocated (MB): starts at ~1 MB, fluctuates upward to 3 MB
CPU_MEMORY_MB = [
    1.00,
    1.12,
    1.28,
    1.19,
    1.45,
    1.67,
    1.58,
    1.89,
    2.03,
    1.95,
    2.14,
    2.31,
    2.47,
    2.39,
    2.52,
    2.68,
    2.57,
    2.74,
    2.88,
    3.00,
]

# GPU temperature (°C): starts at 30 °C, fluctuates upward to 75 °C
GPU_TEMP_C = [
    30.0,
    32.5,
    35.0,
    33.5,
    38.0,
    42.0,
    40.5,
    45.0,
    48.0,
    46.5,
    51.0,
    54.5,
    53.0,
    57.0,
    59.5,
    58.0,
    63.0,
    67.5,
    71.0,
    75.0,
]

# ---------------------------------------------------------------------------
# Schema 3.0 identifiers
# ---------------------------------------------------------------------------

# Physical table names are rocpd_<table>_test (uuid suffix = "_test").
# If the schema version is bumped, rename this file and update _UUID.
_UUID = "_test001"
_GUID = "test001"


def _tbl(logical_name):
    """Return the physical table name for schema 3.0 (with _UUID suffix)."""
    return f"{logical_name}{_UUID}"


# ---------------------------------------------------------------------------
# Database creation
# ---------------------------------------------------------------------------


def create_db(output_dir):
    """Create sample_tracks_schema3_0.db inside *output_dir*."""
    # Imported here so that validate.py can import the constant lists at the top
    # of this module without requiring rocpd to be installed in the pytest venv.
    from rocpd.schema import RocpdSchema  # noqa: PLC0415

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "sample_tracks_schema3_0.db")

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Apply the full rocpd schema 3.0 (tables + indexes + views).
        # RocpdSchema enables PRAGMA foreign_keys = ON, so we insert in FK order.
        schema = RocpdSchema(uuid=_UUID, guid=_GUID)
        schema.write_schema(conn)

        _insert_test_data(conn)
        conn.commit()
    finally:
        conn.close()

    print(f"Created: {db_path}")
    return db_path


def _insert_test_data(conn):
    """Populate the physical tables with synthetic PMC sample data."""

    def last_id():
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # 1. Node (required root for all FK chains)
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_node')}`"
        " (guid, hash, machine_id, system_name, hostname)"
        " VALUES (?, ?, ?, ?, ?)",
        (_GUID, 1, "test-machine-id-1", "Linux", "test-host"),
    )
    node_id = last_id()

    # 2. Process — two constraints:
    #    a) command must be non-empty: write_perfetto calls command_line.front() on it.
    #    b) pmc_events_schema_3_0 exposes rocpd_info_pmc.pid (FK → rocpd_info_process.id)
    #       while select_guid_nid_pid() filters by pitr.pid (rocpd_info_process.pid, the
    #       OS pid).  Setting id = pid = 12345 makes both fields equal so the WHERE clause
    #       matches without any changes to libpyrocpd.cpp.
    _PROC_PID = 12345
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_process')}`"
        " (id, guid, nid, pid, command) VALUES (?, ?, ?, ?, ?)",
        (_PROC_PID, _GUID, node_id, _PROC_PID, "test_app"),
    )
    proc_id = _PROC_PID

    # 3. Agents: one CPU (absolute_index=0) and one GPU (absolute_index=1)
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_agent')}`"
        " (guid, nid, pid, type, absolute_index, logical_index, type_index)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_GUID, node_id, proc_id, "CPU", 0, 0, 0),
    )
    cpu_agent_id = last_id()

    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_agent')}`"
        " (guid, nid, pid, type, absolute_index, logical_index, type_index)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (_GUID, node_id, proc_id, "GPU", 1, 0, 0),
    )
    gpu_agent_id = last_id()

    # 4. PMC info
    #    symbol="MemUsg" → short_description = "<agent_type> Memory Usage" (CASE branch)
    #    symbol="Temp"   → short_description = "<agent_type> Temperature"  (CASE branch)
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_pmc')}`"
        " (guid, nid, pid, agent_id, name, symbol, description, units)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _GUID,
            node_id,
            proc_id,
            cpu_agent_id,
            "cpu_memory_allocated",
            "MemUsg",
            "CPU Memory Allocated",
            "MB",
        ),
    )
    cpu_pmc_id = last_id()

    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_pmc')}`"
        " (guid, nid, pid, agent_id, name, symbol, description, units)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _GUID,
            node_id,
            proc_id,
            gpu_agent_id,
            "gpu_temperature",
            "Temp",
            "GPU Temperature",
            "°C",
        ),
    )
    gpu_pmc_id = last_id()

    # 5. Thread — required so write_perfetto populates thread_tracks and regions can
    #    reference a valid tid.  Use an OS tid distinct from the PID (12345).
    _OS_TID = 12346
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_info_thread')}`"
        " (guid, nid, pid, tid, name) VALUES (?, ?, ?, ?, ?)",
        (_GUID, node_id, proc_id, _OS_TID, "main"),
    )
    thread_id = last_id()

    # 6. Strings: event category + track names + region name
    #    Track name strings must match PMC_I.name or PMC_I.name_plain in the
    #    samples_schema_3_0 LEFT JOIN condition.  Using the plain name (no GPU
    #    index suffix) works for both agents via name_plain matching.
    def insert_string(s):
        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_string')}` (guid, string) VALUES (?, ?)",
            (_GUID, s),
        )
        return last_id()

    category_id = insert_string("pmc_sample")
    cpu_name_id = insert_string("cpu_memory_allocated")
    gpu_name_id = insert_string("gpu_temperature")
    region_name_id = insert_string("test_region")
    region_cat_id = insert_string("user_region")

    # 7. Tracks (PMC tracks carry no thread — tid is NULL, handled by LEFT OUTER JOIN)
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_track')}` (guid, nid, pid, name_id) VALUES (?, ?, ?, ?)",
        (_GUID, node_id, proc_id, cpu_name_id),
    )
    cpu_track_id = last_id()

    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_track')}` (guid, nid, pid, name_id) VALUES (?, ?, ?, ?)",
        (_GUID, node_id, proc_id, gpu_name_id),
    )
    gpu_track_id = last_id()

    # 8. Events, pmc_events, and samples — 20 per agent, inserted together per timestamp
    for ts, cpu_val, gpu_val in zip(TIMESTAMPS_NS, CPU_MEMORY_MB, GPU_TEMP_C):
        # CPU
        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_event')}` (guid, category_id) VALUES (?, ?)",
            (_GUID, category_id),
        )
        cpu_event_id = last_id()

        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_pmc_event')}`"
            " (guid, event_id, pmc_id, value) VALUES (?, ?, ?, ?)",
            (_GUID, cpu_event_id, cpu_pmc_id, cpu_val),
        )
        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_sample')}`"
            " (guid, track_id, timestamp, event_id) VALUES (?, ?, ?, ?)",
            (_GUID, cpu_track_id, ts, cpu_event_id),
        )

        # GPU
        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_event')}` (guid, category_id) VALUES (?, ?)",
            (_GUID, category_id),
        )
        gpu_event_id = last_id()

        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_pmc_event')}`"
            " (guid, event_id, pmc_id, value) VALUES (?, ?, ?, ?)",
            (_GUID, gpu_event_id, gpu_pmc_id, gpu_val),
        )
        conn.execute(
            f"INSERT INTO `{_tbl('rocpd_sample')}`"
            " (guid, track_id, timestamp, event_id) VALUES (?, ?, ?, ?)",
            (_GUID, gpu_track_id, ts, gpu_event_id),
        )

    # 9. Region — spans the full sampling window; references the thread so that
    #    write_perfetto can look up thread_tracks[tid] without throwing.
    _region_start = TIMESTAMPS_NS[0]
    _region_end = TIMESTAMPS_NS[-1] + _PERIOD_NS
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_event')}` (guid, category_id) VALUES (?, ?)",
        (_GUID, region_cat_id),
    )
    region_event_id = last_id()
    conn.execute(
        f"INSERT INTO `{_tbl('rocpd_region')}`"
        " (guid, nid, pid, tid, start, end, name_id, event_id)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            _GUID,
            node_id,
            proc_id,
            thread_id,
            _region_start,
            _region_end,
            region_name_id,
            region_event_id,
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a synthetic rocpd schema 3.0 SQLite database with PMC sample "
            "tracks for CPU memory usage and GPU temperature."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory in which sample_tracks_schema3_0.db will be written.",
    )
    args = parser.parse_args()
    create_db(args.output_dir)


if __name__ == "__main__":
    main()
