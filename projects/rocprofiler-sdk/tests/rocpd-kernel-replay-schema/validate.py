#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Schema-level checks for kernel replay counter values in rocpd.

Kernel replay executes one dispatch once per counter group and every pass reports the same
dispatch_id. The counter readings are stored, and counters_collection reports their average so a
sanity counter repeated in every group is not multiplied by the pass count. These tests drive the
shipped SQL against SQLite, with no GPU and no profiling run.
"""

import os
import sqlite3
from pathlib import Path

import pytest

# SQ_WAVES is repeated in every group. Each pass reports 512 waves.
REPLAY_PASSES = 3
SQ_WAVES_PER_PASS = 512.0

# One counter unique to each group. Those were never multiplied.
GROUP_UNIQUE_COUNTERS = ["GRBM_COUNT", "GRBM_GUI_ACTIVE", "SQ_INSTS_SALU"]


def schema_dir() -> Path:
    """Directory holding the current schema SQL, from the build tree or the source tree."""

    env_path = os.environ.get("ROCPD_SCHEMA_PATH")
    if env_path and (Path(env_path) / "rocpd_tables.sql").exists():
        return Path(env_path)

    source_dir = (
        Path(__file__).resolve().parents[2] / "source" / "share" / "rocprofiler-sdk-rocpd"
    )
    if (source_dir / "rocpd_tables.sql").exists():
        return source_dir

    pytest.skip("could not locate the rocpd schema SQL")


def build_database(version_dir: Path) -> sqlite3.Connection:
    """Create an in-memory database from the schema SQL in the given directory."""

    conn = sqlite3.connect(":memory:")
    for filename in ("rocpd_tables.sql", "rocpd_views.sql", "data_views.sql"):
        sql = (version_dir / filename).read_text()
        # The shipped SQL is a template: {{uuid}} suffixes the physical table names for merged
        # databases and {{guid}} tags the rows with their source. A single unmerged database uses
        # an empty uuid, which is what rocprofv3 itself writes.
        sql = sql.replace("{{uuid}}", "").replace("{{guid}}", "test-guid")
        conn.executescript(sql)
    return conn


def insert_common_rows(conn: sqlite3.Connection, counter_names) -> None:
    """Insert the node/process/agent/kernel rows every counter row has to join against."""

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO rocpd_info_node (id, guid, hash, machine_id, hostname)"
        " VALUES (1, 'test-guid', 1, 'machine', 'host')"
    )
    cur.execute(
        "INSERT INTO rocpd_info_process (id, guid, nid, pid) VALUES (1, 'test-guid', 1, 1000)"
    )
    cur.execute(
        "INSERT INTO rocpd_info_thread (id, guid, nid, pid, tid)"
        " VALUES (1, 'test-guid', 1, 1, 2000)"
    )
    cur.execute(
        "INSERT INTO rocpd_info_agent"
        " (id, guid, nid, pid, type, absolute_index, logical_index, type_index, name)"
        " VALUES (1, 'test-guid', 1, 1, 'GPU', 0, 0, 0, 'agent-0')"
    )
    cur.execute(
        "INSERT INTO rocpd_info_queue (id, guid, nid, pid, name)"
        " VALUES (1, 'test-guid', 1, 1, 'queue-0')"
    )
    cur.execute(
        "INSERT INTO rocpd_info_stream (id, guid, nid, pid, name)"
        " VALUES (1, 'test-guid', 1, 1, 'stream-0')"
    )
    cur.execute(
        "INSERT INTO rocpd_info_code_object (id, guid, nid, pid, agent_id, storage_type)"
        " VALUES (1, 'test-guid', 1, 1, 1, 'FILE')"
    )
    cur.execute(
        "INSERT INTO rocpd_info_kernel_symbol"
        " (id, guid, nid, pid, code_object_id, kernel_name, display_name,"
        "  sgpr_count, arch_vgpr_count, accum_vgpr_count)"
        " VALUES (1, 'test-guid', 1, 1, 1, 'saxpy', 'saxpy', 16, 8, 4)"
    )
    cur.execute(
        "INSERT INTO rocpd_string (id, guid, string) VALUES (1, 'test-guid', 'KERNEL')"
    )

    for pmc_id, name in enumerate(counter_names, start=1):
        cur.execute(
            "INSERT INTO rocpd_info_pmc"
            " (id, guid, nid, pid, agent_id, target_arch, name, symbol, value_type,"
            "  block, component, is_constant, is_derived)"
            " VALUES (?, 'test-guid', 1, 1, 1, 'GPU', ?, ?, 'ABS', 'SQ', 'gfx', 0, 0)",
            (pmc_id, name, name),
        )
    conn.commit()


def insert_dispatch(
    conn: sqlite3.Connection, *, row_id: int, event_id: int, dispatch_id: int, start: int
) -> None:
    """Insert one dispatch and the event row the counter rows attach to."""

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO rocpd_event (id, guid, category_id, stack_id, parent_stack_id, correlation_id)"
        " VALUES (?, 'test-guid', 1, ?, ?, ?)",
        (event_id, dispatch_id, dispatch_id, dispatch_id),
    )
    cur.execute(
        "INSERT INTO rocpd_kernel_dispatch ("
        " id, guid, nid, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id,"
        " start, end, workgroup_size_x, workgroup_size_y, workgroup_size_z,"
        " grid_size_x, grid_size_y, grid_size_z, event_id"
        ") VALUES (?, 'test-guid', 1, 1, 1, 1, 1, ?, 1, 1, ?, ?, 64, 1, 1, 512, 1, 1, ?)",
        (row_id, dispatch_id, start, start + 100, event_id),
    )
    conn.commit()


def insert_counter(
    conn: sqlite3.Connection, *, row_id: int, event_id: int, pmc_id: int, value: float
) -> None:
    conn.execute(
        "INSERT INTO rocpd_pmc_event (id, guid, event_id, pmc_id, value)"
        " VALUES (?, 'test-guid', ?, ?, ?)",
        (row_id, event_id, pmc_id, value),
    )
    conn.commit()


def populate_replay_run(conn: sqlite3.Connection) -> None:
    """One dispatch, three passes, all counter rows on the first event.

    That is the shape the export writes: later passes do not get their own dispatch row, and
    every reading is attached to the event recorded for that dispatch_id.
    """

    counter_names = ["SQ_WAVES"] + GROUP_UNIQUE_COUNTERS
    insert_common_rows(conn, counter_names)
    insert_dispatch(conn, row_id=1, event_id=1, dispatch_id=1, start=1000)

    pmc_ids = {name: idx for idx, name in enumerate(counter_names, start=1)}
    counter_row_id = 1
    for pass_index in range(REPLAY_PASSES):
        insert_counter(
            conn,
            row_id=counter_row_id,
            event_id=1,
            pmc_id=pmc_ids["SQ_WAVES"],
            value=SQ_WAVES_PER_PASS,
        )
        counter_row_id += 1
        insert_counter(
            conn,
            row_id=counter_row_id,
            event_id=1,
            pmc_id=pmc_ids[GROUP_UNIQUE_COUNTERS[pass_index]],
            value=100.0 * (pass_index + 1),
        )
        counter_row_id += 1


def populate_plain_run(conn: sqlite3.Connection) -> None:
    """Three ordinary dispatches, one reading of each counter. AVG must equal that reading."""

    counter_names = ["SQ_WAVES", "SQ_INSTS_VALU"]
    insert_common_rows(conn, counter_names)
    counter_row_id = 1
    for dispatch_id in range(1, 4):
        insert_dispatch(
            conn,
            row_id=dispatch_id,
            event_id=dispatch_id,
            dispatch_id=dispatch_id,
            start=1000 * dispatch_id,
        )
        for pmc_id, value in ((1, 512.0 * dispatch_id), (2, 64.0 * dispatch_id)):
            insert_counter(
                conn,
                row_id=counter_row_id,
                event_id=dispatch_id,
                pmc_id=pmc_id,
                value=value,
            )
            counter_row_id += 1


@pytest.fixture
def current_schema():
    return schema_dir()


def test_repeated_counter_is_averaged(current_schema):
    """SQ_WAVES in all three groups reads 512 once, not 1536."""

    conn = build_database(current_schema)
    populate_replay_run(conn)
    rows = conn.execute(
        "SELECT value FROM counters_collection WHERE counter_name = 'SQ_WAVES'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == SQ_WAVES_PER_PASS


def test_raw_readings_remain(current_schema):
    """The view averages. It does not delete the per-pass readings."""

    conn = build_database(current_schema)
    populate_replay_run(conn)
    rows = conn.execute(
        "SELECT PMC_E.value FROM rocpd_pmc_event PMC_E"
        " JOIN rocpd_info_pmc PMC_I ON PMC_I.id = PMC_E.pmc_id"
        " WHERE PMC_I.name = 'SQ_WAVES' ORDER BY PMC_E.id"
    ).fetchall()
    assert [row[0] for row in rows] == [SQ_WAVES_PER_PASS] * REPLAY_PASSES


def test_sum_of_the_same_rows_is_the_old_inflated_value(current_schema):
    """Documents the number SUM used to report for this shape."""

    conn = build_database(current_schema)
    populate_replay_run(conn)
    total = conn.execute(
        "SELECT SUM(PMC_E.value) FROM rocpd_pmc_event PMC_E"
        " JOIN rocpd_info_pmc PMC_I ON PMC_I.id = PMC_E.pmc_id"
        " WHERE PMC_I.name = 'SQ_WAVES'"
    ).fetchone()[0]
    assert total == SQ_WAVES_PER_PASS * REPLAY_PASSES


def test_group_unique_counters_are_unaffected(current_schema):
    """A counter in only one group is collected once, so AVG does not change it."""

    conn = build_database(current_schema)
    populate_replay_run(conn)
    for pass_index, name in enumerate(GROUP_UNIQUE_COUNTERS):
        rows = conn.execute(
            "SELECT value FROM counters_collection WHERE counter_name = ?",
            (name,),
        ).fetchall()
        assert rows == [(100.0 * (pass_index + 1),)]


def test_different_readings_of_one_counter_still_sum(current_schema):
    """Hardware instances of one counter differ, so the view still adds them."""

    conn = build_database(current_schema)
    insert_common_rows(conn, ["SQ_WAVES"])
    insert_dispatch(conn, row_id=1, event_id=1, dispatch_id=1, start=1000)
    insert_counter(conn, row_id=1, event_id=1, pmc_id=1, value=100.0)
    insert_counter(conn, row_id=2, event_id=1, pmc_id=1, value=300.0)
    rows = conn.execute(
        "SELECT value FROM counters_collection WHERE counter_name = 'SQ_WAVES'"
    ).fetchall()
    assert rows == [(400.0,)]


def test_single_reading_is_unchanged(current_schema):
    """A dispatch with one reading of a counter reports that reading."""

    conn = build_database(current_schema)
    populate_plain_run(conn)
    rows = conn.execute(
        "SELECT dispatch_id, counter_name, value FROM counters_collection"
        " ORDER BY dispatch_id, counter_name"
    ).fetchall()
    assert rows == [
        (1, "SQ_INSTS_VALU", 64.0),
        (1, "SQ_WAVES", 512.0),
        (2, "SQ_INSTS_VALU", 128.0),
        (2, "SQ_WAVES", 1024.0),
        (3, "SQ_INSTS_VALU", 192.0),
        (3, "SQ_WAVES", 1536.0),
    ]
