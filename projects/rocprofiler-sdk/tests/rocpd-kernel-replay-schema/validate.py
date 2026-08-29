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

"""Schema-level checks for how rocpd represents kernel replay.

Kernel replay executes one dispatch once per counter group. Every pass reports the same
dispatch_id, so a schema that treats dispatch_id as the identity of an execution merges passes
that are not the same execution. These tests drive the shipped SQL directly against SQLite, with
no GPU and no profiling run, and cover the two things the schema has to get right:

  * a counter that appears in several counter groups must not be summed across the passes that
    collected it, and
  * a run that did not use kernel replay must produce exactly what it produced before.

The second is checked by building the same data under the previous schema version and comparing.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest

# The dispatch under test is replayed across three counter groups. SQ_WAVES is repeated in every
# group as a sanity counter, which is the usual way these are written and the case that goes wrong:
# it is collected three times, once per pass, and each collection reports the same 512 waves.
REPLAY_PASSES = 3
SQ_WAVES_PER_PASS = 512.0

# One counter unique to each group, to confirm the fix does not disturb counters that were never
# affected.
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
    # counters_collection joins the thread table, so a run with no thread row yields no rows at
    # all rather than the rows under test.
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
    conn: sqlite3.Connection,
    *,
    row_id: int,
    event_id: int,
    dispatch_id: int,
    replay_pass,
    start: int,
    supports_replay_pass: bool,
) -> None:
    """Insert one dispatch execution and the event row that represents it."""

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO rocpd_event (id, guid, category_id, stack_id, parent_stack_id, correlation_id)"
        " VALUES (?, 'test-guid', 1, ?, ?, ?)",
        (event_id, dispatch_id, dispatch_id, dispatch_id),
    )

    columns = [
        "id",
        "guid",
        "nid",
        "pid",
        "tid",
        "agent_id",
        "kernel_id",
        "dispatch_id",
        "queue_id",
        "stream_id",
        "start",
        "end",
        "workgroup_size_x",
        "workgroup_size_y",
        "workgroup_size_z",
        "grid_size_x",
        "grid_size_y",
        "grid_size_z",
        "event_id",
    ]
    values = [
        row_id,
        "test-guid",
        1,
        1,
        1,
        1,
        1,
        dispatch_id,
        1,
        1,
        start,
        start + 100,
        64,
        1,
        1,
        512,
        1,
        1,
        event_id,
    ]
    if supports_replay_pass:
        columns.append("replay_pass")
        values.append(replay_pass)

    placeholders = ", ".join("?" for _ in values)
    cur.execute(
        f"INSERT INTO rocpd_kernel_dispatch ({', '.join(columns)}) VALUES ({placeholders})",
        values,
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


def populate_replay_run(conn: sqlite3.Connection, *, supports_replay_pass: bool) -> None:
    """One dispatch replayed over three counter groups that all share SQ_WAVES.

    Under the previous schema the passes could not be told apart, so all three executions had to
    share a single event. That is the shape this reproduces when supports_replay_pass is False.
    """

    counter_names = ["SQ_WAVES"] + GROUP_UNIQUE_COUNTERS
    insert_common_rows(conn, counter_names)

    pmc_ids = {name: idx for idx, name in enumerate(counter_names, start=1)}
    counter_row_id = 1

    for pass_index in range(REPLAY_PASSES):
        event_id = pass_index + 1 if supports_replay_pass else 1
        if supports_replay_pass or pass_index == 0:
            insert_dispatch(
                conn,
                row_id=pass_index + 1,
                event_id=event_id,
                dispatch_id=1,
                replay_pass=pass_index,
                start=1000 + (pass_index * 1000),
                supports_replay_pass=supports_replay_pass,
            )

        insert_counter(
            conn,
            row_id=counter_row_id,
            event_id=event_id,
            pmc_id=pmc_ids["SQ_WAVES"],
            value=SQ_WAVES_PER_PASS,
        )
        counter_row_id += 1

        unique_name = GROUP_UNIQUE_COUNTERS[pass_index]
        insert_counter(
            conn,
            row_id=counter_row_id,
            event_id=event_id,
            pmc_id=pmc_ids[unique_name],
            value=100.0 * (pass_index + 1),
        )
        counter_row_id += 1


def populate_plain_run(conn: sqlite3.Connection, *, supports_replay_pass: bool) -> None:
    """Three ordinary dispatches, no replay: the case that has to stay bit-for-bit identical."""

    counter_names = ["SQ_WAVES", "SQ_INSTS_VALU"]
    insert_common_rows(conn, counter_names)

    counter_row_id = 1
    for dispatch_id in range(1, 4):
        insert_dispatch(
            conn,
            row_id=dispatch_id,
            event_id=dispatch_id,
            dispatch_id=dispatch_id,
            replay_pass=None,
            start=1000 * dispatch_id,
            supports_replay_pass=supports_replay_pass,
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


@pytest.fixture
def previous_schema(current_schema):
    previous = current_schema / "versions" / "3.0.3"
    if not (previous / "rocpd_tables.sql").exists():
        pytest.skip("3.0.3 schema snapshot is not available")
    return previous


def test_kernel_dispatch_has_replay_pass_column(current_schema):
    """The pass index has somewhere to live."""

    conn = build_database(current_schema)
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(rocpd_kernel_dispatch)").fetchall()
    }
    assert "replay_pass" in columns


def test_shared_counter_is_not_multiplied_across_passes(current_schema):
    """SQ_WAVES appears in all three groups and must still read 512 per pass, not 1536 once."""

    conn = build_database(current_schema)
    populate_replay_run(conn, supports_replay_pass=True)

    rows = conn.execute(
        "SELECT replay_pass, value FROM counters_collection"
        " WHERE counter_name = 'SQ_WAVES' ORDER BY replay_pass"
    ).fetchall()

    assert [row[0] for row in rows] == list(range(REPLAY_PASSES))
    assert [row[1] for row in rows] == [SQ_WAVES_PER_PASS] * REPLAY_PASSES

    total = sum(row[1] for row in rows)
    assert total == SQ_WAVES_PER_PASS * REPLAY_PASSES, (
        "the per-pass values should still add up to what the old view reported as a single row; "
        "the fix separates them rather than discarding any"
    )


def test_old_schema_reproduces_the_multiplication(previous_schema):
    """Guards the claim the fix rests on: the previous schema really does report 1536."""

    conn = build_database(previous_schema)
    populate_replay_run(conn, supports_replay_pass=False)

    rows = conn.execute(
        "SELECT value FROM counters_collection WHERE counter_name = 'SQ_WAVES'"
    ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == SQ_WAVES_PER_PASS * REPLAY_PASSES


def test_group_unique_counters_are_unaffected(current_schema):
    """Counters that appear in only one group were never multiplied and must not change."""

    conn = build_database(current_schema)
    populate_replay_run(conn, supports_replay_pass=True)

    for pass_index, name in enumerate(GROUP_UNIQUE_COUNTERS):
        rows = conn.execute(
            "SELECT replay_pass, value FROM counters_collection WHERE counter_name = ?",
            (name,),
        ).fetchall()
        assert rows == [(pass_index, 100.0 * (pass_index + 1))]


def test_each_pass_keeps_its_own_dispatch_row(current_schema):
    """Each pass is a real execution with its own timing, so it gets its own dispatch row."""

    conn = build_database(current_schema)
    populate_replay_run(conn, supports_replay_pass=True)

    rows = conn.execute(
        "SELECT replay_pass, start FROM rocpd_kernel_dispatch"
        " WHERE dispatch_id = 1 ORDER BY replay_pass"
    ).fetchall()

    assert [row[0] for row in rows] == list(range(REPLAY_PASSES))
    assert len({row[1] for row in rows}) == REPLAY_PASSES


def test_plain_run_matches_previous_schema(current_schema, previous_schema):
    """A run without kernel replay must report exactly what the previous schema reported."""

    columns = (
        "dispatch_id, kernel_id, event_id, correlation_id, pid, tid, agent_id, queue_id,"
        " grid_size, kernel_name, workgroup_size, counter_name, value, start, end, duration"
    )
    query = (
        f"SELECT {columns} FROM counters_collection ORDER BY dispatch_id, counter_name"
    )

    current = build_database(current_schema)
    populate_plain_run(current, supports_replay_pass=True)

    previous = build_database(previous_schema)
    populate_plain_run(previous, supports_replay_pass=False)

    assert current.execute(query).fetchall() == previous.execute(query).fetchall()


def test_plain_run_leaves_replay_pass_null(current_schema):
    """replay_pass carries no information outside kernel replay and stays NULL there."""

    conn = build_database(current_schema)
    populate_plain_run(conn, supports_replay_pass=True)

    rows = conn.execute(
        "SELECT COUNT(*) FROM rocpd_kernel_dispatch WHERE replay_pass IS NOT NULL"
    ).fetchone()
    assert rows[0] == 0


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
