#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Create a minimal rocpd SQLite database for a given schema version.

This script is called by CTest execute tests defined in CMakeLists.txt; it is
not intended to be imported as a Python module.

Usage:
    python3 make_db.py --version 3.0.0 --output /path/to/schema_3_0_0.db

The database contains the minimum number of rows required to exercise all rocpd
conversion paths (CSV, Perfetto, OTF2) without a real GPU trace.
"""

import argparse
import os
import sqlite3
import sys

# ---------------------------------------------------------------------------
# Row-ID constants for the minimal test DB
# ---------------------------------------------------------------------------

_NID = 1
_PID = 100
_TID = 200
_AGENT_ID = 1
_QUEUE_ID = 1
_STREAM_ID = 1
_CODE_OBJ_ID = 1
_KERNEL_SYM_ID = 1
_T_START = 1_000_000_000
_T_END = 2_000_000_000

_STR_EMPTY = 1
_STR_KERNEL_NAME = 2
_STR_REGION_NAME = 3
_STR_MEM_COPY = 4
_STR_CATEGORY = 5
_STR_SPM = 6
_STR_HIP_EVENT_RECORD = 7

_EVT_REGION_1 = 1
_EVT_REGION_2 = 2
_EVT_REGION_3 = 3
_EVT_KERNEL_1 = 4
_EVT_KERNEL_2 = 5
_EVT_GRAPH_1 = 6
_EVT_GRAPH_2 = 7
_EVT_GRAPH_3 = 8
_EVT_HIP_EVENT_1 = 9

_PMC_NO_SPM_ID = 1
_PMC_SPM_ID = 2
_TRACK_SPM_ID = 1
_SAMPLE_SPM_ID = 1


# ---------------------------------------------------------------------------
# UUID / GUID helpers
# ---------------------------------------------------------------------------


def guid_for_version(version: str) -> str:
    """Return a deterministic GUID string for the given schema version string."""
    return f"00ecde8307b8_SCHEMA_{version.replace('.', '_')}"


def uuid_for_version(version: str) -> str:
    """Return a deterministic UUID string (GUID prefixed with '_')."""
    return f"_{guid_for_version(version)}"


# ---------------------------------------------------------------------------
# Minimal-data insertion
# ---------------------------------------------------------------------------


def insert_minimal_data(
    conn: sqlite3.Connection, uuid: str, guid: str, version: str
) -> None:
    """Insert minimum rows needed to exercise all rocpd conversion paths.

    Foreign-key enforcement is disabled so rows can be inserted in any order.
    Commit is the caller's responsibility.

    Note: ``region_name_id`` is set on ``rocpd_kernel_dispatch`` so that the
    ``kernels`` view INNER JOIN on ``rocpd_string`` returns rows.
    """

    def tbl(name: str) -> str:
        return f"`{name}{uuid}`"

    ver = tuple(int(x) for x in version.split("."))

    conn.execute("PRAGMA foreign_keys = OFF")

    conn.executemany(
        f"INSERT INTO {tbl('rocpd_string')} (id, guid, string) VALUES (?,?,?)",
        [
            (_STR_EMPTY, guid, ""),
            (_STR_KERNEL_NAME, guid, "myKernel"),
            (_STR_REGION_NAME, guid, "myRegion"),
            (_STR_MEM_COPY, guid, "Pageable to Device"),
            (_STR_CATEGORY, guid, "MARKER"),
            (_STR_SPM, guid, "SPM"),
            (_STR_HIP_EVENT_RECORD, guid, "HIP_EVENT_RECORD"),
        ],
    )

    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_node')} "
        "(id, guid, hash, machine_id, system_name, hostname, release, version, hardware_name, domain_name) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            _NID,
            guid,
            0,
            "test-machine",
            "Linux",
            "testhost",
            "6.8.0",
            "#1",
            "x86_64",
            "none",
        ),
    )

    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_process')} "
        "(id, guid, nid, ppid, pid, init, fini, start, end, command) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            _PID,
            guid,
            _NID,
            1,
            _PID,
            _T_START - 1_000_000,
            _T_END + 1_000_000,
            _T_START - 1_000_000,
            _T_END + 1_000_000,
            "test_program",
        ),
    )

    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_thread')} "
        "(id, guid, nid, ppid, pid, tid) VALUES (?,?,?,?,?,?)",
        (_TID, guid, _NID, 1, _PID, _TID),
    )

    conn.executemany(
        f"INSERT INTO {tbl('rocpd_info_agent')} "
        "(id, guid, nid, pid, type, absolute_index, logical_index, type_index) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (0, guid, _NID, _PID, "CPU", 0, 0, 0),
            (_AGENT_ID, guid, _NID, _PID, "GPU", 1, 0, 0),
        ],
    )

    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_queue')} (id, guid, nid, pid, name) VALUES (?,?,?,?,?)",
        (_QUEUE_ID, guid, _NID, _PID, "Queue 0"),
    )
    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_stream')} (id, guid, nid, pid, name) VALUES (?,?,?,?,?)",
        (_STREAM_ID, guid, _NID, _PID, "Stream 0"),
    )

    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_code_object')} "
        "(id, guid, nid, pid, agent_id, uri, load_base, load_size, load_delta) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (_CODE_OBJ_ID, guid, _NID, _PID, _AGENT_ID, "file:///test#offset=0", 0, 4096, 0),
    )
    conn.execute(
        f"INSERT INTO {tbl('rocpd_info_kernel_symbol')} "
        "(id, guid, nid, pid, code_object_id, kernel_name, display_name) "
        "VALUES (?,?,?,?,?,?,?)",
        (_KERNEL_SYM_ID, guid, _NID, _PID, _CODE_OBJ_ID, "myKernel.kd", "myKernel"),
    )

    for eid in (
        _EVT_REGION_1,
        _EVT_REGION_2,
        _EVT_REGION_3,
        _EVT_KERNEL_1,
        _EVT_KERNEL_2,
    ):
        conn.execute(
            f"INSERT INTO {tbl('rocpd_event')} "
            "(id, guid, category_id, stack_id, parent_stack_id, correlation_id) "
            "VALUES (?,?,?,?,?,?)",
            (eid, guid, _STR_CATEGORY, eid, 0, 0),
        )

    if ver >= (3, 0, 2):
        conn.executemany(
            f"INSERT INTO {tbl('rocpd_kernel_dispatch')} "
            "(id, guid, nid, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, workgroup_size_x, workgroup_size_y, workgroup_size_z, "
            "grid_size_x, grid_size_y, grid_size_z, graph_exec_id, graph_node_id, "
            "region_name_id, event_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    1,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _KERNEL_SYM_ID,
                    1,
                    _QUEUE_ID,
                    _STREAM_ID,
                    _T_START,
                    _T_START + 100_000,
                    256,
                    1,
                    1,
                    1024,
                    1,
                    1,
                    1,
                    1,
                    _STR_EMPTY,
                    _EVT_KERNEL_1,
                ),
                (
                    2,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _KERNEL_SYM_ID,
                    2,
                    _QUEUE_ID,
                    _STREAM_ID,
                    _T_START + 200_000,
                    _T_START + 300_000,
                    256,
                    1,
                    1,
                    1024,
                    1,
                    1,
                    2,
                    2,
                    _STR_EMPTY,
                    _EVT_KERNEL_2,
                ),
            ],
        )
    else:
        conn.executemany(
            f"INSERT INTO {tbl('rocpd_kernel_dispatch')} "
            "(id, guid, nid, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, workgroup_size_x, workgroup_size_y, workgroup_size_z, "
            "grid_size_x, grid_size_y, grid_size_z, region_name_id, event_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    1,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _KERNEL_SYM_ID,
                    1,
                    _QUEUE_ID,
                    _STREAM_ID,
                    _T_START,
                    _T_START + 100_000,
                    256,
                    1,
                    1,
                    1024,
                    1,
                    1,
                    _STR_EMPTY,
                    _EVT_KERNEL_1,
                ),
                (
                    2,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _KERNEL_SYM_ID,
                    2,
                    _QUEUE_ID,
                    _STREAM_ID,
                    _T_START + 200_000,
                    _T_START + 300_000,
                    256,
                    1,
                    1,
                    1024,
                    1,
                    1,
                    _STR_EMPTY,
                    _EVT_KERNEL_2,
                ),
            ],
        )

    conn.executemany(
        f"INSERT INTO {tbl('rocpd_region')} "
        "(id, guid, nid, pid, tid, start, end, name_id, event_id) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                1,
                guid,
                _NID,
                _PID,
                _TID,
                _T_START - 500_000,
                _T_END + 500_000,
                _STR_REGION_NAME,
                _EVT_REGION_1,
            ),
            (
                2,
                guid,
                _NID,
                _PID,
                _TID,
                _T_START - 100_000,
                _T_START + 500_000,
                _STR_REGION_NAME,
                _EVT_REGION_2,
            ),
            (
                3,
                guid,
                _NID,
                _PID,
                _TID,
                _T_START + 100_000,
                _T_END - 100_000,
                _STR_REGION_NAME,
                _EVT_REGION_3,
            ),
        ],
    )

    # graph_launch data was introduced in schema 3.0.2. Add some dummy data for it.
    if ver >= (3, 0, 2):
        for eid in (_EVT_GRAPH_1, _EVT_GRAPH_2, _EVT_GRAPH_3):
            conn.execute(
                f"INSERT INTO {tbl('rocpd_event')} "
                "(id, guid, category_id, stack_id, parent_stack_id, correlation_id) "
                "VALUES (?,?,?,?,?,?)",
                (eid, guid, _STR_CATEGORY, eid, 0, 0),
            )
        conn.executemany(
            f"INSERT INTO {tbl('rocpd_graph_launch')} "
            "(id, guid, nid, pid, tid, agent_id, queue_id, start, end, "
            "graph_exec_id, kernel_dispatch_count, event_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    1,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _QUEUE_ID,
                    _T_START,
                    _T_START + 200_000,
                    1,
                    2,
                    _EVT_GRAPH_1,
                ),
                (
                    2,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _QUEUE_ID,
                    _T_START + 500_000,
                    _T_START + 700_000,
                    2,
                    2,
                    _EVT_GRAPH_2,
                ),
                (
                    3,
                    guid,
                    _NID,
                    _PID,
                    _TID,
                    _AGENT_ID,
                    _QUEUE_ID,
                    _T_START + 1_000_000,
                    _T_START + 1_200_000,
                    3,
                    2,
                    _EVT_GRAPH_3,
                ),
            ],
        )

    # SPM counter data was introduced in schema 3.0.3.
    if ver >= (3, 0, 3):
        _SQ_WAVES_DESC = (
            "Count number of waves sent to distributed sequencers (SQs). "
            "This value represents the number of waves that are sent to each SQ."
        )

        conn.executemany(
            f"INSERT INTO {tbl('rocpd_info_pmc')} "
            "(id, guid, nid, pid, agent_id, target_arch, name, symbol, description, "
            "component, value_type, block, expression, is_constant, is_derived, "
            "spm_support, extdata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    _PMC_NO_SPM_ID,
                    guid,
                    _NID,
                    _PID,
                    _AGENT_ID,
                    "GPU",
                    "SQ_WAVES",
                    "SQ_WAVES",
                    _SQ_WAVES_DESC,
                    "rocm",
                    "ABS",
                    "SQ",
                    "",
                    0,
                    0,
                    None,  # spm_support = NULL (not SPM-capable)
                    "{}",
                ),
                (
                    _PMC_SPM_ID,
                    guid,
                    _NID,
                    _PID,
                    _AGENT_ID,
                    "GPU",
                    "SQ_WAVES",
                    "SQ_WAVES",
                    _SQ_WAVES_DESC,
                    "rocm",
                    "ABS",
                    "SQ",
                    "",
                    0,
                    0,
                    1,  # spm_support = 1 (SPM-capable)
                    "{}",
                ),
            ],
        )

        conn.execute(
            f"INSERT INTO {tbl('rocpd_track')} "
            "(id, guid, nid, pid, tid, name_id, extdata) "
            "VALUES (?,?,?,?,?,?,?)",
            (_TRACK_SPM_ID, guid, _NID, _PID, _TID, _STR_SPM, "{}"),
        )

        conn.execute(
            f"INSERT INTO {tbl('rocpd_sample')} "
            "(id, guid, track_id, timestamp, event_id, extdata) "
            "VALUES (?,?,?,?,?,?)",
            (_SAMPLE_SPM_ID, guid, _TRACK_SPM_ID, _T_START + 50_000, _EVT_KERNEL_1, "{}"),
        )

        conn.executemany(
            f"INSERT INTO {tbl('rocpd_pmc_event')} "
            "(id, guid, event_id, sample_id, pmc_id, value, xcc, shader_engine, instance) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (1, guid, _EVT_KERNEL_1, _SAMPLE_SPM_ID, _PMC_SPM_ID, 16.0, 0, 0, 0),
                (2, guid, _EVT_KERNEL_1, _SAMPLE_SPM_ID, _PMC_SPM_ID, 16.0, 0, 1, 0),
            ],
        )

    # HIP event barrier data was introduced in schema 3.0.4.
    if ver >= (3, 0, 4):
        conn.execute(
            f"INSERT INTO {tbl('rocpd_event')} "
            "(id, guid, category_id, stack_id, parent_stack_id, correlation_id) "
            "VALUES (?,?,?,?,?,?)",
            (_EVT_HIP_EVENT_1, guid, _STR_CATEGORY, _EVT_HIP_EVENT_1, 0, 1),
        )
        conn.execute(
            f"INSERT INTO {tbl('rocpd_hip_event')} "
            "(id, guid, nid, pid, tid, start, end, name_id, agent_id, queue_id, stream_id, "
            "hip_event_handle, source_queue_id, event_id, extdata) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                1,
                guid,
                _NID,
                _PID,
                _TID,
                _T_START,
                _T_START + 50_000,
                _STR_HIP_EVENT_RECORD,
                _AGENT_ID,
                _QUEUE_ID,
                _STREAM_ID,
                123456789,  # dummy hip_event_handle pointer value
                _QUEUE_ID,  # source_queue_id == queue_id for a RECORD operation
                _EVT_HIP_EVENT_1,
                "{}",
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Schema version, e.g. 3.0.0")
    parser.add_argument(
        "--output", required=True, help="Destination path for the .db file"
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    from rocpd.schema import RocpdSchema

    guid = guid_for_version(args.version)
    uuid = uuid_for_version(args.version)

    schema = RocpdSchema(uuid=uuid, guid=guid, version=args.version)

    conn = sqlite3.connect(output_path)
    schema.write_schema(conn)
    conn.close()

    conn = sqlite3.connect(output_path)
    insert_minimal_data(conn, uuid, guid, args.version)
    conn.commit()
    conn.close()

    print(f"Created: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
