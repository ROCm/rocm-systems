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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

import pytest


def _count_rows(conn, table_or_view):
    return conn.execute(f"SELECT COUNT(*) FROM {table_or_view}").fetchone()[0]


def _tool(json_data):
    return json_data["rocprofiler-sdk-tool"]


def _json_records(json_data, key):
    records = _tool(json_data)["buffer_records"].get(key, [])
    return list(records) if records else []


# ---------------------------------------------------------------------------
# JSON <-> decoded-view field mapping.
#
# The JSON tool record is {"record": {...}, "inst_index": N}; the decoded view
# flattens every field into its own column.  Each entry below maps a decoded-view
# column to the value extracted from the JSON "record" object.  hw_id / arbiter /
# workgroup fields that the CSV projection dropped are all covered here.
# ---------------------------------------------------------------------------
_HW_ID_FIELDS = [
    "chiplet",
    "wave_id",
    "simd_id",
    "pipe_id",
    "cu_or_wgp_id",
    "shader_array_id",
    "shader_engine_id",
    "workgroup_id",
    "vm_id",
    "queue_id",
    "microengine_id",
]

# Arbiter pipes tracked by the gfx9 stochastic snapshot.
_ARB_PIPES = [
    "valu",
    "matrix",
    "lds",
    "scalar",
    "vmem_tex",
    "flat",
    "exp",
    "misc",
]


def _hw_id_value(hw_id, name):
    # The SDK serializes hw_id.workgroup_id with a trailing space in its JSON key
    # (upstream cereal quirk); accept either spelling.
    if name == "workgroup_id" and "workgroup_id " in hw_id:
        return hw_id["workgroup_id "]
    return hw_id[name]


def _field_pairs(record, stochastic):
    """Yield (decoded_view_column, expected_json_value, is_name_string) tuples."""
    hw = record["hw_id"]
    wg = record["wrkgrp_id"]
    corr = record["corr_id"]

    pairs = [
        ("timestamp", record["timestamp"], False),
        ("dispatch_id", record["dispatch_id"], False),
        ("exec_mask", record["exec_mask"], False),
        ("code_object_id", record["pc"]["code_object_id"], False),
        ("code_object_offset", record["pc"]["code_object_offset"], False),
        ("correlation_id", corr["internal"], False),
        ("external_correlation_id", corr["external"], False),
        ("wave_in_group", record["wave_in_grp"], False),
        ("workgroup_id_x", wg["x"], False),
        ("workgroup_id_y", wg["y"], False),
        ("workgroup_id_z", wg["z"], False),
    ]
    for name in _HW_ID_FIELDS:
        pairs.append(("hw_id_" + name, _hw_id_value(hw, name), False))

    if stochastic:
        snapshot = record["snapshot"]
        pairs += [
            ("wave_issued", record["wave_issued"], False),
            ("wave_count", record["wave_cnt"], False),
            # inst_type / stall_reason are serialized as enum name strings in JSON;
            # compare against the decoded view's *_name columns.
            ("inst_type_name", record["inst_type"], True),
            ("stall_reason_name", snapshot["stall_reason"], True),
            ("dual_issue_valu", snapshot["dual_issue_valu"], False),
        ]
        for pipe in _ARB_PIPES:
            pairs.append(
                ("arb_state_issue_" + pipe, snapshot["arb_state_issue_" + pipe], False)
            )
        for pipe in _ARB_PIPES:
            pairs.append(
                ("arb_state_stall_" + pipe, snapshot["arb_state_stall_" + pipe], False)
            )
    return pairs


def _decoded_rows(conn, stochastic):
    # The decoded view's correlation_id is the *internal* id; join rocpd_event to
    # also expose the user-facing *external* id.  Row order follows insertion order
    # (autoincrement id), which matches the generator order the JSON is serialized
    # from, so the two sequences line up 1:1.
    #
    # Select every decoded column except instruction/instruction_comment: those two
    # trigger on-demand ISA disassembly per row (expensive over the full sample set)
    # and are validated separately by the disassembly tests, not here.
    probe = conn.execute("SELECT * FROM rocpd_gpu_pc_sample_decoded LIMIT 0")
    skip = {"instruction", "instruction_comment"}
    keep = [desc[0] for desc in probe.description if desc[0] not in skip]
    select_list = ", ".join(f'd."{name}"' for name in keep)
    null_test = "IS NOT NULL" if stochastic else "IS NULL"
    query = (
        f"SELECT {select_list}, e.correlation_id AS external_correlation_id "
        "FROM rocpd_gpu_pc_sample_decoded d "
        "JOIN rocpd_event e ON e.id = d.event_id "
        f"WHERE d.wave_issued {null_test} "
        "ORDER BY d.id ASC"
    )
    cursor = conn.execute(query)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def test_rocpd_tables_populated(rocpd_connection):
    assert _count_rows(rocpd_connection, "rocpd_gpu_pc_sample") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_schema") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_field") > 0


def test_rocpd_sample_count_matches_json(rocpd_connection, json_data):
    # Independent oracle: the ROCPD row count must equal the number of PC-sample
    # records emitted to JSON from the same generators.
    tool = _tool(json_data)
    json_count = len(tool["buffer_records"].get("pc_sample_host_trap", [])) + len(
        tool["buffer_records"].get("pc_sample_stochastic", [])
    )
    assert json_count > 0
    assert _count_rows(rocpd_connection, "rocpd_gpu_pc_sample") == json_count


def test_rocpd_parent_dispatch_linkage(rocpd_connection):
    # The producer runs with --kernel-trace, so samples taken inside a captured
    # dispatch link to that dispatch's event via rocpd_event.parent_id.
    linked = rocpd_connection.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample S "
        "JOIN rocpd_event E ON E.id = S.event_id "
        "WHERE E.parent_id IS NOT NULL"
    ).fetchone()[0]
    assert linked > 0


def test_json_pc_sampling_records_present(json_data):
    tool = _tool(json_data)

    host_trap_records = tool["buffer_records"].get("pc_sample_host_trap", [])
    stochastic_records = tool["buffer_records"].get("pc_sample_stochastic", [])

    assert len(host_trap_records) + len(stochastic_records) > 0


# Per-method blob-schema names and the stochastic-only columns, used to assert a
# fixture actually exercised the sampling method it requested.
_METHOD_SCHEMA = {
    "stochastic": "pc_sample_extdata_stochastic",
    "host_trap": "pc_sample_extdata_hosttrap",
}
_STOCHASTIC_ONLY_COLUMNS = ("wave_issued", "inst_type", "stall_reason", "wave_count")


def test_rocpd_vs_json_all_fields(rocpd_connection, json_data):
    # Full-field validation: compare every field the ROCPD database stores for each
    # PC sample against the rocprofv3 JSON oracle produced by the same run.  This
    # covers all the fields the (lossy) CSV projection omitted -- hw_id, arbiter
    # issue/stall state, workgroup coordinates, exec_mask, correlation ids, and the
    # inst_type / stall_reason enum names -- by reading the self-describing decoded
    # view.  Every column _field_pairs() maps for the method must be present in the
    # decoded view (a dropped/renamed column fails loudly instead of being silently
    # skipped).  JSON records and DB rows are matched positionally (shared generator
    # order); counts are asserted equal first.
    total_records = 0
    total_field_checks = 0
    for method_key, stochastic in (
        ("pc_sample_stochastic", True),
        ("pc_sample_host_trap", False),
    ):
        records = _json_records(json_data, method_key)
        rows = _decoded_rows(rocpd_connection, stochastic)
        assert len(records) == len(
            rows
        ), f"{method_key}: JSON has {len(records)} samples, DB has {len(rows)}"
        total_records += len(records)
        if not rows:
            continue
        # Require every column _field_pairs() maps for this method: a schema
        # regression that drops a HW-ID or arbiter field must fail here rather than
        # be silently skipped by the per-row comparison below.  The mapped column
        # set depends only on the method, so probing the first record is sufficient.
        expected_columns = [
            column for column, _, _ in _field_pairs(records[0]["record"], stochastic)
        ]
        present = set(rows[0].keys())
        missing = [column for column in expected_columns if column not in present]
        assert not missing, (
            f"{method_key}: decoded view is missing mapped columns {missing}; "
            "the full-field comparison would otherwise skip them"
        )
        for index, (sample, row) in enumerate(zip(records, rows)):
            record = sample["record"]
            for column, expected, is_name in _field_pairs(record, stochastic):
                actual = row[column]
                if is_name:
                    assert str(actual) == str(
                        expected
                    ), f"{method_key}[{index}].{column}: DB={actual!r} JSON={expected!r}"
                elif isinstance(actual, float):
                    # SQLite has no unsigned-64-bit integer storage: a uint64
                    # value above INT64_MAX bound to a numeric-affinity column
                    # (the BIGINT timestamp is the only PC-sample field that can
                    # exceed it; exec_mask uses a TEXT column precisely to avoid
                    # this) is coerced to REAL and comes back as a float.  Compare
                    # with the same rounding SQLite applied -- values SQLite can
                    # store exactly are returned as ints and checked exactly below.
                    assert float(actual) == float(
                        int(expected)
                    ), f"{method_key}[{index}].{column}: DB={actual} JSON={expected}"
                else:
                    assert int(actual) == int(
                        expected
                    ), f"{method_key}[{index}].{column}: DB={actual} JSON={expected}"
                total_field_checks += 1

    assert total_records > 0, "no PC-sample records found in JSON oracle"
    assert total_field_checks > 0, "no fields were compared"


def test_rocpd_method_identity(rocpd_connection, json_data, expected_method):
    # Guard against a fixture silently exercising the wrong sampling method: the
    # JSON oracle and the database must contain only the requested method, the
    # matching per-method blob schema must be registered (and the other absent),
    # and the stochastic-only columns must be fully populated for a stochastic run
    # and entirely NULL for a host-trap run.
    assert expected_method in _METHOD_SCHEMA, f"unknown method {expected_method!r}"
    conn = rocpd_connection
    tool = _tool(json_data)
    host_trap = tool["buffer_records"].get("pc_sample_host_trap", [])
    stochastic = tool["buffer_records"].get("pc_sample_stochastic", [])

    other_method = "host_trap" if expected_method == "stochastic" else "stochastic"
    if expected_method == "stochastic":
        expected_records, opposite_records = stochastic, host_trap
    else:
        expected_records, opposite_records = host_trap, stochastic
    assert len(expected_records) > 0, f"no {expected_method} records in JSON oracle"
    assert (
        len(opposite_records) == 0
    ), f"{expected_method} run unexpectedly produced {other_method} records"

    registered_schemas = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT name FROM rocpd_info_blob_schema"
        ).fetchall()
    }
    assert _METHOD_SCHEMA[expected_method] in registered_schemas
    assert _METHOD_SCHEMA[other_method] not in registered_schemas

    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    for column in _STOCHASTIC_ONLY_COLUMNS:
        # Column names are fixed schema identifiers, not user input.
        non_null = conn.execute(
            f"SELECT COUNT(*) FROM rocpd_gpu_pc_sample WHERE {column} IS NOT NULL"
        ).fetchone()[0]
        if expected_method == "stochastic":
            assert (
                non_null == total
            ), f"{column} must be populated for every stochastic sample"
        else:
            assert non_null == 0, f"{column} must be NULL for every host-trap sample"


def test_rocpd_vs_json_kernel_and_agents(rocpd_connection, json_data):
    # Cross-check dispatch and GPU-agent counts directly against the DB (previously
    # done via the CSV exports).
    tool = _tool(json_data)

    kernel_records = tool["buffer_records"]["kernel_dispatch"]
    assert len(kernel_records) > 0
    assert _count_rows(rocpd_connection, "rocpd_kernel_dispatch") == len(kernel_records)

    json_gpu_agents = [a for a in tool["agents"] if a["type"] == 2]
    assert len(json_gpu_agents) > 0
    db_gpu_agents = rocpd_connection.execute(
        "SELECT COUNT(*) FROM rocpd_info_agent WHERE type = 'GPU'"
    ).fetchone()[0]
    assert db_gpu_agents == len(json_gpu_agents)


def test_setup_blob_views_decoded_view(rocpd_connection):
    from rocpd.query import update_query_for_blob_views

    # update_query_for_blob_views should return a query with base-table names rewritten to decoded view names.
    # The decoded views are created by importer.setup_blob_views during RocpdImportData init.
    rewritten = update_query_for_blob_views(
        rocpd_connection,
        "SELECT timestamp FROM rocpd_gpu_pc_sample LIMIT 1",
    )

    assert rewritten is not None
    assert "rocpd_gpu_pc_sample_decoded" in rewritten

    view_exists = rocpd_connection.execute(
        "SELECT COUNT(*) FROM sqlite_temp_master "
        "WHERE type='view' AND name='rocpd_gpu_pc_sample_decoded'"
    ).fetchone()[0]
    assert view_exists == 1

    # Query decoded blob fields to ensure the view evaluates correctly.
    row = rocpd_connection.execute(
        "SELECT timestamp, hw_id_simd_id, hw_id_wave_id, code_object_offset "
        "FROM rocpd_gpu_pc_sample_decoded "
        "LIMIT 1"
    ).fetchone()

    assert row is not None
    assert row[0] is not None
    assert row[1] is not None
    assert row[2] is not None
    assert row[3] is not None


def test_rocpd_stochastic_columns_populated(rocpd_connection):
    # Stochastic samples must populate wave_issued/wave_count for every row.
    conn = rocpd_connection
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    # Host-trap databases leave wave_issued NULL for every sample; this check is
    # stochastic-only, so skip it when the run produced no stochastic samples.
    stochastic_rows = conn.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample WHERE wave_issued IS NOT NULL"
    ).fetchone()[0]
    if stochastic_rows == 0:
        pytest.skip("no stochastic PC samples in database (host-trap run)")
    populated = conn.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample "
        "WHERE wave_issued IS NOT NULL AND wave_count IS NOT NULL"
    ).fetchone()[0]
    assert populated == total
    # Every decoded arbiter-state blob field must evaluate to a valid 0/1 flag, and the
    # packed hw_id bitfields must decode within their hardware-defined ranges (simd_id is
    # 2 bits, shader_array_id is 1 bit).  A wrong blob offset/size would violate these.
    invalid = conn.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample_decoded "
        "WHERE arb_state_issue_valu NOT IN (0, 1) "
        "OR arb_state_stall_valu NOT IN (0, 1) "
        "OR dual_issue_valu NOT IN (0, 1) "
        "OR hw_id_simd_id > 3 OR hw_id_shader_array_id > 1"
    ).fetchone()[0]
    assert invalid == 0


def test_rocpd_on_demand_disassembly(rocpd_connection):
    # Default path: without --complete-isa-decode no instruction text is persisted
    # at collection, so the decoded view must disassemble on demand via the
    # registered SQLite UDFs (rocpd_isa_instruction / rocpd_isa_comment).  The
    # write-back cache is on by default, but validation opens the database
    # read-only (cache_disassembly=False), so nothing is written back and the
    # rocpd_disassembly_data table stays empty.
    conn = rocpd_connection
    assert _count_rows(conn, "rocpd_disassembly_data") == 0
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    # Materialize both decoded columns rather than a bare COUNT(*) over the
    # projection (which the planner may satisfy without calling the UDFs at all):
    # fetching the rows forces SQLite to evaluate rocpd_isa_instruction AND
    # rocpd_isa_comment for every row, proving both UDFs are wired, and stays 1:1
    # with the sample rows.
    decoded = conn.execute(
        "SELECT instruction, instruction_comment FROM rocpd_gpu_pc_sample_decoded"
    ).fetchall()
    assert len(decoded) == total
    # On-demand decoding must resolve instruction text for some sampled PCs (the
    # workload's code objects are decodable), so an all-NULL view fails.
    assert any(instruction is not None for instruction, _ in decoded)


def test_rocpd_persisted_disassembly(rocpd_disasm_connection):
    # Opt-in path: produced with --complete-isa-decode, so instruction text is
    # persisted at finalization into rocpd_disassembly_data and served by the decoded view
    # (COALESCE prefers the stored text over on-demand decoding).
    conn = rocpd_disasm_connection
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    assert _count_rows(conn, "rocpd_disassembly_data") > 0
    # Materialize the decoded instruction text for every row (forces evaluation)
    # and keep it 1:1 with the sample rows.
    decoded = conn.execute(
        "SELECT instruction FROM rocpd_gpu_pc_sample_decoded"
    ).fetchall()
    assert len(decoded) == total
    # Prove the *persisted* path is actually used rather than the lazy UDF
    # fallback: for every sampled PC that has a stored disassembly row the decoded
    # view must return exactly the stored text (COALESCE(d.instruction, ...)), and
    # at least one such stored PC must be surfaced through the view.
    served = conn.execute("""
        SELECT COUNT(*)
        FROM rocpd_gpu_pc_sample_decoded v
        JOIN rocpd_disassembly_data d
          ON d.guid = v.guid
         AND d.code_object_id = v.code_object_id
         AND d.code_object_offset = v.code_object_offset
        WHERE v.instruction IS NOT NULL
        """).fetchone()[0]
    assert served > 0
    mismatched = conn.execute("""
        SELECT COUNT(*)
        FROM rocpd_gpu_pc_sample_decoded v
        JOIN rocpd_disassembly_data d
          ON d.guid = v.guid
         AND d.code_object_id = v.code_object_id
         AND d.code_object_offset = v.code_object_offset
        WHERE v.instruction IS NOT d.instruction
        """).fetchone()[0]
    assert mismatched == 0, "decoded view did not serve the persisted disassembly text"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
