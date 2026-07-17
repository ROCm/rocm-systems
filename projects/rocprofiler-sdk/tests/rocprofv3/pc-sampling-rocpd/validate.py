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


def test_rocpd_tables_populated(rocpd_connection):
    assert _count_rows(rocpd_connection, "rocpd_gpu_pc_sample") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_schema") > 0
    assert _count_rows(rocpd_connection, "rocpd_info_blob_field") > 0


def test_rocpd_sample_count_matches_json(rocpd_connection, json_data):
    # Independent oracle: the ROCPD row count must equal the number of PC-sample
    # records emitted to JSON from the same generators.
    tool = json_data["rocprofiler-sdk-tool"]
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


def test_rocpd2csv_pc_sampling_matches_db(
    rocpd_connection, rocpd2csv_pc_sampling_dataframe
):
    # The stochastic CSV export row count must equal the stochastic samples in the DB.
    stochastic_count = rocpd_connection.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample WHERE wave_issued IS NOT NULL"
    ).fetchone()[0]
    assert len(rocpd2csv_pc_sampling_dataframe) == stochastic_count


def test_validate_rocpd2csv_exec_mask_manipulation(rocpd2csv_pc_sampling_dataframe):
    # End-to-end data-correctness check (not just row counts): the exec-mask
    # manipulation workload emits deterministic, verifiable sample values, so the
    # rocpd -> CSV export must satisfy the same semantic invariants the shared
    # validator enforces for the direct rocprofv3 CSV -- exec-mask popcount ==
    # correlation id, the per-CID mask pattern, and v_rcp_f64/v_rcp_f32 instruction
    # decoding within their source-line ranges.  This exercises the full rocpd
    # pipeline end to end: blob pack/unpack, on-demand disassembly, and the rocpd2csv
    # projection.
    from rocprofiler_sdk.pc_sampling.exec_mask_manipulation.csv import (
        exec_mask_manipulation_validate_csv,
    )

    # The producer runs stochastic sampling at --pc-sampling-interval 1048576,
    # identical to the standalone stochastic exec-mask test (which validates the same
    # CSV with all_sampled=False).  Sparse sampling means not every kernel or source
    # line is guaranteed to be sampled, so all_sampled must be False; the last kernel
    # is heavy enough to always be sampled, which is the only sampling guarantee the
    # validator relies on.
    exec_mask_manipulation_validate_csv(
        rocpd2csv_pc_sampling_dataframe, all_sampled=False
    )


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


def test_json_vs_rocpd2csv_exports(
    json_data,
    rocpd2csv_kernel_data,
    rocpd2csv_agent_data,
):
    assert len(rocpd2csv_kernel_data) > 0
    assert len(rocpd2csv_agent_data) > 0

    tool = json_data["rocprofiler-sdk-tool"]

    # Validate kernel row count against JSON kernel-dispatch records.
    kernel_records = tool["buffer_records"]["kernel_dispatch"]
    assert len(rocpd2csv_kernel_data) == len(kernel_records)

    # Validate GPU agent count against rocpd2csv agent-info export.
    # Agent_Type in rocpd2csv uses strings such as "GPU".
    csv_gpu_agents = [r for r in rocpd2csv_agent_data if r.get("Agent_Type") == "GPU"]
    json_gpu_agents = [a for a in tool["agents"] if a["type"] == 2]
    assert len(csv_gpu_agents) == len(json_gpu_agents)


def test_json_pc_sampling_records_present(json_data):
    tool = json_data["rocprofiler-sdk-tool"]

    host_trap_records = tool["buffer_records"].get("pc_sample_host_trap", [])
    stochastic_records = tool["buffer_records"].get("pc_sample_stochastic", [])

    assert len(host_trap_records) + len(stochastic_records) > 0


def test_rocpd_stochastic_columns_populated(rocpd_connection):
    # Stochastic samples must populate wave_issued/wave_count for every row.
    conn = rocpd_connection
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
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
    # Default path: without --complete-isa-decode no instruction text is
    # persisted, so the decoded view must disassemble on demand via the registered SQLite
    # UDFs (rocpd_isa_instruction / rocpd_isa_comment).
    conn = rocpd_connection
    assert _count_rows(conn, "rocpd_disassembly_data") == 0
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    # The instruction/instruction_comment columns must project from the decoded view
    # without error (UDFs wired) and stay 1:1 with the sample rows.
    decoded = conn.execute(
        "SELECT COUNT(*) FROM "
        "(SELECT instruction, instruction_comment FROM rocpd_gpu_pc_sample_decoded)"
    ).fetchone()[0]
    assert decoded == total


def test_rocpd_persisted_disassembly(rocpd_disasm_connection):
    # Opt-in path: produced with --complete-isa-decode, so instruction text is
    # persisted at finalization into rocpd_disassembly_data and served by the decoded view
    # (COALESCE prefers the stored text over on-demand decoding).
    conn = rocpd_disasm_connection
    total = _count_rows(conn, "rocpd_gpu_pc_sample")
    assert total > 0
    assert _count_rows(conn, "rocpd_disassembly_data") > 0
    decoded = conn.execute(
        "SELECT COUNT(*) FROM (SELECT instruction FROM rocpd_gpu_pc_sample_decoded)"
    ).fetchone()[0]
    assert decoded == total
    non_null = conn.execute(
        "SELECT COUNT(*) FROM rocpd_gpu_pc_sample_decoded WHERE instruction IS NOT NULL"
    ).fetchone()[0]
    assert non_null > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
