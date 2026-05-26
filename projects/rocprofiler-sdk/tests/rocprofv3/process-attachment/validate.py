#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import os
import sqlite3
import sys

import pytest

_HIP_REGION_CATEGORIES = (
    "HIP_RUNTIME_API",
    "HIP_RUNTIME_API_EXT",
    "HIP_COMPILER_API",
    "HIP_COMPILER_API_EXT",
)


def _int_field(row, *keys, default=0):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return int(row[key])
    return default


def _assert_simple_kernel_rows(rows, *, source):
    """Validate attachment-test simple_kernel dispatches (CSV or rocpd kernels view)."""
    assert len(rows) > 0, f"no kernel dispatches in {source}"

    kernel_names = [
        str(row.get("Kernel_Name") or row.get("name") or "") for row in rows
    ]
    assert any(
        "simple_kernel" in name for name in kernel_names
    ), f"expected simple_kernel in {source}, got names: {kernel_names[:20]}"

    simple_rows = [
        row
        for row in rows
        if "simple_kernel" in str(row.get("Kernel_Name") or row.get("name") or "")
    ]
    assert len(simple_rows) > 0

    for row in simple_rows:
        kind = row.get("Kind")
        if kind is not None:
            assert kind == "KERNEL_DISPATCH"

        start = _int_field(row, "Start_Timestamp", "start")
        end = _int_field(row, "End_Timestamp", "end")
        assert end >= start, f"invalid timestamps in {source}: start={start} end={end}"

        queue_id = _int_field(row, "Queue_Id", "queue_id")
        dispatch_id = _int_field(row, "dispatch_id", "Kernel_Id", "kernel_Id")
        correlation = _int_field(row, "Correlation_Id", "stack_id")
        if queue_id:
            assert queue_id > 0
        if dispatch_id:
            assert dispatch_id > 0
        if correlation:
            assert correlation > 0

        # attachment-test uses 256x1x1 thread blocks
        wg_x = _int_field(row, "Workgroup_Size_X", "workgroup_x")
        wg_y = _int_field(row, "Workgroup_Size_Y", "workgroup_y")
        wg_z = _int_field(row, "Workgroup_Size_Z", "workgroup_z")
        if wg_x:
            assert wg_x == 256, f"unexpected Workgroup_Size_X in {source}: {wg_x}"
            assert wg_y == 1
            assert wg_z == 1

        grid_x = _int_field(row, "Grid_Size_X", "grid_x")
        if grid_x:
            assert grid_x >= 1


def _load_rocpd_kernel_rows(db_path):
    conn = sqlite3.connect(db_path)
    try:
        views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('view', 'table')"
            ).fetchall()
        }
        assert "kernels" in views, f"kernels view missing in rocpd db: {db_path}"

        cursor = conn.execute(
            """
            SELECT
                name AS Kernel_Name,
                start AS Start_Timestamp,
                end AS End_Timestamp,
                queue_id AS Queue_Id,
                dispatch_id,
                stack_id AS Correlation_Id,
                workgroup_x AS Workgroup_Size_X,
                workgroup_y AS Workgroup_Size_Y,
                workgroup_z AS Workgroup_Size_Z,
                grid_x AS Grid_Size_X,
                grid_y AS Grid_Size_Y,
                grid_z AS Grid_Size_Z
            FROM kernels
            """
        )
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        for row in rows:
            row["Kind"] = "KERNEL_DISPATCH"
        return rows
    finally:
        conn.close()


def _rocpd_view_row_count(db_path, view_name, where_clause="", params=()):
    conn = sqlite3.connect(db_path)
    try:
        query = f"SELECT COUNT(*) FROM {view_name}"
        if where_clause:
            query += f" WHERE {where_clause}"
        return conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()


def test_rocpd_database_exists(rocpd_input_path):
    """Verify rocpd output was produced and contains tables."""
    assert os.path.isfile(rocpd_input_path), f"missing rocpd db: {rocpd_input_path}"
    assert os.path.getsize(rocpd_input_path) > 0

    conn = sqlite3.connect(rocpd_input_path)
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        assert len(tables) > 0, "rocpd database has no tables"
    finally:
        conn.close()


def _assert_matrix_transpose_kernel_rows(rows, *, source):
    """Validate simple-transpose / openmp-target matrixTranspose dispatches."""
    assert len(rows) > 0, f"no kernel dispatches in {source}"
    kernel_names = [
        str(row.get("Kernel_Name") or row.get("name") or "") for row in rows
    ]
    assert any(
        "matrixTranspose" in name for name in kernel_names
    ), f"expected matrixTranspose in {source}, got names: {kernel_names[:20]}"
    matched = [
        row
        for row in rows
        if "matrixTranspose" in str(row.get("Kernel_Name") or row.get("name") or "")
    ]
    for row in matched:
        start = _int_field(row, "Start_Timestamp", "start")
        end = _int_field(row, "End_Timestamp", "end")
        assert end >= start, f"invalid timestamps in {source}: start={start} end={end}"


def test_rocpd_kernels_captured(rocpd_input_path):
    """Verify live attach captured simple_kernel dispatches in rocpd."""
    rows = _load_rocpd_kernel_rows(rocpd_input_path)
    _assert_simple_kernel_rows(rows, source=rocpd_input_path)


def test_rocpd_matrix_transpose_kernels_captured(rocpd_input_path):
    """Verify live attach captured matrixTranspose kernels (MPI simple-transpose)."""
    rows = _load_rocpd_kernel_rows(rocpd_input_path)
    _assert_matrix_transpose_kernel_rows(rows, source=rocpd_input_path)


def test_rocpd_openmp_offload_kernels_captured(rocpd_input_path):
    """Verify live attach captured OpenMP offload kernel dispatches (openmp-attach)."""
    rows = _load_rocpd_kernel_rows(rocpd_input_path)
    assert len(rows) > 0, f"no kernel dispatches in {rocpd_input_path}"


def test_rocpd_hip_regions_captured(rocpd_input_path):
    """Verify HIP API regions were recorded (hip-rocpd execute path)."""
    placeholders = ", ".join("?" for _ in _HIP_REGION_CATEGORIES)
    count = _rocpd_view_row_count(
        rocpd_input_path,
        "regions",
        f"category IN ({placeholders})",
        _HIP_REGION_CATEGORIES,
    )
    assert count > 0, "no HIP API regions in rocpd output"


def test_rocpd_pmc_counters_captured(rocpd_input_path):
    """Verify SQ_WAVES PMC samples were stored after live attach with --pmc."""
    conn = sqlite3.connect(rocpd_input_path)
    try:
        views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('view', 'table')"
            ).fetchall()
        }
        assert "counters_collection" in views, (
            f"counters_collection view missing in {rocpd_input_path}"
        )
        count = conn.execute(
            """
            SELECT COUNT(*) FROM counters_collection
            WHERE counter_name LIKE '%SQ_WAVES%'
            """
        ).fetchone()[0]
        assert count > 0, "no SQ_WAVES samples in rocpd counters_collection"

        values = conn.execute(
            """
            SELECT value FROM counters_collection
            WHERE counter_name LIKE '%SQ_WAVES%'
            """
        ).fetchall()
        assert all(row[0] is not None for row in values)
    finally:
        conn.close()


def test_rocpd_foreign_key_integrity(rocpd_input_path):
    """rocpd database must pass SQLite foreign-key check after detach."""
    conn = sqlite3.connect(rocpd_input_path)
    try:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert len(violations) == 0, (
            f"rocpd foreign key violations in {rocpd_input_path}: {violations}"
        )
    finally:
        conn.close()


def test_rocpd_no_duplicate_kernel_timestamps(rocpd_input_path):
    """Detect double-buffer flush duplicates in rocpd kernels view."""
    conn = sqlite3.connect(rocpd_input_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM kernels").fetchone()[0]
        unique_starts = conn.execute(
            "SELECT COUNT(DISTINCT start) FROM kernels"
        ).fetchone()[0]
        unique_ends = conn.execute("SELECT COUNT(DISTINCT end) FROM kernels").fetchone()[0]
    finally:
        conn.close()

    assert total > 0
    assert total == unique_starts == unique_ends, (
        f"duplicate kernel records: total={total}, unique starts={unique_starts}, "
        f"unique ends={unique_ends}"
    )


def test_csv_output_exists(kernel_trace_csv_path):
    """Verify kernel trace CSV was written (sys-trace attach path)."""
    assert kernel_trace_csv_path, "kernel trace csv path required"
    assert os.path.isfile(kernel_trace_csv_path), f"missing csv: {kernel_trace_csv_path}"
    assert os.path.getsize(kernel_trace_csv_path) > 0


def test_kernel_trace_captured(kernel_input_data):
    """Verify kernel dispatches were captured during live attach (sys-trace CSV)."""
    _assert_simple_kernel_rows(kernel_input_data, source="kernel trace csv")


def test_json_results_exist(json_input_path):
    """Verify rocprofv3 JSON output was written (PC sampling attach path)."""
    assert os.path.isfile(json_input_path), f"missing json: {json_input_path}"
    assert os.path.getsize(json_input_path) > 0


def test_pc_sampling_host_trap_samples(json_data):
    """Verify host-trap PC sampling records were collected during live attach."""
    data = json_data["rocprofiler-sdk-tool"]
    assert "pc_sample_host_trap" in data["buffer_records"], (
        "pc_sample_host_trap records missing from JSON output"
    )
    samples = data["buffer_records"]["pc_sample_host_trap"]
    assert len(samples) > 0, "expected at least one pc_sample_host_trap record"


def test_selected_regions_markers_captured(json_data):
    """Verify roctx marker records exist when profiling selected-regions on transpose."""
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]
    assert "marker_api" in buffer_records, "marker_api records missing from JSON output"
    assert len(buffer_records["marker_api"]) > 0, "expected at least one marker_api record"

    marker_names = [entry["value"] for entry in data["strings"]["marker_api"]]
    assert any("run/" in name for name in marker_names), (
        "expected transpose roctx range names (run/...) in marker_api strings"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
