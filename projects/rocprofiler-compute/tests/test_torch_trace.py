##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

##############################################################################

import json
import sqlite3
from pathlib import Path

import pandas as pd
import test_utils

from utils.rocpd_data import (
    COUNTERS_COLLECTION_QUERY,
    MARKER_API_TRACE_QUERY,
    convert_dbs_to_csv,
)
from utils.utils import process_torch_trace_output

GUID = "abc-1234-def"

MARKER_ROWS = [
    (
        "roctx",
        "nn.Module.Linear.forward:#1@test.py:10",
        100,
        200,
        1000,
        GUID,
        1000,
        2000,
    ),
    (
        "roctx",
        "nn.Module.Linear.forward:#2@test.py:10",
        100,
        200,
        1001,
        GUID,
        3000,
        4000,
    ),
    ("roctx", "torch.mm:#1@test.py:15", 100, 200, 1002, GUID, 5000, 6000),
]

COUNTER_ROWS = [
    (
        0,
        GUID,
        1000,
        0,
        100,
        64,
        256,
        0,
        0,
        32,
        0,
        16,
        "kernel_gemm",
        1100,
        1900,
        0,
        "SQ_WAVES",
        42,
    ),
    (
        0,
        GUID,
        1001,
        1,
        100,
        64,
        256,
        0,
        0,
        32,
        0,
        16,
        "kernel_gemm",
        3100,
        3900,
        0,
        "SQ_WAVES",
        50,
    ),
    (
        0,
        GUID,
        1002,
        2,
        100,
        64,
        256,
        0,
        0,
        32,
        0,
        16,
        "kernel_mm",
        5100,
        5900,
        0,
        "SQ_WAVES",
        30,
    ),
]


# ---- SQL query constants reference stack_id ----


def test_counters_query_uses_stack_id():
    """Test that the counters query uses stack_id as Correlation_Id."""
    assert "stack_id as Correlation_Id" in COUNTERS_COLLECTION_QUERY

    query_lower = COUNTERS_COLLECTION_QUERY.lower()
    assert "correlation_id as " not in query_lower
    assert "\n    correlation_id" not in query_lower


def test_marker_query_uses_stack_id():
    """Test that the marker query uses stack_id as Correlation_Id."""
    assert "stack_id AS Correlation_Id" in MARKER_API_TRACE_QUERY

    query_lower = MARKER_API_TRACE_QUERY.lower()
    assert "correlation_id as " not in query_lower
    assert "\n    correlation_id" not in query_lower


# ---- Test 2: convert_dbs_to_csv populates Correlation_Id from stack_id ----


def create_rocpd_test_db(workload_dir):
    """
    Build a minimal rocpd-style SQLite database with counters_collection
    and regions tables whose schemas match the production queries.
    """
    db_path = str(Path(workload_dir) / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE counters_collection (
            agent_id INTEGER, guid TEXT, stack_id INTEGER, dispatch_id INTEGER,
            pid INTEGER, grid_size INTEGER, workgroup_size INTEGER,
            lds_block_size INTEGER, scratch_size INTEGER, vgpr_count INTEGER,
            accum_vgpr_count INTEGER, sgpr_count INTEGER, kernel_name TEXT,
            start INTEGER, end INTEGER, kernel_id INTEGER,
            counter_name TEXT, value REAL
        )"""
    )
    conn.executemany(
        "INSERT INTO counters_collection VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        COUNTER_ROWS,
    )
    conn.execute(
        """CREATE TABLE regions (
            category TEXT, extdata TEXT, pid INTEGER, tid INTEGER,
            stack_id INTEGER, guid TEXT, start INTEGER, end INTEGER
        )"""
    )
    region_rows = [
        (cat, json.dumps({"message": func}), pid, tid, sid, guid, s, e)
        for cat, func, pid, tid, sid, guid, s, e in MARKER_ROWS
    ]
    conn.executemany(
        "INSERT INTO regions VALUES (?,?,?,?,?,?,?,?)",
        region_rows,
    )
    conn.commit()
    conn.close()
    return db_path


def test_counter_csv_has_correlation_id_from_stack_id():
    """Test that the counter CSV has correlation_id from stack_id."""
    workload_dir = test_utils.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    counter_csv = str(Path(workload_dir) / "counter_collection.csv")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv")

    db_path = create_rocpd_test_db(workload_dir)
    convert_dbs_to_csv([db_path], counter_csv, marker_csv)

    df = pd.read_csv(counter_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = [row[2] for row in COUNTER_ROWS]
    assert list(df["Correlation_Id"]) == expected_ids

    test_utils.clean_output_dir(True, workload_dir)


def test_marker_csv_has_correlation_id_from_stack_id():
    """Test that the marker CSV has correlation_id from stack_id."""
    workload_dir = test_utils.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    counter_csv = str(Path(workload_dir) / "counter_collection.csv")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv")

    db_path = create_rocpd_test_db(workload_dir)
    convert_dbs_to_csv([db_path], counter_csv, marker_csv)

    df = pd.read_csv(marker_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = sorted(row[4] for row in MARKER_ROWS)
    assert sorted(df["Correlation_Id"].tolist()) == expected_ids

    test_utils.clean_output_dir(True, workload_dir)


# ---- process_torch_trace_output parity for rocpd vs csv layouts ----


MARKER_COLUMNS_ROCPD = [
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "GUID",
    "Start_Timestamp",
    "End_Timestamp",
]

COUNTER_COLUMNS_ROCPD = [
    "GPU_ID",
    "GUID",
    "Correlation_Id",
    "Dispatch_ID",
    "PID",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Scratch_Per_Workitem",
    "Arch_VGPR",
    "Accum_VGPR",
    "SGPR",
    "Kernel_Name",
    "Start_Timestamp",
    "End_Timestamp",
    "Kernel_ID",
    "Counter_Name",
    "Counter_Value",
]

MARKER_COLUMNS_CSV = [
    "Domain",
    "Function",
    "Process_Id",
    "Thread_Id",
    "Correlation_Id",
    "Start_Timestamp",
    "End_Timestamp",
]

COUNTER_COLUMNS_CSV = [
    "Correlation_Id",
    "Kernel_Name",
    "Counter_Name",
    "Counter_Value",
    "Start_Timestamp",
    "End_Timestamp",
]


def build_marker_df(include_guid):
    """Build a dataframe from the marker rows."""
    data = {
        "Domain": [r[0] for r in MARKER_ROWS],
        "Function": [r[1] for r in MARKER_ROWS],
        "Process_Id": [r[2] for r in MARKER_ROWS],
        "Thread_Id": [r[3] for r in MARKER_ROWS],
        "Correlation_Id": [r[4] for r in MARKER_ROWS],
        "Start_Timestamp": [r[6] for r in MARKER_ROWS],
        "End_Timestamp": [r[7] for r in MARKER_ROWS],
    }

    if include_guid:
        data["GUID"] = [r[5] for r in MARKER_ROWS]

    return pd.DataFrame(data)


def build_counter_df(include_guid):
    """Build a dataframe from the counter rows."""
    data = {
        "Correlation_Id": [r[2] for r in COUNTER_ROWS],
        "Kernel_Name": [r[12] for r in COUNTER_ROWS],
        "Counter_Name": [r[16] for r in COUNTER_ROWS],
        "Counter_Value": [r[17] for r in COUNTER_ROWS],
        "Start_Timestamp": [r[13] for r in COUNTER_ROWS],
        "End_Timestamp": [r[14] for r in COUNTER_ROWS],
    }

    if include_guid:
        data["GUID"] = [r[1] for r in COUNTER_ROWS]
        data["GPU_ID"] = [r[0] for r in COUNTER_ROWS]
        data["Dispatch_ID"] = [r[3] for r in COUNTER_ROWS]
        data["PID"] = [r[4] for r in COUNTER_ROWS]
        data["Grid_Size"] = [r[5] for r in COUNTER_ROWS]
        data["Workgroup_Size"] = [r[6] for r in COUNTER_ROWS]
        data["LDS_Per_Workgroup"] = [r[7] for r in COUNTER_ROWS]
        data["Scratch_Per_Workitem"] = [r[8] for r in COUNTER_ROWS]
        data["Arch_VGPR"] = [r[9] for r in COUNTER_ROWS]
        data["Accum_VGPR"] = [r[10] for r in COUNTER_ROWS]
        data["SGPR"] = [r[11] for r in COUNTER_ROWS]
        data["Kernel_ID"] = [r[15] for r in COUNTER_ROWS]

    return pd.DataFrame(data)


def write_rocpd_layout(workload_dir, fbase="run0"):
    """Write marker/counter CSVs at workload root (rocpd layout)."""
    marker_df = build_marker_df(include_guid=True)
    counter_df = build_counter_df(include_guid=True)

    marker_path = Path(workload_dir) / f"torch_trace_{fbase}_marker_api_trace.csv"
    counter_path = Path(workload_dir) / f"torch_trace_{fbase}_counter_collection.csv"

    marker_df.to_csv(marker_path, index=False)
    counter_df.to_csv(counter_path, index=False)


def write_csv_layout(workload_dir, fbase="run0", pid="12345"):
    """Write marker/counter CSVs in a subdirectory (csv layout)."""
    subdir = Path(workload_dir) / fbase
    subdir.mkdir(parents=True, exist_ok=True)

    marker_df = build_marker_df(include_guid=False)
    counter_df = build_counter_df(include_guid=False)

    marker_path = subdir / f"torch_trace_{pid}_marker_api_trace.csv"
    counter_path = subdir / f"torch_trace_{pid}_counter_collection.csv"

    marker_df.to_csv(marker_path, index=False)
    counter_df.to_csv(counter_path, index=False)


def read_operator_csvs(torch_trace_dir):
    """Return a dict mapping filename -> sorted DataFrame for comparison."""
    result = {}

    for csv_file in sorted(Path(torch_trace_dir).glob("*.csv")):
        df = pd.read_csv(csv_file)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        result[csv_file.name] = df

    return result


def test_torch_trace_output_same_for_rocpd_and_csv():
    """Test that the torch trace output is the same for rocpd and csv files."""
    rocpd_dir = test_utils.get_output_dir(suffix="_rocpd")
    csv_dir = test_utils.get_output_dir(suffix="_csv")

    Path(rocpd_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    write_rocpd_layout(rocpd_dir)
    write_csv_layout(csv_dir)

    process_torch_trace_output(rocpd_dir)
    process_torch_trace_output(csv_dir)

    rocpd_results = read_operator_csvs(Path(rocpd_dir) / "torch_trace")
    csv_results = read_operator_csvs(Path(csv_dir) / "torch_trace")

    assert rocpd_results.keys() == csv_results.keys(), (
        f"Operator files differ: rocpd={sorted(rocpd_results.keys())} "
        f"csv={sorted(csv_results.keys())}"
    )

    for filename in rocpd_results:
        pd.testing.assert_frame_equal(
            rocpd_results[filename],
            csv_results[filename],
            check_dtype=False,
            obj=filename,
        )

    test_utils.clean_output_dir(True, rocpd_dir)
    test_utils.clean_output_dir(True, csv_dir)
