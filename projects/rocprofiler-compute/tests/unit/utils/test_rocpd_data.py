# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import sqlite3
from pathlib import Path

import common
import pandas as pd

from utils.rocpd_data import (
    COUNTERS_COLLECTION_QUERY,
    KERNEL_SYMBOLS_QUERY,
    MARKER_API_TRACE_QUERY,
    convert_dbs_to_csv,
)

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

# (display_name, truncated_kernel_name), as the kernel_symbols view exposes
# them. A symbol repeats per process, and rocclr symbols keep their .kd.
KERNEL_SYMBOL_ROWS = [
    ("kernel_gemm(float*, float*)", "kernel_gemm"),
    ("kernel_mm(float*)", "kernel_mm"),
    ("kernel_gemm(float*, float*)", "kernel_gemm"),
    ("__amd_rocclr_copyBuffer", "__amd_rocclr_copyBuffer.kd"),
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


def test_kernel_symbols_query_reads_the_kernel_symbols_view():
    """The short name is read at its own grain: one row per kernel symbol."""
    assert "FROM kernel_symbols" in KERNEL_SYMBOLS_QUERY
    assert "display_name as Kernel_Name" in KERNEL_SYMBOLS_QUERY
    assert "truncated_kernel_name as Kernel_Short_Name" in KERNEL_SYMBOLS_QUERY


def test_kernel_symbols_query_orders_its_rows():
    """Sorting keeps the CSV byte-identical across runs and SQLite versions."""
    assert "ORDER BY Kernel_Name, Kernel_Short_Name" in KERNEL_SYMBOLS_QUERY


def test_counters_query_carries_no_short_name():
    """The counter query streams row by row, so it stays at counter grain."""
    assert "Kernel_Short_Name" not in COUNTERS_COLLECTION_QUERY
    assert "truncated_kernel_name" not in COUNTERS_COLLECTION_QUERY


# ---- Test 2: convert_dbs_to_csv populates Correlation_Id from stack_id ----


def create_rocpd_test_db(workload_dir):
    """
    Build a minimal rocpd-style SQLite database with counters_collection,
    regions, and kernel_symbols tables whose schemas match the production
    queries.
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
    conn.execute(
        """CREATE TABLE kernel_symbols (
            display_name TEXT, truncated_kernel_name TEXT
        )"""
    )
    conn.executemany(
        "INSERT INTO kernel_symbols VALUES (?,?)",
        KERNEL_SYMBOL_ROWS,
    )
    conn.commit()
    conn.close()
    return db_path


def convert_test_db(workload_dir):
    """Run the conversion over one test db, returning the three output paths."""
    output_paths = tuple(
        str(Path(workload_dir) / name)
        for name in (
            "counter_collection.csv.gz",
            "marker_api_trace.csv.gz",
            "kernel_symbols.csv.gz",
        )
    )
    convert_dbs_to_csv([create_rocpd_test_db(workload_dir)], *output_paths)
    return output_paths


def test_counter_csv_has_correlation_id_from_stack_id():
    """Test that the counter CSV has correlation_id from stack_id."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    counter_csv, _, _ = convert_test_db(workload_dir)

    df = pd.read_csv(counter_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = [row[2] for row in COUNTER_ROWS]
    assert list(df["Correlation_Id"]) == expected_ids

    common.clean_output_dir(True, workload_dir)


def test_marker_csv_has_correlation_id_from_stack_id():
    """Test that the marker CSV has correlation_id from stack_id."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    _, marker_csv, _ = convert_test_db(workload_dir)

    df = pd.read_csv(marker_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = sorted(row[4] for row in MARKER_ROWS)
    assert sorted(df["Correlation_Id"].tolist()) == expected_ids

    common.clean_output_dir(True, workload_dir)


def test_kernel_symbols_csv_pairs_each_kernel_name_with_its_short_name():
    """The conversion writes a third file, at one row per kernel symbol."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    _, _, kernel_symbols_csv = convert_test_db(workload_dir)

    df = pd.read_csv(kernel_symbols_csv)
    assert list(df.columns) == ["Kernel_Name", "Kernel_Short_Name"]
    # The query sorts, so the CSV holds the symbols in name order.
    assert list(zip(df["Kernel_Name"], df["Kernel_Short_Name"])) == sorted(
        KERNEL_SYMBOL_ROWS
    )

    common.clean_output_dir(True, workload_dir)
