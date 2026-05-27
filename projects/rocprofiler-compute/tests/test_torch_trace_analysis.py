# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import sqlite3
from pathlib import Path

import common
import pandas as pd
import pytest

from utils.file_io import write_pmc_perf_from_rocpd
from utils.rocpd_data import (
    COUNTERS_COLLECTION_QUERY,
    MARKER_API_TRACE_QUERY,
    build_pass_db,
    get_rocpd_pass_db_paths,
    read_counter_collection_rows,
    read_marker_api_trace_rows,
)
from utils.utils_analysis import (
    build_call_trees_with_kernel_ids,
    normalize_rocpd_counter_dataframe,
    process_torch_trace_output,
    write_torch_trace_consolidated_csv,
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


# ---- Test 2: rocpd reads populate Correlation_Id from stack_id ----


def create_rocpd_test_db(
    workload_dir,
    db_name="test.db",
    counter_rows=None,
    include_regions=True,
):
    """
    Build a minimal rocpd-style SQLite database with counters_collection
    and regions tables whose schemas match the production queries.
    """
    rows = COUNTER_ROWS if counter_rows is None else counter_rows
    db_path = str(Path(workload_dir) / db_name)
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
        rows,
    )
    if include_regions:
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


def create_perfmon_pass_config(workload_dir, pass_name="pmc_perf_0"):
    """Create a minimal perfmon pass config matching a root pass DB name."""
    perfmon_dir = Path(workload_dir) / "perfmon"
    perfmon_dir.mkdir(parents=True, exist_ok=True)
    pass_config_path = perfmon_dir / f"{pass_name}.yaml"
    pass_config_path.write_text("pmc: []\n")
    return pass_config_path


def test_counter_rows_have_correlation_id_from_stack_id():
    """Test that counter rows have correlation_id from stack_id."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    db_path = create_rocpd_test_db(workload_dir)
    rows = read_counter_collection_rows([db_path])

    df = pd.DataFrame(rows)
    assert "Correlation_Id" in df.columns

    expected_ids = [row[2] for row in COUNTER_ROWS]
    assert list(df["Correlation_Id"]) == expected_ids

    common.clean_output_dir(True, workload_dir)


def test_marker_rows_have_correlation_id_from_stack_id():
    """Test that marker rows have correlation_id from stack_id."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    db_path = create_rocpd_test_db(workload_dir)
    rows = read_marker_api_trace_rows([db_path])

    df = pd.DataFrame(rows)
    assert "Correlation_Id" in df.columns

    expected_ids = sorted(row[4] for row in MARKER_ROWS)
    assert sorted(df["Correlation_Id"].tolist()) == expected_ids

    common.clean_output_dir(True, workload_dir)


def test_build_pass_db_preserves_rocpd_query_surfaces():
    """Test that pass DB creation keeps query surfaces used by analysis."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    db_path = create_rocpd_test_db(workload_dir)
    pass_db_path = str(Path(workload_dir) / "pmc_perf_0.db")
    build_pass_db([db_path], pass_db_path)

    counter_rows = read_counter_collection_rows([pass_db_path])
    marker_rows = read_marker_api_trace_rows([pass_db_path])

    assert len(counter_rows) == len(COUNTER_ROWS)
    assert len(marker_rows) == len(MARKER_ROWS)

    common.clean_output_dir(True, workload_dir)


def test_build_pass_db_merges_multiple_rocpd_databases():
    """Test same-pass DB merge across multiple profiler-produced DB files."""
    workload_dir = common.get_output_dir()
    first_dir = Path(workload_dir) / "first"
    second_dir = Path(workload_dir) / "second"
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)

    first_db_path = create_rocpd_test_db(first_dir)
    second_db_path = create_rocpd_test_db(second_dir)
    pass_db_path = str(Path(workload_dir) / "pmc_perf_0.db")
    build_pass_db([first_db_path, second_db_path], pass_db_path)

    counter_rows = read_counter_collection_rows([pass_db_path])
    marker_rows = read_marker_api_trace_rows([pass_db_path])

    assert len(counter_rows) == len(COUNTER_ROWS) * 2
    assert len(marker_rows) == len(MARKER_ROWS) * 2

    common.clean_output_dir(True, workload_dir)


def test_build_pass_db_tolerates_missing_regions_surface():
    """Test pass DB creation keeps counters when optional regions are absent."""
    workload_dir = common.get_output_dir()
    first_dir = Path(workload_dir) / "first"
    second_dir = Path(workload_dir) / "second"
    first_dir.mkdir(parents=True, exist_ok=True)
    second_dir.mkdir(parents=True, exist_ok=True)

    first_db_path = create_rocpd_test_db(first_dir, include_regions=False)
    second_db_path = create_rocpd_test_db(second_dir)
    pass_db_path = str(Path(workload_dir) / "pmc_perf_0.db")
    build_pass_db([first_db_path, second_db_path], pass_db_path)

    counter_rows = read_counter_collection_rows([pass_db_path])
    marker_rows = read_marker_api_trace_rows([pass_db_path])

    assert len(counter_rows) == len(COUNTER_ROWS) * 2
    assert len(marker_rows) == len(MARKER_ROWS)

    common.clean_output_dir(True, workload_dir)


def test_write_pmc_perf_from_rocpd_loads_database_without_results_csv():
    """Test that analyze can build pmc_perf.csv directly from a rocpd database."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_perfmon_pass_config(workload_dir)
    create_rocpd_test_db(workload_dir, db_name="pmc_perf_0.db")
    pmc_perf_path = Path(workload_dir) / "pmc_perf.csv"

    returned_df = write_pmc_perf_from_rocpd(workload_dir, str(pmc_perf_path))
    assert returned_df is not None

    pmc_df = pd.read_csv(pmc_perf_path)
    assert returned_df.equals(pmc_df)
    assert "SQ_WAVES" in pmc_df.columns
    assert "Counter_Name" not in pmc_df.columns
    assert list(pmc_df["Dispatch_ID"]) == [0, 1, 2]

    common.clean_output_dir(True, workload_dir)


def test_get_rocpd_pass_db_paths_uses_perfmon_pass_names():
    """Test rocpd discovery ignores unrelated root DB files."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_perfmon_pass_config(workload_dir)
    expected_db_path = Path(create_rocpd_test_db(workload_dir, db_name="pmc_perf_0.db"))
    create_rocpd_test_db(workload_dir, db_name="unrelated.db")

    assert get_rocpd_pass_db_paths(Path(workload_dir)) == [expected_db_path]

    common.clean_output_dir(True, workload_dir)


def test_write_pmc_perf_from_rocpd_ignores_unmatched_database():
    """Test analyze does not consume arbitrary root SQLite databases."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_rocpd_test_db(workload_dir, db_name="unrelated.db")
    pmc_perf_path = Path(workload_dir) / "pmc_perf.csv"

    assert write_pmc_perf_from_rocpd(workload_dir, str(pmc_perf_path)) is None
    assert not pmc_perf_path.exists()

    common.clean_output_dir(True, workload_dir)


def test_write_pmc_perf_from_rocpd_rejects_empty_counter_database():
    """Test empty pass DBs do not become successful pmc_perf.csv artifacts."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_perfmon_pass_config(workload_dir)
    create_rocpd_test_db(
        workload_dir,
        db_name="pmc_perf_0.db",
        counter_rows=[],
    )
    pmc_perf_path = Path(workload_dir) / "pmc_perf.csv"

    assert write_pmc_perf_from_rocpd(workload_dir, str(pmc_perf_path)) is None
    assert not pmc_perf_path.exists()

    common.clean_output_dir(True, workload_dir)


def test_write_pmc_perf_from_rocpd_normalizes_dispatch_ids_per_pass():
    """Dispatch_IDs restart per pass within each kernel group.

    rocprof can split counter collection across multiple passes when the
    requested set does not fit in a single pass. Each pass re-runs the same
    workload, so the i-th invocation of a kernel in pass 0 corresponds to
    the i-th invocation in pass 1 (logical kernel identity). Numbering
    Dispatch_IDs per pass ensures process_rocpd_csv's pivot can merge the
    counter columns from every pass into one wide row per logical kernel
    invocation.
    """
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_perfmon_pass_config(workload_dir, "pmc_perf_0")
    create_perfmon_pass_config(workload_dir, "pmc_perf_1")
    second_process_rows = [
        (*row[:4], 200, *row[5:16], "GRBM_COUNT", row[17] + 100) for row in COUNTER_ROWS
    ]
    create_rocpd_test_db(workload_dir, db_name="pmc_perf_0.db")
    create_rocpd_test_db(
        workload_dir,
        db_name="pmc_perf_1.db",
        counter_rows=second_process_rows,
    )
    pmc_perf_path = Path(workload_dir) / "pmc_perf.csv"

    returned_df = write_pmc_perf_from_rocpd(workload_dir, str(pmc_perf_path))
    assert returned_df is not None

    pmc_df = pd.read_csv(pmc_perf_path)
    assert returned_df.equals(pmc_df)
    # Counters from both pass DBs are joined into one row per logical
    # dispatch, rather than appended as another set of long-format rows.
    assert list(pmc_df["Dispatch_ID"]) == [0, 1, 2]
    assert list(pmc_df["SQ_WAVES"]) == [42, 50, 30]
    assert list(pmc_df["GRBM_COUNT"]) == [142, 150, 130]

    common.clean_output_dir(True, workload_dir)


def test_normalize_rocpd_counter_dataframe_aligns_dispatches_across_passes():
    """Per-pass Dispatch_ID + stable Kernel_ID across multiple passes.

    Inputs a synthetic 3-pass long-form DataFrame where each pass has the
    same 2 logical kernels invoked twice each (4 dispatches per pass, 12
    rows total: 2 kernels x 2 invocations x 3 passes). Asserts:

    - Dispatch_ID is the per-kernel-group, per-pass cumcount (0..1 for
      each (Pass_ID, Kernel_Name) combination).
    - Kernel_ID is stable across passes for the same kernel group.
    - Pass_ID column is dropped from the returned frame.
    - Input frame is not mutated.
    """
    rows = []
    for pass_id in range(3):
        # kernel_a: 2 invocations
        for invocation in range(2):
            rows.append({
                "Pass_ID": pass_id,
                "Kernel_Name": "kernel_a",
                "Grid_Size": 256,
                "Workgroup_Size": 64,
                "LDS_Per_Workgroup": 0,
                "Start_Timestamp": 1000 + invocation * 100,
                "End_Timestamp": 1050 + invocation * 100,
                "Counter_Name": "SQ_WAVES",
                "Counter_Value": 10 + pass_id,
            })
        # kernel_b: 2 invocations
        for invocation in range(2):
            rows.append({
                "Pass_ID": pass_id,
                "Kernel_Name": "kernel_b",
                "Grid_Size": 512,
                "Workgroup_Size": 128,
                "LDS_Per_Workgroup": 0,
                "Start_Timestamp": 2000 + invocation * 100,
                "End_Timestamp": 2050 + invocation * 100,
                "Counter_Name": "SQ_WAVES",
                "Counter_Value": 20 + pass_id,
            })
    input_df = pd.DataFrame(rows)
    input_df_before = input_df.copy()

    normalized = normalize_rocpd_counter_dataframe(input_df)

    # input not mutated
    assert input_df.equals(input_df_before)
    # Pass_ID dropped from output
    assert "Pass_ID" not in normalized.columns
    assert "Dispatch_ID" in normalized.columns
    assert "Kernel_ID" in normalized.columns

    # Dispatch_ID is per-kernel, per-pass cumcount across the unique dispatches.
    # Within each pass: kernel_a dispatch 0 and 1, kernel_b dispatch 0 and 1.
    for kernel in ("kernel_a", "kernel_b"):
        ids = sorted(
            normalized
            .loc[normalized["Kernel_Name"] == kernel, "Dispatch_ID"]
            .unique()
            .tolist()
        )
        assert ids == [0, 1], f"{kernel} Dispatch_IDs were {ids}"

    # Kernel_ID is stable per kernel-group across all 3 passes.
    kernel_a_ids = normalized.loc[
        normalized["Kernel_Name"] == "kernel_a", "Kernel_ID"
    ].unique()
    kernel_b_ids = normalized.loc[
        normalized["Kernel_Name"] == "kernel_b", "Kernel_ID"
    ].unique()
    assert len(kernel_a_ids) == 1, kernel_a_ids
    assert len(kernel_b_ids) == 1, kernel_b_ids
    assert kernel_a_ids[0] != kernel_b_ids[0]


def test_check_profile_output_files_rejects_unmatched_rocpd_database():
    """Test profile validation does not accept arbitrary root DB files."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_rocpd_test_db(workload_dir, db_name="unrelated.db")

    with pytest.raises(AssertionError):
        common.check_profile_output_files(workload_dir, 1, 1)

    common.clean_output_dir(True, workload_dir)


def test_check_profile_output_files_validates_rocpd_counter_rows():
    """Test rocpd DB validation mirrors the CSV minimum-row check."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    create_perfmon_pass_config(workload_dir)
    create_rocpd_test_db(workload_dir, db_name="pmc_perf_0.db")

    common.check_profile_output_files(workload_dir, 1, len(COUNTER_ROWS))
    with pytest.raises(AssertionError):
        common.check_profile_output_files(workload_dir, 1, len(COUNTER_ROWS) + 1)

    common.clean_output_dir(True, workload_dir)


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


def read_torch_trace_csvs(torch_trace_dir):
    """Return a dict mapping filename -> sorted DataFrame for comparison."""
    result = {}

    for csv_file in sorted(Path(torch_trace_dir).glob("*.csv")):
        df = pd.read_csv(csv_file)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        result[csv_file.name] = df

    return result


def build_kernel_top_df():
    """Build a kernel top stats DataFrame matching the test kernel names."""
    return pd.DataFrame({
        "Kernel_Name": sorted({r[12] for r in COUNTER_ROWS}),
    })


def test_torch_trace_output_same_for_rocpd_and_csv():
    """Test that the torch trace output is the same for rocpd and csv files."""
    rocpd_dir = common.get_output_dir(suffix="_rocpd")
    csv_dir = common.get_output_dir(suffix="_csv")

    Path(rocpd_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    write_rocpd_layout(rocpd_dir)
    write_csv_layout(csv_dir)

    kernel_top_df = build_kernel_top_df()
    rocpd_output = process_torch_trace_output(rocpd_dir)
    csv_output = process_torch_trace_output(csv_dir)
    assert rocpd_output is not None
    assert csv_output is not None
    rocpd_df, rocpd_trace_path = rocpd_output
    csv_df, csv_trace_path = csv_output

    write_torch_trace_consolidated_csv(rocpd_df, rocpd_trace_path)
    write_torch_trace_consolidated_csv(csv_df, csv_trace_path)
    rocpd_trees = build_call_trees_with_kernel_ids(rocpd_df, kernel_top_df)
    csv_trees = build_call_trees_with_kernel_ids(csv_df, kernel_top_df)

    for trees in (rocpd_trees, csv_trees):
        assert "test.py:10" in trees
        assert "test.py:15" in trees

        linear_root = trees["test.py:10"]
        assert linear_root.kernel_launches == 2
        assert "nn.Module.Linear.forward" in linear_root.children
        linear_node = linear_root.children["nn.Module.Linear.forward"]
        assert "kernel_gemm" in linear_node.kernels
        assert linear_node.kernels["kernel_gemm"].launches == 2

        mm_root = trees["test.py:15"]
        assert mm_root.kernel_launches == 1
        assert "torch.mm" in mm_root.children
        mm_node = mm_root.children["torch.mm"]
        assert "kernel_mm" in mm_node.kernels
        assert mm_node.kernels["kernel_mm"].launches == 1

    rocpd_results = read_torch_trace_csvs(Path(rocpd_dir) / "torch_trace")
    csv_results = read_torch_trace_csvs(Path(csv_dir) / "torch_trace")

    assert rocpd_results.keys() == csv_results.keys(), (
        f"Torch trace CSV files differ: rocpd={sorted(rocpd_results.keys())} "
        f"csv={sorted(csv_results.keys())}"
    )

    for filename in rocpd_results:
        pd.testing.assert_frame_equal(
            rocpd_results[filename],
            csv_results[filename],
            check_dtype=False,
            obj=filename,
        )

    common.clean_output_dir(True, rocpd_dir)
    common.clean_output_dir(True, csv_dir)
