# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import json
import sqlite3
from pathlib import Path

import common
import pandas as pd

from utils.rocpd_data import (
    COUNTERS_COLLECTION_QUERY,
    MARKER_API_TRACE_QUERY,
    convert_dbs_to_csv,
)
from utils.utils_analysis import (
    build_call_trees_with_kernel_ids,
    process_ml_api_trace_output,
    write_ml_api_trace_consolidated_csv,
)
from utils.utils_profile import _augment_marker_csv, _parse_function_backend

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
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    counter_csv = str(Path(workload_dir) / "counter_collection.csv")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv")

    db_path = create_rocpd_test_db(workload_dir)
    convert_dbs_to_csv([db_path], counter_csv, marker_csv)

    df = pd.read_csv(counter_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = [row[2] for row in COUNTER_ROWS]
    assert list(df["Correlation_Id"]) == expected_ids

    common.clean_output_dir(True, workload_dir)


def test_marker_csv_has_correlation_id_from_stack_id():
    """Test that the marker CSV has correlation_id from stack_id."""
    workload_dir = common.get_output_dir()
    Path(workload_dir).mkdir(parents=True, exist_ok=True)

    counter_csv = str(Path(workload_dir) / "counter_collection.csv")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv")

    db_path = create_rocpd_test_db(workload_dir)
    convert_dbs_to_csv([db_path], counter_csv, marker_csv)

    df = pd.read_csv(marker_csv)
    assert "Correlation_Id" in df.columns

    expected_ids = sorted(row[4] for row in MARKER_ROWS)
    assert sorted(df["Correlation_Id"].tolist()) == expected_ids

    common.clean_output_dir(True, workload_dir)


# ---- process_ml_api_trace_output parity for rocpd vs csv layouts ----


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

    marker_path = Path(workload_dir) / f"ml_api_trace_{fbase}_marker_api_trace.csv"
    counter_path = Path(workload_dir) / f"ml_api_trace_{fbase}_counter_collection.csv"

    marker_df.to_csv(marker_path, index=False)
    counter_df.to_csv(counter_path, index=False)


def write_csv_layout(workload_dir, fbase="run0", pid="12345"):
    """Write marker/counter CSVs in a subdirectory (csv layout)."""
    subdir = Path(workload_dir) / fbase
    subdir.mkdir(parents=True, exist_ok=True)

    marker_df = build_marker_df(include_guid=False)
    counter_df = build_counter_df(include_guid=False)

    marker_path = subdir / f"ml_api_trace_{pid}_marker_api_trace.csv"
    counter_path = subdir / f"ml_api_trace_{pid}_counter_collection.csv"

    marker_df.to_csv(marker_path, index=False)
    counter_df.to_csv(counter_path, index=False)


def read_ml_api_trace_csvs(ml_api_trace_dir):
    """Return a dict mapping filename -> sorted DataFrame for comparison."""
    result = {}

    for csv_file in sorted(Path(ml_api_trace_dir).glob("*.csv")):
        df = pd.read_csv(csv_file)
        df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        result[csv_file.name] = df

    return result


def build_kernel_top_df():
    """Build a kernel top stats DataFrame matching the test kernel names."""
    return pd.DataFrame({
        "Kernel_Name": sorted({r[12] for r in COUNTER_ROWS}),
    })


def test_ml_api_trace_output_same_for_rocpd_and_csv():
    """Test that the ML API trace output is the same for rocpd and csv files."""
    rocpd_dir = common.get_output_dir(suffix="_rocpd")
    csv_dir = common.get_output_dir(suffix="_csv")

    Path(rocpd_dir).mkdir(parents=True, exist_ok=True)
    Path(csv_dir).mkdir(parents=True, exist_ok=True)

    write_rocpd_layout(rocpd_dir)
    write_csv_layout(csv_dir)

    kernel_top_df = build_kernel_top_df()
    rocpd_output = process_ml_api_trace_output(rocpd_dir)
    csv_output = process_ml_api_trace_output(csv_dir)
    assert rocpd_output is not None
    assert csv_output is not None
    rocpd_df, rocpd_trace_path = rocpd_output
    csv_df, csv_trace_path = csv_output

    write_ml_api_trace_consolidated_csv(rocpd_df, rocpd_trace_path)
    write_ml_api_trace_consolidated_csv(csv_df, csv_trace_path)
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

    rocpd_results = read_ml_api_trace_csvs(Path(rocpd_dir) / "ml_api_trace")
    csv_results = read_ml_api_trace_csvs(Path(csv_dir) / "ml_api_trace")

    assert rocpd_results.keys() == csv_results.keys(), (
        f"ML API trace CSV files differ: rocpd={sorted(rocpd_results.keys())} "
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


# ---- Backend column unpacking in save_ml_api_trace_inputs ----


def test_parse_function_backend_untagged_is_unknown():
    """Untagged rows surface as Backend='unknown'."""
    clean, backend = _parse_function_backend("torch.empty:#1@linear.py:109")
    assert clean == "torch.empty:#1@linear.py:109"
    assert backend == "unknown"


def test_parse_function_backend_tagged_torch_is_stripped():
    """Tagged single-frame markers expose backend and lose the suffix."""
    clean, backend = _parse_function_backend(
        "nn.Module.MyModel.forward:#1@train.py:42|torch"
    )
    assert clean == "nn.Module.MyModel.forward:#1@train.py:42"
    assert backend == "torch"


def test_parse_function_backend_tagged_triton_leaf():
    """Row-level suffix attributes the entire wire to its producing backend."""
    clean, backend = _parse_function_backend(
        "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2|triton"
    )
    assert clean == ("torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2")
    assert backend == "triton"


def test_parse_function_backend_aten_leaf_is_unknown():
    """Untagged ATen leaf surfaces as Backend='unknown'."""
    clean, backend = _parse_function_backend(
        "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0"
    )
    assert clean == "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0"
    assert backend == "unknown"


def test_parse_function_backend_edge_cases():
    """Bogus suffix, empty string, and None all fall back to 'unknown'."""
    assert _parse_function_backend("op|bogus") == ("op|bogus", "unknown")
    assert _parse_function_backend("") == ("", "unknown")
    assert _parse_function_backend(None) == ("", "unknown")


def test_augment_marker_csv_untagged_row_warns(tmp_path, monkeypatch):
    """Untagged rows are tagged 'unknown' and emit a warning."""
    from utils import utils_profile

    src = tmp_path / "src_marker_api_trace.csv"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv"
    pd.DataFrame({"Function": ["aten::sum"]}).to_csv(src, index=False)

    warnings: list[tuple] = []
    monkeypatch.setattr(utils_profile, "console_warning", lambda *a: warnings.append(a))

    _augment_marker_csv(str(src), str(dst))

    out_df = pd.read_csv(dst)
    assert out_df["Function"].tolist() == ["aten::sum"]
    assert out_df["Backend"].tolist() == ["unknown"]
    assert warnings, "untagged rows must emit a warning"
    assert any("unknown" in str(a) for a in warnings[0])


def test_augment_marker_csv_adds_backend_column(tmp_path):
    """End-to-end: tagged + untagged rows survive copy; Backend is populated."""
    src = tmp_path / "src_marker_api_trace.csv"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv"

    src_df = pd.DataFrame({
        "Domain": ["MARKER_CORE_RANGE_API"] * 3,
        "Function": [
            "nn.Module.X.forward:#1@a.py:1|torch",
            "triton.CompiledKernel.k:#1@b.py:2|triton",
            "torch.empty:#1@c.py:3",
        ],
        "Correlation_Id": [1, 2, 3],
        "Start_Timestamp": [100, 200, 300],
        "End_Timestamp": [150, 250, 350],
    })
    src_df.to_csv(src, index=False)

    _augment_marker_csv(str(src), str(dst))

    out_df = pd.read_csv(dst)
    assert "Backend" in out_df.columns
    assert out_df["Backend"].tolist() == ["torch", "triton", "unknown"]
    assert out_df["Function"].tolist() == [
        "nn.Module.X.forward:#1@a.py:1",
        "triton.CompiledKernel.k:#1@b.py:2",
        "torch.empty:#1@c.py:3",
    ]
    for col in ("Domain", "Correlation_Id", "Start_Timestamp", "End_Timestamp"):
        assert col in out_df.columns


def test_augment_marker_csv_handles_unknown_schema(tmp_path):
    """A CSV without a Function column copies verbatim instead of corrupting."""
    src = tmp_path / "src.csv"
    dst = tmp_path / "dst.csv"
    src.write_text("Foo,Bar\n1,2\n3,4\n", encoding="utf-8")

    _augment_marker_csv(str(src), str(dst))

    assert dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


def test_process_ml_api_trace_output_defaults_backend_for_untagged(tmp_path):
    """Untagged rows default to Backend='torch' in the consolidated df."""
    workload_dir = str(tmp_path)
    write_rocpd_layout(workload_dir)

    consolidated_df, _ = process_ml_api_trace_output(workload_dir)

    assert "Backend" in consolidated_df.columns
    assert (consolidated_df["Backend"] == "torch").all()


def test_process_ml_api_trace_output_preserves_per_row_backend(tmp_path):
    """A pre-stripped + tagged CSV (as produced by _augment_marker_csv)
    surfaces the per-row Backend value into the consolidated dataframe.
    """
    workload_dir = str(tmp_path)
    write_rocpd_layout(workload_dir)

    # Overwrite the fixture with what save_ml_api_trace_inputs would produce:
    # Function has prefixes stripped, Backend carries per-row attribution.
    marker_path = Path(workload_dir) / "ml_api_trace_run0_marker_api_trace.csv"
    df = pd.read_csv(marker_path)
    df["Backend"] = ["torch", "torch", "triton"]
    df.to_csv(marker_path, index=False)

    consolidated_df, _ = process_ml_api_trace_output(workload_dir)

    assert "Backend" in consolidated_df.columns
    backend_by_operator = dict(
        zip(consolidated_df["Operator_Name"], consolidated_df["Backend"])
    )
    assert backend_by_operator.get("torch.mm") == "triton"
    assert backend_by_operator.get("nn.Module.Linear.forward") == "torch"
