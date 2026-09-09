# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import gzip
import json
import sqlite3
from pathlib import Path

import common
import pandas as pd
import pytest

from utils.rocpd_data import (
    COUNTERS_COLLECTION_QUERY,
    MARKER_API_TRACE_QUERY,
    convert_dbs_to_csv,
)
from utils.utils_analysis import (
    build_call_trees,
    build_call_trees_with_kernel_ids,
    format_operator_args,
    process_ml_api_trace_output,
    split_operator_args,
    write_ml_api_trace_consolidated_csv,
)
from utils.utils_profile import (
    _augment_marker_csv,
    _parse_function_fields,
    save_ml_api_trace_inputs,
)

GUID = "abc-1234-def"


def _write_gzip_csv(path, df):
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        df.to_csv(f, index=False)


# Function values carry the wire "|<backend>" suffix.
MARKER_ROWS = [
    (
        "roctx",
        "nn.Module.Linear.forward:#1@test.py:10|torch",
        100,
        200,
        1000,
        GUID,
        1000,
        2000,
    ),
    (
        "roctx",
        "nn.Module.Linear.forward:#2@test.py:10|torch",
        100,
        200,
        1001,
        GUID,
        3000,
        4000,
    ),
    ("roctx", "torch.mm:#1@test.py:15|torch", 100, 200, 1002, GUID, 5000, 6000),
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

    counter_csv = str(Path(workload_dir) / "counter_collection.csv.gz")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv.gz")

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

    counter_csv = str(Path(workload_dir) / "counter_collection.csv.gz")
    marker_csv = str(Path(workload_dir) / "marker_api_trace.csv.gz")

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
    """Build the augmented marker dataframe, splitting Function and Backend from
    each raw marker value."""
    parsed = [_parse_function_fields(r[1]) for r in MARKER_ROWS]
    data = {
        "Domain": [r[0] for r in MARKER_ROWS],
        "Function": [function for function, _backend, _args in parsed],
        "Process_Id": [r[2] for r in MARKER_ROWS],
        "Thread_Id": [r[3] for r in MARKER_ROWS],
        "Correlation_Id": [r[4] for r in MARKER_ROWS],
        "Start_Timestamp": [r[6] for r in MARKER_ROWS],
        "End_Timestamp": [r[7] for r in MARKER_ROWS],
        "Backend": [backend for _function, backend, _args in parsed],
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

    marker_path = Path(workload_dir) / f"ml_api_trace_{fbase}_marker_api_trace.csv.gz"
    counter_path = (
        Path(workload_dir) / f"ml_api_trace_{fbase}_counter_collection.csv.gz"
    )

    _write_gzip_csv(marker_path, marker_df)
    _write_gzip_csv(counter_path, counter_df)


def write_csv_layout(workload_dir, fbase="run0", pid="12345"):
    """Write marker/counter CSVs in a subdirectory (csv layout)."""
    subdir = Path(workload_dir) / fbase
    subdir.mkdir(parents=True, exist_ok=True)

    marker_df = build_marker_df(include_guid=False)
    counter_df = build_counter_df(include_guid=False)

    marker_path = subdir / f"ml_api_trace_{pid}_marker_api_trace.csv.gz"
    counter_path = subdir / f"ml_api_trace_{pid}_counter_collection.csv.gz"

    _write_gzip_csv(marker_path, marker_df)
    _write_gzip_csv(counter_path, counter_df)


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


def test_ml_api_trace_counter_copy_stays_compressed(tmp_path):
    """Counter copy stays compressed when results_*.csv is .csv.gz."""
    src_counter = tmp_path / "results_run0.csv.gz"
    with gzip.open(src_counter, "wt", newline="") as f:
        build_counter_df(include_guid=True).to_csv(f, index=False)
    src_dir = tmp_path / "out" / "pmc_1"
    src_dir.mkdir(parents=True)
    build_marker_df(include_guid=True).to_csv(
        src_dir / "run0_marker_api_trace.csv.gz", index=False
    )

    save_ml_api_trace_inputs(str(tmp_path), "run0", src_counter)

    dst = tmp_path / "ml_api_trace_run0_counter_collection.csv.gz"
    assert dst.is_file(), "counter copy should keep the compressed name"
    assert dst.read_bytes() == src_counter.read_bytes(), "should be a byte copy"

    consolidated_df, _ = process_ml_api_trace_output(str(tmp_path))
    assert not consolidated_df.empty, "analyze must resolve the compressed copy"


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


# ---- Function cell parsing in save_ml_api_trace_inputs ----


@pytest.mark.parametrize(
    "raw, expect_function, expect_backend, expect_args",
    [
        ("aten::add", "aten::add", "unknown", ""),
        ("aten::mm|torch", "aten::mm", "torch", ""),
        # Untagged single-frame marker with context.
        (
            "torch.empty:#1@linear.py:109",
            "torch.empty:#1@linear.py:109",
            "unknown",
            "",
        ),
        # Tagged single-frame marker: suffix stripped, backend exposed.
        (
            "nn.Module.MyModel.forward:#1@train.py:42|torch",
            "nn.Module.MyModel.forward:#1@train.py:42",
            "torch",
            "",
        ),
        # Multi-frame leaf attributed to its producing backend.
        (
            "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2|triton",
            "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2",
            "triton",
            "",
        ),
        # Untagged multi-frame ATen leaf.
        (
            "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0",
            "nn.Module.X.forward/aten::add:#1@m.py:9/#1@aten:0",
            "unknown",
            "",
        ),
        # Unrecognized suffix, empty string, and None fall back to "unknown".
        ("op|bogus", "op|bogus", "unknown", ""),
        ("", "", "unknown", ""),
        (None, "", "unknown", ""),
        (
            "aten::mm:#1@aten:0|args=(self=float32[2x3], mat2=float32[3x4])|torch",
            "aten::mm:#1@aten:0",
            "torch",
            "(self=float32[2x3], mat2=float32[3x4])",
        ),
        (
            "aten::relu:#1@aten:0|args=(self=float32[5x20])|torch",
            "aten::relu:#1@aten:0",
            "torch",
            "(self=float32[5x20])",
        ),
        (
            "aten::cat:#1@aten.nested:0"
            "|args=(tensors=[float32[2x3], float32[2x3]], dim=Int)|torch",
            "aten::cat:#1@aten.nested:0",
            "torch",
            "(tensors=[float32[2x3], float32[2x3]], dim=Int)",
        ),
        (
            "aten::mm:#1@m.py:7|args=(f32[2x2])|torch",
            "aten::mm:#1@m.py:7",
            "torch",
            "(f32[2x2])",
        ),
        (
            "triton.k:#1@m.py:7|args=(x_ptr=f32[8], n=1024)|triton",
            "triton.k:#1@m.py:7",
            "triton",
            "(x_ptr=f32[8], n=1024)",
        ),
        (
            "aten::cat:#1@m.py:7|args=a%7Cb|torch",
            "aten::cat:#1@m.py:7",
            "torch",
            "a|b",
        ),
        (
            "aten::mm:#1@m.py:7|args=(self=f32[2x2])%3Bextra|torch",
            "aten::mm:#1@m.py:7",
            "torch",
            "(self=f32[2x2]);extra",
        ),
        (
            "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2|triton",
            "torch.compile.fn/triton.CompiledKernel.foo:#1@a.py:1/#1@b.py:2",
            "triton",
            "",
        ),
        ("op|bogus", "op|bogus", "unknown", ""),
        ("", "", "unknown", ""),
        (None, "", "unknown", ""),
    ],
)
def test_parse_function_fields_splits_args(
    raw, expect_function, expect_backend, expect_args
):
    """The args segment is split out and decoded; unrecognized values fall back
    to backend 'unknown'."""
    fn, backend, args = _parse_function_fields(raw)
    assert fn == expect_function
    assert backend == expect_backend
    assert args == expect_args


def test_augment_marker_csv_splits_args_into_dedicated_column(tmp_path):
    """The wire args segment is moved into a dedicated Args column."""
    src = tmp_path / "src_marker_api_trace.csv.gz"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv.gz"
    _write_gzip_csv(
        src,
        pd.DataFrame({
            "Function": [
                "aten::mm:#1@m.py:7|args=(f32[2x2])|torch",
                "aten::relu:#2@m.py:8|torch",
            ],
            "Start": [1, 2],
        }),
    )

    _augment_marker_csv(str(src), str(dst))

    out_df = pd.read_csv(dst, keep_default_na=False)
    assert "Args" in out_df.columns
    assert out_df["Function"].tolist() == [
        "aten::mm:#1@m.py:7",
        "aten::relu:#2@m.py:8",
    ]
    assert out_df["Args"].tolist() == ["(f32[2x2])", ""]
    assert out_df["Backend"].tolist() == ["torch", "torch"]


def test_augment_marker_csv_untagged_row_warns(tmp_path, monkeypatch):
    """Untagged rows are tagged 'unknown' and emit a warning."""
    from utils import utils_profile

    src = tmp_path / "src_marker_api_trace.csv.gz"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv.gz"
    _write_gzip_csv(src, pd.DataFrame({"Function": ["aten::sum"]}))

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
    src = tmp_path / "src_marker_api_trace.csv.gz"
    dst = tmp_path / "ml_api_trace_dst_marker_api_trace.csv.gz"

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
    _write_gzip_csv(src, src_df)

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
    src = tmp_path / "src.csv.gz"
    dst = tmp_path / "dst.csv.gz"
    with gzip.open(src, "wt", newline="", encoding="utf-8") as f:
        f.write("Foo,Bar\n1,2\n3,4\n")

    _augment_marker_csv(str(src), str(dst))

    assert dst.read_bytes() == src.read_bytes()


def test_process_ml_api_trace_output_preserves_per_row_backend(tmp_path):
    """A pre-stripped + tagged CSV (as produced by _augment_marker_csv)
    surfaces the per-row Backend value into the consolidated dataframe.
    """
    workload_dir = str(tmp_path)
    write_rocpd_layout(workload_dir)

    # Overwrite the fixture with what save_ml_api_trace_inputs would produce:
    # Function has prefixes stripped, Backend carries per-row attribution.
    marker_path = Path(workload_dir) / "ml_api_trace_run0_marker_api_trace.csv.gz"
    df = pd.read_csv(marker_path)
    df["Backend"] = ["torch", "torch", "triton"]
    _write_gzip_csv(marker_path, df)

    consolidated_df, _ = process_ml_api_trace_output(workload_dir)

    assert "Backend" in consolidated_df.columns
    backend_by_operator = dict(
        zip(consolidated_df["Operator_Name"], consolidated_df["Backend"])
    )
    assert backend_by_operator.get("torch.mm") == "triton"
    assert backend_by_operator.get("nn.Module.Linear.forward") == "torch"


def test_process_ml_api_trace_output_defaults_args_to_empty(tmp_path):
    """A marker CSV without an Args column yields an empty Args column."""
    workload_dir = str(tmp_path)
    write_rocpd_layout(workload_dir)

    consolidated_df, _ = process_ml_api_trace_output(workload_dir)

    assert "Args" in consolidated_df.columns
    assert (consolidated_df["Args"] == "").all()


def test_process_ml_api_trace_output_preserves_per_row_args(tmp_path):
    """A tagged marker CSV surfaces its per-row Args value into the
    consolidated dataframe.
    """
    workload_dir = str(tmp_path)
    write_rocpd_layout(workload_dir)

    marker_path = Path(workload_dir) / "ml_api_trace_run0_marker_api_trace.csv.gz"
    df = pd.read_csv(marker_path)
    df["Args"] = [
        "(input=float32[2x2])",
        "(input=float32[2x2])",
        "(self=float32[2x2])",
    ]
    _write_gzip_csv(marker_path, df)

    consolidated_df, _ = process_ml_api_trace_output(workload_dir)

    assert "Args" in consolidated_df.columns
    args_by_operator = dict(
        zip(consolidated_df["Operator_Name"], consolidated_df["Args"])
    )
    assert args_by_operator.get("torch.mm") == "(self=float32[2x2])"
    assert args_by_operator.get("nn.Module.Linear.forward") == "(input=float32[2x2])"


# ---- Operator-args parsing and rendering for the analyze display ----


@pytest.mark.parametrize(
    "blob, expected",
    [
        ("", []),
        ("()", []),
        ("(  )", []),
        ("(self=float32[2x2])", ["self=float32[2x2]"]),
        (
            "(self=float32[2x2], mat2=float32[2x3])",
            ["self=float32[2x2]", "mat2=float32[2x3]"],
        ),
        # Commas inside a tensor list must not split the argument.
        (
            "(tensors=[float32[2], float32[2]], dim=0)",
            ["tensors=[float32[2], float32[2]]", "dim=0"],
        ),
        # Commas inside nested parentheses stay within the token.
        (
            "(size=(2, 3), stride=(3, 1))",
            ["size=(2, 3)", "stride=(3, 1)"],
        ),
        # Commas inside braces stay within the token.
        (
            "(options={dtype: float32, device: cuda}, x=1)",
            ["options={dtype: float32, device: cuda}", "x=1"],
        ),
        # Commas inside a double-quoted string stay within the token.
        (
            '(name="a, b", count=2)',
            ['name="a, b"', "count=2"],
        ),
        # Commas inside a single-quoted string stay within the token.
        (
            "(label='x, y', n=1)",
            ["label='x, y'", "n=1"],
        ),
        # An escaped quote does not close the string, so its comma is kept.
        (
            '(s="a\\"b, c", n=1)',
            ['s="a\\"b, c"', "n=1"],
        ),
        # A blob without the wrapping parentheses still splits.
        ("a=1, b=2", ["a=1", "b=2"]),
    ],
)
def test_split_operator_args(blob, expected):
    """Top-level argument tokens are split while bracketed groups stay intact."""
    assert split_operator_args(blob) == expected


def test_format_operator_args_empty_inputs():
    """Empty and contentless blobs render as an empty string."""
    assert format_operator_args("") == ""
    assert format_operator_args("()") == ""


def test_format_operator_args_passthrough_when_short():
    """A short blob is returned in full."""
    assert (
        format_operator_args("(self=float32[2x2], mat2=float32[2x3])")
        == "(self=float32[2x2], mat2=float32[2x3])"
    )


def test_format_operator_args_truncates_item_count():
    """Items beyond max_items collapse into an ellipsis token."""
    blob = "(" + ", ".join(f"a{i}=1" for i in range(10)) + ")"
    rendered = format_operator_args(blob, max_items=3)
    assert rendered == "(a0=1, a1=1, a2=1, ...)"


def test_format_operator_args_truncates_length():
    """An over-long rendering is capped with a trailing ellipsis."""
    blob = "(name=" + "x" * 200 + ")"
    rendered = format_operator_args(blob, max_chars=40)
    assert len(rendered) <= 40
    assert rendered.endswith("...)")


def test_build_call_trees_attaches_leaf_args():
    """Args from the wire attach to the leaf operator node of the call tree."""
    df = pd.DataFrame({
        "Operator_Name": ["nn.Module.Linear.forward/aten::mm"],
        "Context_Id": ["1@m.py:10/#1@m.py:12"],
        "Kernel_Name": ["kernel_gemm"],
        "Args": ["(self=float32[2x2], mat2=float32[2x3])"],
    })

    trees = build_call_trees(df)

    root = trees["m.py:10"]
    parent = root.children["nn.Module.Linear.forward"]
    leaf = parent.children["aten::mm"]
    assert leaf.args_variants == [("(self=float32[2x2], mat2=float32[2x3])", 1)]
    # Args belong to the leaf only; ancestor frames stay empty.
    assert parent.args_variants == []


def test_build_call_trees_without_args_column():
    """A trace lacking the Args column records no argument variants."""
    df = pd.DataFrame({
        "Operator_Name": ["aten::mm"],
        "Context_Id": ["1@m.py:10"],
        "Kernel_Name": ["kernel_gemm"],
    })

    trees = build_call_trees(df)

    leaf = trees["m.py:10"].children["aten::mm"]
    assert leaf.args_variants == []


def test_build_call_trees_counts_every_args_variant():
    """Distinct argument sets are kept with their call counts, most frequent
    first, and the counts total the node's call count."""
    df = pd.DataFrame({
        "Operator_Name": ["aten::mm"] * 3,
        "Context_Id": ["1@m.py:10", "2@m.py:10", "3@m.py:10"],
        "Kernel_Name": ["kernel_gemm"] * 3,
        "Args": [
            "(self=float32[4x4])",
            "(self=float32[2x2])",
            "(self=float32[4x4])",
        ],
    })

    trees = build_call_trees(df)

    leaf = trees["m.py:10"].children["aten::mm"]
    assert leaf.args_variants == [
        ("(self=float32[4x4])", 2),
        ("(self=float32[2x2])", 1),
    ]
    assert sum(count for _blob, count in leaf.args_variants) == leaf.call_count


def test_build_call_trees_ignores_contentless_args_placeholder():
    """A contentless "()" blob is not recorded as an argument variant."""
    df = pd.DataFrame({
        "Operator_Name": ["aten::mm"] * 2,
        "Context_Id": ["1@m.py:10", "2@m.py:10"],
        "Kernel_Name": ["kernel_gemm"] * 2,
        "Args": ["()", "(self=float32[2x2])"],
    })

    trees = build_call_trees(df)

    leaf = trees["m.py:10"].children["aten::mm"]
    assert leaf.args_variants == [("(self=float32[2x2])", 1)]


def test_build_call_trees_args_variants_without_context_ids():
    """A trace without Context_Id records its blobs with a call count of 0."""
    df = pd.DataFrame({
        "Operator_Name": ["aten::mm"] * 2,
        "Kernel_Name": ["kernel_gemm", "kernel_other"],
        "Args": ["(self=float32[4x4])", "(self=float32[2x2])"],
    })

    trees = build_call_trees(df)

    leaf = trees["unknown:0"].children["aten::mm"]
    assert leaf.args_variants == [
        ("(self=float32[2x2])", 0),
        ("(self=float32[4x4])", 0),
    ]


def test_build_call_trees_args_variants_deeper_than_context_path():
    """A frame deeper than the Context_Id path records its blobs with a call
    count of 0."""
    df = pd.DataFrame({
        "Operator_Name": ["aten::matmul/aten::mm"] * 2,
        "Context_Id": ["1@m.py:10"] * 2,
        "Kernel_Name": ["kernel_gemm", "kernel_other"],
        "Args": ["(self=float32[4x4])", "(self=float32[2x2])"],
    })

    trees = build_call_trees(df)

    outer = trees["m.py:10"].children["aten::matmul"]
    leaf = outer.children["aten::mm"]
    assert outer.call_count == 1
    assert leaf.call_count == 0
    assert leaf.args_variants == [
        ("(self=float32[2x2])", 0),
        ("(self=float32[4x4])", 0),
    ]
