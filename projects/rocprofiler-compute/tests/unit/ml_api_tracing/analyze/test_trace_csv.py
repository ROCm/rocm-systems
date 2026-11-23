# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for reading ML API trace CSV files."""

from pathlib import Path

import pandas as pd
import pytest

from ml_api_tracing.analyze.errors import MlApiTraceInputError
from ml_api_tracing.analyze.trace_csv import (
    load_counter_trace_dataframe,
    load_ml_api_trace_pairs,
)

MARKER_COLUMNS = [
    "Function",
    "Thread_Id",
    "Correlation_ID",
    "GUID",
    "Start_Timestamp",
    "End_Timestamp",
]
COUNTER_COLUMNS = [
    "Correlation_ID",
    "GUID",
    "Dispatch_ID",
    "Kernel_Name",
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Start_Timestamp",
    "End_Timestamp",
]


def pass_csv_path(workload_dir, pass_id, kind):
    suffix = "marker_api_trace" if kind == "marker" else "counter_collection"
    return workload_dir / f"ml_api_trace_pmc_perf_{pass_id}_{suffix}.csv.gz"


def read_pass_csv(workload_dir, kind):
    if kind == "marker":
        return load_ml_api_trace_pairs(workload_dir)[0].marker_df
    return load_counter_trace_dataframe(pass_csv_path(workload_dir, 0, kind))


def test_load_pairs_uses_lexical_paths_and_sibling_counter_files(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row
):
    for subdirectory, pass_id, correlation in [("b", 1, 1), ("a", 2, 2), ("a", 10, 10)]:
        write_ml_api_pass(
            tmp_path / subdirectory,
            pass_id,
            [linear_marker_row(correlation, 0, 100)],
            [counter_row(correlation, "kernel", 10, 20)],
        )

    pairs = load_ml_api_trace_pairs(tmp_path)

    assert [pair.marker_df["Correlation_ID"].iloc[0] for pair in pairs] == [10, 2, 1]
    assert [pair.counter_path.relative_to(tmp_path) for pair in pairs] == [
        Path("a/ml_api_trace_pmc_perf_10_counter_collection.csv.gz"),
        Path("a/ml_api_trace_pmc_perf_2_counter_collection.csv.gz"),
        Path("b/ml_api_trace_pmc_perf_1_counter_collection.csv.gz"),
    ]


def test_load_pairs_skips_missing_or_plain_counter_and_plain_marker(tmp_path):
    for marker_name, counter_name in [
        ("ml_api_trace_a_marker_api_trace.csv.gz", None),
        (
            "ml_api_trace_b_marker_api_trace.csv.gz",
            "ml_api_trace_b_counter_collection.csv",
        ),
        (
            "ml_api_trace_c_marker_api_trace.csv",
            "ml_api_trace_c_counter_collection.csv.gz",
        ),
    ]:
        (tmp_path / marker_name).touch()
        if counter_name is not None:
            (tmp_path / counter_name).touch()

    assert load_ml_api_trace_pairs(tmp_path) == []


@pytest.mark.parametrize("kind", ["marker", "counter"])
@pytest.mark.parametrize("alias_only", [False, True])
def test_load_csv_normalizes_aliases_and_preserves_projection_dtypes(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row, kind, alias_only
):
    marker = linear_marker_row(7, 10, 100, thread_id=42)
    counter = counter_row(7, "kernel", 20, 30, dispatch_id=9)
    target = marker if kind == "marker" else counter
    aliases = [("Correlation_ID", "Correlation_Id")]
    if kind == "counter":
        aliases.append(("Dispatch_ID", "Dispatch_Id"))
    for canonical, alias in aliases:
        target[alias] = target.pop(canonical) if alias_only else 999
    target["Ignored"] = None
    write_ml_api_pass(tmp_path, 0, [marker], [counter])
    expected = pd.DataFrame([
        linear_marker_row(7, 10, 100, thread_id=42)
        if kind == "marker"
        else counter_row(7, "kernel", 20, 30, dispatch_id=9)
    ])[MARKER_COLUMNS if kind == "marker" else COUNTER_COLUMNS]

    loaded = read_pass_csv(tmp_path, kind)

    pd.testing.assert_frame_equal(loaded, expected)


@pytest.mark.parametrize(
    ("kind", "failure", "detail"),
    [
        ("marker", "missing", "is missing required columns ['GUID']"),
        ("counter", "null", "has null values in ['Kernel_Name']"),
    ],
)
def test_load_csv_reports_required_column_errors(
    tmp_path, write_ml_api_pass, linear_marker_row, counter_row, kind, failure, detail
):
    marker = linear_marker_row(1, 0, 100)
    counter = counter_row(1, "kernel", 10, 20)
    if failure == "missing":
        del marker["GUID"]
    else:
        counter["Kernel_Name"] = None
    write_ml_api_pass(tmp_path, 0, [marker], [counter])

    with pytest.raises(MlApiTraceInputError) as excinfo:
        read_pass_csv(tmp_path, kind)

    assert str(excinfo.value) == (
        f"{kind.capitalize()} CSV {pass_csv_path(tmp_path, 0, kind)} {detail}"
    )


def test_load_pairs_keeps_header_only_discovered_pair(tmp_path):
    marker_path = pass_csv_path(tmp_path, 0, "marker")
    counter_path = pass_csv_path(tmp_path, 0, "counter")
    pd.DataFrame(columns=MARKER_COLUMNS).to_csv(
        marker_path, index=False, compression="gzip"
    )
    pd.DataFrame(columns=COUNTER_COLUMNS).to_csv(
        counter_path, index=False, compression="gzip"
    )

    pairs = load_ml_api_trace_pairs(tmp_path)
    counter_df = load_counter_trace_dataframe(counter_path)

    assert len(pairs) == 1
    pd.testing.assert_frame_equal(
        pairs[0].marker_df, pd.DataFrame(columns=MARKER_COLUMNS)
    )
    pd.testing.assert_frame_equal(counter_df, pd.DataFrame(columns=COUNTER_COLUMNS))
