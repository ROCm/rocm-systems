# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit coverage for the per-kernel roofline CSV export: flattening the FLOP
figure's kernel view-model into tidy rows and writing them to disk.
"""

from pathlib import Path

import pandas as pd

from roofline.roofline_csv import (
    KERNEL_ROOFLINE_CSV_COLUMNS,
    KERNEL_ROOFLINE_CSV_FILENAME,
    build_kernel_roofline_dataframe,
    collect_kernel_roofline_rows,
    write_kernel_roofline_csv,
)

KERNELS = [
    {
        "name": "kernelA",
        "pctRuntime": 60.0,
        "count": 4.0,
        "totalTime": 1000.0,
        "points": [
            {
                "peak": "HBM",
                "ai": 2.0,
                "perf": 500.0,
                "roofPerf": 1000.0,
                "pctRoof": 50.0,
                "bandwidth": 500.0,
            },
            {
                "peak": "L2",
                "ai": 4.0,
                "perf": 500.0,
                "roofPerf": None,
                "pctRoof": None,
                "bandwidth": None,
            },
        ],
    },
    {
        "name": "kernelB",
        "pctRuntime": 40.0,
        "count": None,
        "totalTime": None,
        "points": [],
    },
]


def test_collect_kernel_roofline_rows_flattens_one_row_per_point() -> None:
    """One row per kernel per plotted memory level; a kernel with no points
    contributes no rows at all."""
    rows = collect_kernel_roofline_rows(KERNELS, time_unit="ns")

    assert [row["Kernel_Name"] for row in rows] == ["kernelA", "kernelA"]
    assert [row["Cache_Level"] for row in rows] == ["HBM", "L2"]
    assert rows[0]["Arithmetic_Intensity"] == 2.0
    assert rows[0]["Performance_GFLOPs"] == 500.0
    assert rows[0]["Roofline_Peak_GFLOPs"] == 1000.0
    assert rows[0]["Pct_of_Roofline"] == 50.0
    assert rows[0]["Bandwidth_GBps"] == 500.0
    assert rows[0]["Count"] == 4.0
    assert rows[0]["Total_Time"] == 1000.0
    assert rows[0]["Time_Unit"] == "ns"
    assert rows[0]["Pct_Runtime"] == 60.0

    # An unroofed point still gets a row, with the missing values as None.
    assert rows[1]["Roofline_Peak_GFLOPs"] is None
    assert rows[1]["Bandwidth_GBps"] is None


def test_collect_kernel_roofline_rows_empty_kernels_produce_no_rows() -> None:
    assert collect_kernel_roofline_rows([], time_unit="ns") == []


def test_build_kernel_roofline_dataframe_uses_fixed_column_order() -> None:
    rows = collect_kernel_roofline_rows(KERNELS, time_unit="ns")

    df = build_kernel_roofline_dataframe(rows)

    assert list(df.columns) == KERNEL_ROOFLINE_CSV_COLUMNS
    assert len(df) == 2


def test_build_kernel_roofline_dataframe_empty_rows_keeps_columns() -> None:
    """An empty rows list still yields a DataFrame with the expected columns,
    so callers do not need to special-case an empty result."""
    df = build_kernel_roofline_dataframe([])

    assert list(df.columns) == KERNEL_ROOFLINE_CSV_COLUMNS
    assert len(df) == 0


def test_write_kernel_roofline_csv_writes_expected_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [{"Kernel_Name": "kernelA", "Cache_Level": "HBM"}],
        columns=["Kernel_Name", "Cache_Level"],
    )

    output_path = write_kernel_roofline_csv(df, tmp_path)

    assert output_path == tmp_path / KERNEL_ROOFLINE_CSV_FILENAME
    assert output_path.is_file()
    written = pd.read_csv(output_path)
    assert written.loc[0, "Kernel_Name"] == "kernelA"
    assert written.loc[0, "Cache_Level"] == "HBM"
