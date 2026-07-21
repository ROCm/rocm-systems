# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils.file_io.create_df_kernel_top_stats."""

import tempfile

import pandas as pd
import pytest

from utils.file_io import create_df_kernel_top_stats


def _raw_pmc() -> pd.DataFrame:
    """Flat raw_pmc DataFrame for create_df_kernel_top_stats tests."""
    return pd.DataFrame({
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a", "kernel_c"],
        "GPU_ID": [0, 0, 1, 0],
        "Dispatch_ID": [1, 2, 3, 4],
        "Start_Timestamp": [1000, 2000, 3000, 4000],
        "End_Timestamp": [1500, 2800, 3400, 4200],
    })


def test_returns_valid_dataframes() -> None:
    """create_df_kernel_top_stats returns valid DFs with correct structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, dispatch_info_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="sum",
        )

        assert isinstance(kernel_top_df, pd.DataFrame)
        assert isinstance(dispatch_info_df, pd.DataFrame)

        expected_columns = [
            "Kernel_Name",
            "Count",
            "Sum(ns)",
            "Mean(ns)",
            "Median(ns)",
            "Percent",
        ]
        for col in expected_columns:
            assert col in kernel_top_df.columns, f"Missing column: {col}"

        assert "Kernel_Name" in dispatch_info_df.columns
        assert "GPU_ID" in dispatch_info_df.columns
        assert "Dispatch_ID" in dispatch_info_df.columns

        assert kernel_top_df.index[0] == 0
        assert kernel_top_df["Percent"].sum() == pytest.approx(100.0, abs=0.01)


def test_grouping_and_aggregation() -> None:
    """Kernel grouping, aggregation functions, and sorting behavior."""
    with tempfile.TemporaryDirectory() as temp_dir:
        kernel_top_df, _ = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="sum",
        )

        # kernel_a appears twice in input and must group into one row.
        kernel_a_row = kernel_top_df[kernel_top_df["Kernel_Name"] == "kernel_a"]
        assert len(kernel_a_row) == 1
        assert kernel_a_row["Count"].iloc[0] == 2

        # Sorting by sum is descending.
        sum_values = kernel_top_df["Sum(ns)"].tolist()
        assert sum_values == sorted(sum_values, reverse=True)

        kernel_top_df_sorted, _ = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
            kernel_verbose=0,
            sortby="kernel",
        )

        # Sorting by kernel name is ascending.
        kernel_names = kernel_top_df_sorted["Kernel_Name"].tolist()
        assert kernel_names == sorted(kernel_names)


def test_filters() -> None:
    """GPU ID, dispatch ID (including '> n' syntax), and empty input handling."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # GPU ID filter: GPU_ID=0 excludes kernel_a at GPU 1 (3 dispatches).
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids="0",
            filter_dispatch_ids=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(dispatch_df) == 3

        # Dispatch ID filter with "> n" syntax keeps IDs 3 and 4.
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["> 2"],
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(dispatch_df) == 2
        assert all(dispatch_df["Dispatch_ID"] > 2)

        # Dispatch ID filter with specific IDs.
        _, dispatch_df = create_df_kernel_top_stats(
            df_in=_raw_pmc(),
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=["1", "2"],
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(dispatch_df) == 2

        # Empty input yields empty outputs.
        empty_raw_pmc = pd.DataFrame({
            "Kernel_Name": [],
            "GPU_ID": [],
            "Dispatch_ID": [],
            "Start_Timestamp": [],
            "End_Timestamp": [],
        })
        kernel_top_df, dispatch_df = create_df_kernel_top_stats(
            df_in=empty_raw_pmc,
            raw_data_dir=temp_dir,
            filter_gpu_ids=None,
            filter_dispatch_ids=None,
            time_unit="ns",
            kernel_verbose=0,
        )
        assert len(kernel_top_df) == 0
        assert len(dispatch_df) == 0
