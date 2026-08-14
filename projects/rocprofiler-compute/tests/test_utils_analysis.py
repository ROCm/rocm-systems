# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/utils_analysis.py."""

import gzip
import math
import os
from pathlib import Path

import pandas as pd
import pytest

import utils.utils_analysis as utils_analysis
from utils.utils_analysis import (
    CallTreeNode,
    KernelStats,
    NodeRollup,
    build_call_trees,
    build_operator_summary,
    parse_top_level_location,
    rollup_node_stats,
)

# =============================================================================
# TESTS FOR EMPTY WORKLOAD
#
# Normal Functionality:
#
# Valid CSV files with data
# Mixed valid and invalid data
# Large datasets
# Unicode content handling
# Edge Cases:
#
# Empty CSV files
# CSV with only headers
# Files with all NaN values that become empty after dropna()
# Malformed CSV files
# Missing pmc_perf.csv file
# Nonexistent directories
# Error Conditions:
#
# File permission errors
# CSV reading errors
# Directory access issues
# String Formatting and Dependencies:
#
# Console error message formatting
# Path handling (string vs Path)
# Pandas dependency verification
# Return value consistency
# Special Scenarios:
#
# Special characters in paths
# Unicode content in CSV files
# Large datasets with performance implications
# Different input path types
# =============================================================================


def test_validate_workload_valid_data_file(tmp_path):
    """
    Test validate_workload with a valid pmc_perf.csv file containing data.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles valid data files without errors.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    valid_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200
kernel2,1,150,250
kernel3,0,120,220"""
    pmc_perf_file.write_text(valid_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 0


def test_validate_workload_file_with_nan_values(tmp_path):
    """
    Test validate_workload with pmc_perf.csv containing NaN values.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects and reports empty cells after dropping NaN.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    nan_data = """Kernel_Name,GPU_ID,Counter1,Counter2
,,NaN,
,NaN,,NaN
NaN,,,"""
    pmc_perf_file.write_text(nan_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "pmc_perf.csv" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_validate_workload_completely_empty_gzip_csv(tmp_path):
    """
    Test validate_workload with a valid gzip file containing no CSV data.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects empty CSV file.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    result_file = workload_dir / "results_pmc_perf_0.csv.gz"
    result_file.write_bytes(gzip.compress(b""))

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "profiling"
    assert "No counter data" in error_args[1]
    assert str(result_file) in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_validate_workload_headers_only_csv(tmp_path):
    """
    Test validate_workload with CSV containing only headers.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects CSV with headers but no data.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    headers_only = "Kernel_Name,GPU_ID,Counter1,Counter2"
    pmc_perf_file.write_text(headers_only)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_validate_workload_no_pmc_perf_file(tmp_path):
    """
    Test validate_workload when pmc_perf.csv file doesn't exist.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects missing profiling data file.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_validate_workload_nonexistent_directory():
    """
    Test validate_workload with nonexistent directory path.

    Returns:
        None: Asserts function handles nonexistent directories.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload("/nonexistent/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_validate_workload_malformed_csv(tmp_path):
    """
    Test validate_workload with malformed CSV that causes pandas read error.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles pandas CSV reading errors gracefully.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    malformed_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200,extra_column_data
kernel2,1,150
incomplete_row"""
    pmc_perf_file.write_text(malformed_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        try:
            utils_analysis.validate_workload(str(workload_dir))
        except Exception:
            pass


def test_validate_workload_mixed_valid_invalid_data(tmp_path):
    """
    Test validate_workload with CSV containing mix of valid and invalid (NaN) data.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles mixed data correctly.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    mixed_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200
kernel2,,NaN,250
kernel3,1,120,
,0,110,240"""
    pmc_perf_file.write_text(mixed_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 0


def test_validate_workload_large_dataset_with_nans(tmp_path):
    """
    Test validate_workload with large dataset that becomes empty after dropping NaNs.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function correctly processes large datasets.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    headers = "Kernel_Name,GPU_ID,Counter1,Counter2\n"
    nan_rows = []
    for i in range(1000):
        nan_rows.append("NaN,NaN,NaN,NaN")
    large_nan_data = headers + "\n".join(nan_rows)
    pmc_perf_file.write_text(large_nan_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_validate_workload_unicode_content(tmp_path):
    """
    Test validate_workload with CSV containing Unicode characters.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles Unicode content correctly.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    unicode_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel_测试,0,100,200
kernel_тест,1,150,250
kernel_tëst,0,120,220"""
    pmc_perf_file.write_text(unicode_data, encoding="utf-8")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 0


def test_validate_workload_special_path_characters(tmp_path):
    """
    Test validate_workload with directory paths containing special characters.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles special characters in paths.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload-test_dir.with.dots"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    valid_data = """Kernel_Name,GPU_ID,Counter1,Counter2
kernel1,0,100,200"""
    pmc_perf_file.write_text(valid_data)

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 0


def test_validate_workload_csv_read_permission_error(tmp_path):
    """
    Test validate_workload when CSV file exists but cannot be read due to permissions.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles file permission errors.
    """
    from unittest.mock import patch

    if os.name == "nt":
        pytest.skip("Permission test not applicable on Windows")

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nkernel1,0")
    pmc_perf_file.chmod(0o000)  # Remove all permissions

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    try:
        with patch(
            "utils.utils_analysis.console_error", side_effect=mock_console_error
        ):
            utils_analysis.validate_workload(str(workload_dir))
    except PermissionError:
        pass
    finally:
        pmc_perf_file.chmod(0o644)


def test_validate_workload_string_path_input():
    """
    Test validate_workload with string path input vs Path.

    Returns:
        None: Asserts function handles different path input types.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload("/nonexistent/string/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_validate_workload_console_error_string_formatting(tmp_path):
    """
    Test validate_workload string formatting in console_error messages.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts console_error messages are properly formatted.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nNaN,NaN")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.validate_workload(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    expected_path = str(workload_dir / "pmc_perf.csv")
    assert expected_path in error_args[1]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_validate_workload_function_return_value(tmp_path):
    """
    Test that validate_workload function return behavior (implicitly returns None).

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function return value consistency.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("Kernel_Name,GPU_ID\nkernel1,0")

    with patch("utils.utils_analysis.console_error"):
        result = utils_analysis.validate_workload(str(workload_dir))

    assert result is None

    workload_dir2 = tmp_path / "workload2"
    workload_dir2.mkdir()

    with patch("utils.utils_analysis.console_error"):
        result2 = utils_analysis.validate_workload(str(workload_dir2))

    assert result2 is None


def test_validate_workload_pandas_import_dependency():
    """
    Test validate_workload dependency on pandas module.

    Returns:
        None: Asserts function properly uses pandas functionality.
    """
    from unittest.mock import MagicMock, patch

    mock_pandas = MagicMock()
    mock_df = MagicMock()
    mock_df.dropna.return_value.empty = False
    mock_pandas.read_csv.return_value = mock_df

    with patch.dict("sys.modules", {"pandas": mock_pandas}):
        with patch("utils.utils_analysis.pd", mock_pandas):
            with patch("utils.utils_analysis.console_error"):
                with patch("pathlib.Path.is_file", return_value=True):
                    utils_analysis.validate_workload("/test/path")

    mock_pandas.read_csv.assert_called_once()
    mock_df.dropna.assert_called_once()


# =============================================================================
# TESTS FOR ITERATION MULTIPLEXING
# =============================================================================


def test_impute_counters_iteration_multiplex(tmp_path: Path) -> None:
    """Test impute_counters_iteration_multiplex with sample DataFrame."""
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    # For "kernel" policy
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # Assert Counter1 and Counter2 imputed for first two dispatches
    assert result["Counter2"].iloc[0] == 500
    assert result["Counter1"].iloc[1] == 100

    # For "kernel_launch_params" policy
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # No imputation possible
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[2])

    # Test multi_kernel
    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    # For "kernel" policy
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # For "kernel_launch_params" policy
    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows
    # No imputation possible
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[2])

    # Test incomplete last subgroup handling and no cross-subgroup contamination
    # Scenario: 3 counter buckets, 8 dispatches (2 complete subgroups + incomplete last)
    # Subgroup 0: rows 0-2, Subgroup 1: rows 3-5, Subgroup 2 (incomplete): rows 6-7
    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": ["kernel_a"] * 8,
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
        "End_Timestamp": [1100, 1300, 1500, 1700, 1900, 2100, 2300, 2500],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1, 1],
        # Counter bucket pattern: A, B, C (repeats)
        "Counter_A": [100, None, None, 200, None, None, 300, None],
        "Counter_B": [None, 110, None, None, 210, None, None, 310],
        "Counter_C": [None, None, 120, None, None, 220, None, None],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    # Verify complete subgroups: all rows should have all counters
    assert result["Counter_A"].iloc[0] == 100
    assert result["Counter_A"].iloc[1] == 100
    assert result["Counter_A"].iloc[2] == 100
    assert result["Counter_B"].iloc[0] == 110
    assert result["Counter_C"].iloc[0] == 120

    # Verify no cross-subgroup contamination: subgroup 1 has its own values
    assert result["Counter_A"].iloc[3] == 200
    assert result["Counter_A"].iloc[4] == 200
    assert result["Counter_B"].iloc[3] == 210
    assert result["Counter_C"].iloc[3] == 220

    # Verify incomplete last subgroup gets filled from previous subgroup
    # Row 6-7 only have Counter_A and Counter_B, missing Counter_C
    assert result["Counter_A"].iloc[6] == 300
    assert result["Counter_A"].iloc[7] == 300
    assert result["Counter_B"].iloc[6] == 310
    assert result["Counter_B"].iloc[7] == 310
    # Counter_C should be filled from previous subgroup via global ffill
    assert result["Counter_C"].iloc[6] == 220
    assert result["Counter_C"].iloc[7] == 220

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8  # Ensure same number of rows


# ---------------------------------------------------------------------------
# Tests for call-tree functions (build/display)
# ---------------------------------------------------------------------------


def test_parse_location_normal():
    assert parse_top_level_location("10@main.py:60/#10@main.py:21") == "main.py:60"


def test_parse_location_single_entry():
    assert parse_top_level_location("5@train.py:42") == "train.py:42"


def test_parse_location_nan():
    assert parse_top_level_location(float("nan")) == "unknown:0"


def test_parse_location_none():
    assert parse_top_level_location(None) == "unknown:0"


def test_parse_location_empty():
    assert parse_top_level_location("") == "unknown:0"
    assert parse_top_level_location("   ") == "unknown:0"


def test_parse_location_no_at_sign():
    assert parse_top_level_location("no_at_sign") == "unknown:0"


def test_parse_location_no_colon():
    assert parse_top_level_location("10@mainpy") == "unknown:0"


def test_kernel_stats_defaults_min_max_to_none():
    stats = KernelStats()
    assert stats.min_duration_ns is None
    assert stats.max_duration_ns is None


def test_call_tree_node_defaults_dispatch_stats_to_none():
    node = CallTreeNode(name="x")
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns is None


def test_call_tree_node_call_count_is_property_of_invocation_ids():
    node = CallTreeNode(name="x")
    assert node.call_count == 0
    node.invocation_ids.add("ctx1")
    node.invocation_ids.add("ctx2")
    assert node.call_count == 2


def test_rollup_leaf_node():
    node = CallTreeNode(name="leaf")
    node.kernels["kern_a"] = KernelStats(launches=2, total_duration_ns=1000.0)
    rollup = rollup_node_stats(node)
    assert rollup.launches == 2
    assert rollup.total_duration_ns == 1000.0
    assert node.kernel_launches == 2


def test_rollup_leaf_node_with_no_min_max_returns_none():
    node = CallTreeNode(name="leaf")
    node.kernels["kern"] = KernelStats(launches=1, total_duration_ns=0.0)
    rollup = rollup_node_stats(node)
    assert isinstance(rollup, NodeRollup)
    assert rollup.min_dispatch_ns is None
    assert rollup.max_dispatch_ns is None
    assert node.min_dispatch_ns is None
    assert node.max_dispatch_ns is None
    assert node.mean_dispatch_ns == 0.0


def test_rollup_leaf_node_with_zero_launches_has_mean_none():
    node = CallTreeNode(name="leaf")
    rollup = rollup_node_stats(node)
    assert rollup.launches == 0
    assert node.mean_dispatch_ns is None


def test_rollup_propagates_min_max_from_kernel_stats():
    node = CallTreeNode(name="leaf")
    node.kernels["k"] = KernelStats(
        launches=2,
        total_duration_ns=3000.0,
        min_duration_ns=1000.0,
        max_duration_ns=2000.0,
    )
    rollup_node_stats(node)
    assert node.min_dispatch_ns == 1000.0
    assert node.max_dispatch_ns == 2000.0
    assert node.mean_dispatch_ns == 1500.0


def test_rollup_parent_rolls_up_children():
    child = CallTreeNode(name="child")
    child.kernels["kern_a"] = KernelStats(launches=3, total_duration_ns=3000.0)
    parent = CallTreeNode(name="parent")
    parent.children["child"] = child
    parent.kernels["kern_b"] = KernelStats(launches=1, total_duration_ns=500.0)
    rollup_node_stats(parent)
    assert parent.kernel_launches == 4
    assert child.kernel_launches == 3


def test_rollup_deep_hierarchy():
    grandchild = CallTreeNode(name="grandchild")
    grandchild.kernels["k"] = KernelStats(launches=1, total_duration_ns=100.0)
    child = CallTreeNode(name="child")
    child.children["grandchild"] = grandchild
    child.kernels["k2"] = KernelStats(launches=2, total_duration_ns=200.0)
    root = CallTreeNode(name="root")
    root.children["child"] = child
    rollup_node_stats(root)
    assert grandchild.kernel_launches == 1
    assert child.kernel_launches == 3
    assert root.kernel_launches == 3


def test_build_call_trees_empty_df():
    assert build_call_trees(pd.DataFrame()) == {}


def test_build_call_trees_missing_columns():
    assert build_call_trees(pd.DataFrame([{"Operator_Name": "a"}])) == {}


def test_build_call_trees_single_dispatch():
    df = pd.DataFrame([
        {
            "Operator_Name": "torch.nn.Linear",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@train.py:42",
            "Start_Timestamp_kernel": 1000,
            "End_Timestamp_kernel": 2000,
        }
    ])
    call_trees = build_call_trees(df)
    assert "train.py:42" in call_trees
    assert call_trees["train.py:42"].kernel_launches == 1
    assert "torch.nn.Linear" in call_trees["train.py:42"].children


def test_build_call_trees_hierarchy_split():
    df = pd.DataFrame([
        {
            "Operator_Name": "aten/linear/addmm",
            "Kernel_Name": "gemm_kernel",
            "Context_Id": "10@file.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    call_trees = build_call_trees(df)
    root = call_trees["file.py:1"]
    assert "aten" in root.children
    assert "linear" in root.children["aten"].children
    assert "addmm" in root.children["aten"].children["linear"].children


def test_build_call_trees_multiple_dispatches_same_kernel():
    rows = [
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": i * 1000,
            "End_Timestamp_kernel": (i + 1) * 1000,
        }
        for i in range(3)
    ]
    call_trees = build_call_trees(pd.DataFrame(rows))
    assert call_trees["f.py:1"].kernel_launches == 3
    assert call_trees["f.py:1"].children["op_a"].kernels["kern"].launches == 3


def test_build_call_trees_dedup_identical_timestamps():
    row = {
        "Operator_Name": "op",
        "Kernel_Name": "kern",
        "Context_Id": "10@f.py:1",
        "Start_Timestamp_kernel": 1000,
        "End_Timestamp_kernel": 2000,
    }
    assert build_call_trees(pd.DataFrame([row, row]))["f.py:1"].kernel_launches == 1


def test_build_call_trees_no_context_id():
    df = pd.DataFrame([
        {
            "Operator_Name": "op",
            "Kernel_Name": "kern",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        }
    ])
    assert "unknown:0" in build_call_trees(df)


def test_build_call_trees_duration_rollup():
    df = pd.DataFrame([
        {
            "Operator_Name": "parent/child",
            "Kernel_Name": "kern_a",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "parent",
            "Kernel_Name": "kern_b",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 2_000_000,
            "End_Timestamp_kernel": 3_000_000,
        },
    ])
    call_trees = build_call_trees(df)
    root = call_trees["f.py:1"]
    assert root.kernel_launches == 2
    assert root.children["parent"].kernel_launches == 2
    assert root.children["parent"].children["child"].kernel_launches == 1


def test_build_call_trees_multiple_source_locations():
    df = pd.DataFrame([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@a.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "10@b.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1000,
        },
    ])
    call_trees = build_call_trees(df)
    assert "a.py:1" in call_trees
    assert "b.py:2" in call_trees


# ---------------------------------------------------------------------------
# build_operator_summary
# ---------------------------------------------------------------------------


_OPERATOR_SUMMARY_COLUMNS = [
    "Operator",
    "Location",
    "Calls",
    "Dispatches",
    "Dispatches_Per_Call",
    "Total_GPU",
    "Pct_Total_GPU",
    "Mean_Per_Call",
    "Mean_Per_Dispatch",
    "Min_Dispatch",
    "Max_Dispatch",
]


def _build_summary_from_dataframe(rows):
    call_trees = build_call_trees(pd.DataFrame(rows))
    return build_operator_summary(call_trees)


def test_build_operator_summary_empty_input_returns_empty_with_full_schema():
    summary = build_operator_summary({})
    assert list(summary.columns) == _OPERATOR_SUMMARY_COLUMNS
    assert summary.empty


def test_build_operator_summary_skips_synthetic_location_root():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        }
    ])
    assert "f.py:1" not in summary["Operator"].tolist()
    assert "op_a" in summary["Operator"].tolist()


def test_build_operator_summary_row_values_for_single_dispatch():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 2_000_000,
        }
    ])
    row = summary.loc[summary["Operator"] == "op_a"].iloc[0]
    assert row["Location"] == "f.py:1"
    assert row["Calls"] == 1
    assert row["Dispatches"] == 1
    assert row["Dispatches_Per_Call"] == 1.0
    assert row["Total_GPU"] == pytest.approx(2.0)
    assert row["Pct_Total_GPU"] == pytest.approx(100.0)
    assert row["Mean_Per_Call"] == pytest.approx(2.0)
    assert row["Mean_Per_Dispatch"] == pytest.approx(2.0)
    assert row["Min_Dispatch"] == pytest.approx(2.0)
    assert row["Max_Dispatch"] == pytest.approx(2.0)


def test_build_operator_summary_sort_by_total_descending():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "small_op",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
        {
            "Operator_Name": "big_op",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 10_000_000,
        },
    ])
    operators_in_order = summary["Operator"].tolist()
    assert operators_in_order.index("big_op") < operators_in_order.index("small_op")


def test_build_operator_summary_pct_total_gpu_sums_to_100_at_top_level():
    summary = _build_summary_from_dataframe([
        {
            "Operator_Name": "op_a",
            "Kernel_Name": "kern",
            "Context_Id": "10@f.py:1",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 3_000_000,
        },
        {
            "Operator_Name": "op_b",
            "Kernel_Name": "kern",
            "Context_Id": "20@f.py:2",
            "Start_Timestamp_kernel": 0,
            "End_Timestamp_kernel": 1_000_000,
        },
    ])
    op_a_pct = summary.loc[summary["Operator"] == "op_a", "Pct_Total_GPU"].iloc[0]
    op_b_pct = summary.loc[summary["Operator"] == "op_b", "Pct_Total_GPU"].iloc[0]
    assert op_a_pct == pytest.approx(75.0)
    assert op_b_pct == pytest.approx(25.0)


def test_build_operator_summary_pct_total_gpu_is_nan_when_grand_total_zero():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 0.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    pct = summary.loc[summary["Operator"] == "op", "Pct_Total_GPU"].iloc[0]
    assert math.isnan(pct)


def test_build_operator_summary_min_max_mean_are_nan_when_no_dispatch_stats():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="op")
    op.kernel_launches = 1
    op.total_duration_ms = 5.0
    op.invocation_ids.add("ctx")
    root.children["op"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "op"].iloc[0]
    assert math.isnan(row["Min_Dispatch"])
    assert math.isnan(row["Max_Dispatch"])
    assert math.isnan(row["Mean_Per_Dispatch"])


def test_build_operator_summary_calls_nan_when_no_invocation_ids():
    root = CallTreeNode(name="f.py:1")
    op = CallTreeNode(name="torch.ops.x")
    op.kernel_launches = 2
    op.total_duration_ms = 4.0
    op.mean_dispatch_ns = 2_000_000.0
    op.min_dispatch_ns = 2_000_000.0
    op.max_dispatch_ns = 2_000_000.0
    root.children["torch.ops.x"] = op
    summary = build_operator_summary({"f.py:1": root})
    row = summary.loc[summary["Operator"] == "torch.ops.x"].iloc[0]
    assert math.isnan(row["Calls"])
    assert math.isnan(row["Dispatches_Per_Call"])
    assert math.isnan(row["Mean_Per_Call"])
    assert row["Dispatches"] == 2


# get_matrix_ops_type Tests
##############################################################################


def test_get_matrix_ops_type():
    """
    CDNA2/3/4 GPU series should return MFMA.
    Non-CDNA GPU series should return WMMA, including unknown series or empty str.
    """
    from utils.utils_analysis import get_matrix_ops_type

    assert get_matrix_ops_type("MI200") == "MFMA"
    assert get_matrix_ops_type("MI300") == "MFMA"
    assert get_matrix_ops_type("MI350") == "MFMA"

    assert get_matrix_ops_type("navi3") == "WMMA"
    assert get_matrix_ops_type("unknown_series") == "WMMA"
    assert get_matrix_ops_type("") == "WMMA"


# =============================================================================
# TESTS FOR COUNTER IMPUTATION
# =============================================================================


def seed_perfmon_files(tmp_path: Path, count: int) -> None:
    """Create empty pmc_perf_*.yaml files so the imputation function sees the
    expected number of counter buckets. Clears any existing perfmon files
    first so the helper is safe to call multiple times in one test."""
    perfmon = tmp_path / "perfmon"
    perfmon.mkdir(exist_ok=True)
    for stale in perfmon.glob("pmc_perf_*.yaml"):
        stale.unlink()
    for stale in perfmon.glob("*.txt"):
        stale.unlink()
    for i in range(count):
        (perfmon / f"pmc_perf_{i}.yaml").touch()


def test_impute_multiplex_kernel_policy(tmp_path: Path) -> None:
    """Test imputation with kernel policy on a single kernel."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for second dispatch
    assert result["Counter2"].iloc[0] == 500
    assert result["Counter1"].iloc[1] == 100


def test_impute_multiplex_kernel_launch_params_policy(tmp_path: Path) -> None:
    """Test imputation with kernel_launch_params policy on a single kernel."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 512, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    # Assert Counter2 imputed for first dispatch, Counter1 imputed for last dispatch
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_impute_multiplex_kernel_launch_params_no_imputation(tmp_path: Path) -> None:
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 32],
        "LDS_Per_Workgroup": [32, 24, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, 300],
        "Counter2": [None, 500, None],
    }

    df = pd.DataFrame(data)
    # Counter1 and Counter2 form 2 round-robin buckets.
    num_counter_bucket = 2
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Each dispatch still has NaN in the other counter after imputation,
    # so all counter columns are nullified for all dispatches.
    assert pd.isna(result["Counter1"].iloc[0])
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[1])
    assert pd.isna(result["Counter1"].iloc[2])
    assert pd.isna(result["Counter2"].iloc[2])


def test_impute_multiplex_multi_kernel_kernel_policy(tmp_path: Path) -> None:
    """Test imputation with kernel policy on multiple kernels."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)

    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    # Assert Counter1 and Counter2 imputed for first and last dispatches
    assert result["Counter2"].iloc[0] == 300
    assert result["Counter1"].iloc[2] == 100

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows


def test_impute_multiplex_multi_kernel_kernel_launch_params_no_imputation(
    tmp_path: Path,
) -> None:
    """Test imputation with kernel_launch_params when no imputation is possible."""

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 512],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "Counter1": [100, None, None],
        "Counter2": [None, 500, 300],
    }

    df = pd.DataFrame(data)
    # Counter1 and Counter2 form 2 round-robin buckets.
    num_counter_bucket = 2
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )

    # Sort by Dispatch_ID to ensure consistent order
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3  # Ensure same number of rows

    # Each dispatch still has NaN in the other counter after imputation,
    # so all counter columns are nullified for all dispatches.
    assert pd.isna(result["Counter1"].iloc[0])
    assert pd.isna(result["Counter2"].iloc[0])
    assert pd.isna(result["Counter1"].iloc[1])
    assert pd.isna(result["Counter2"].iloc[1])
    assert pd.isna(result["Counter1"].iloc[2])
    assert pd.isna(result["Counter2"].iloc[2])


def test_fewer_dispatches_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with
    fewer dispatches than buckets.

    1 kernel, 3 counters, only 2 dispatches (missing C3 bucket).
    C3 remains NaN because there are no previous_fill_values.
    """

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
        "C3": [None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but the kernel only had 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # C3 was never collected (NaN on all rows after imputation), so all rows are
    # nullified — C1 and C2 are also set to NaN to fully exclude these dispatches.
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])


def test_fewer_dispatches_multiple_kernels_both_incomplete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels, both incomplete.

    kernel_a: buckets {C1}, {C2} (missing C3)
    kernel_b: buckets {C1}, {C2} (missing C3)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_b", "kernel_b"],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 2, 2],
        "C1": [10, None, 40, None],
        "C2": [None, 20, None, 60],
        "C3": [None, None, None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but each kernel has only 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # kernel_a (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # kernel_b (dispatches 3-4): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[2])
    assert pd.isna(result["C2"].iloc[2])
    assert pd.isna(result["C3"].iloc[2])
    assert pd.isna(result["C1"].iloc[3])
    assert pd.isna(result["C2"].iloc[3])
    assert pd.isna(result["C3"].iloc[3])


def test_fewer_dispatches_one_incomplete_one_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on one kernel incomplete, second complete.

    kernel_a: 2 dispatches (missing C3 bucket)
    kernel_b: 3 dispatches covering all 3 buckets
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "GPU_ID": [0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300],
        "Kernel_ID": [1, 1, 2, 2, 2],
        "C1": [10, None, 50, None, None],
        "C2": [None, 20, None, 60, None],
        "C3": [None, None, None, None, 70],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets; kernel_a has only 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # kernel_a (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # kernel_b (dispatches 3-5): all 3 counters fully imputed, no NaN → not nullified
    assert result["C1"].iloc[2] == 50
    assert result["C2"].iloc[2] == 60
    assert result["C3"].iloc[2] == 70
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C3"].iloc[3] == 70
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70


def test_fewer_dispatches_same_kernel_different_launch_params(tmp_path: Path) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy splits into 2 groups, each incomplete.
    Config 1 (Grid=1024, WG=64, LDS=32): buckets {C1}, {C2}
    Config 2 (Grid=512,  WG=32, LDS=16): buckets {C1}, {C2}
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 512, 512],
        "Workgroup_Size": [64, 64, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 1, 1],
        "C1": [10, None, 30, None],
        "C2": [None, 20, None, 40],
        "C3": [None, None, None, None],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but each launch config has 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Config 1 (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Config 2 (dispatches 3-4): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[2])
    assert pd.isna(result["C2"].iloc[2])
    assert pd.isna(result["C3"].iloc[2])
    assert pd.isna(result["C1"].iloc[3])
    assert pd.isna(result["C2"].iloc[3])
    assert pd.isna(result["C3"].iloc[3])


def test_fewer_dispatches_same_kernel_one_incomplete_one_complete(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 2 dispatches (missing C3 bucket)
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches (all 3 buckets)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5],
        "GPU_ID": [0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 512, 512, 512],
        "Workgroup_Size": [64, 64, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300],
        "Kernel_ID": [1, 1, 1, 1, 1],
        "C1": [10, None, 50, None, None],
        "C2": [None, 20, None, 60, None],
        "C3": [None, None, None, None, 70],
    }

    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets; the first launch config has 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 5

    # Config 1 (dispatches 1-2): C3 never collected → all rows nullified
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Config 2 (dispatches 3-5): all 3 counters fully imputed, no NaN → not nullified
    assert result["C1"].iloc[2] == 50
    assert result["C2"].iloc[2] == 60
    assert result["C3"].iloc[2] == 70
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C3"].iloc[3] == 70
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70


def test_incomplete_last_group_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with incomplete last group.

    1 kernel, 2 counters, 3 dispatches (2 buckets, 1 full round + 1 trailing).
    The trailing subgroup uses previous_fill_values to fill its gaps.
    """

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [10, None, 30],
        "C2": [None, 20, None],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3

    # Subgroup 1 (dispatches 1-2): C1 and C2 imputed within the subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Subgroup 2 (dispatch 3, incomplete): C2 filled from previous subgroup
    # via cross-subgroup ffill; no NaN remains so the row is kept as valid.
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20


def test_incomplete_last_group_multiple_kernels_both_incomplete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels,
    both with incomplete last groups.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 5 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        "Start_Timestamp": [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
        ],
        "End_Timestamp": [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
        ],
        "Kernel_ID": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        "C1": [10, None, None, 40, 50, None, None, 80, None],
        "C2": [None, 20, None, None, None, 60, None, None, 90],
        "C3": [None, None, 30, None, None, None, 70, None, None],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 20
    assert result["C3"].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60
    assert result["C3"].iloc[5] == 70
    assert result["C1"].iloc[6] == 50
    assert result["C2"].iloc[6] == 60
    assert result["C3"].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-9, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[7] == 80
    assert result["C2"].iloc[7] == 90
    assert result["C3"].iloc[7] == 70
    assert result["C1"].iloc[8] == 80
    assert result["C2"].iloc[8] == 90
    assert result["C3"].iloc[8] == 70


def test_incomplete_last_group_one_incomplete_other_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on one kernel incomplete, second kernel complete.

    kernel_a: 4 dispatches, 3 buckets {C1},{C2},{C3} (incomplete last)
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3} (2 complete rounds)
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
        ],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        "Start_Timestamp": [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
        ],
        "End_Timestamp": [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
            3300,
        ],
        "Kernel_ID": [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "C1": [10, None, None, 40, 50, None, None, 80, None, None],
        "C2": [None, 20, None, None, None, 60, None, None, 90, None],
        "C3": [None, None, 30, None, None, None, 70, None, None, 100],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10

    # kernel_a subgroup 1 (dispatches 1-3): all 3 counters imputed within subgroup
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a subgroup 2 (dispatch 4, incomplete): filled via cross-subgroup ffill
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 20
    assert result["C3"].iloc[3] == 30

    # kernel_b subgroup 1 (dispatches 5-7): all 3 counters imputed within subgroup
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C3"].iloc[4] == 70
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60
    assert result["C3"].iloc[5] == 70
    assert result["C1"].iloc[6] == 50
    assert result["C2"].iloc[6] == 60
    assert result["C3"].iloc[6] == 70

    # kernel_b subgroup 2 (dispatches 8-10): complete round, no nullification
    assert result["C1"].iloc[7] == 80
    assert result["C2"].iloc[7] == 90
    assert result["C3"].iloc[7] == 100
    assert result["C1"].iloc[8] == 80
    assert result["C2"].iloc[8] == 90
    assert result["C3"].iloc[8] == 100
    assert result["C1"].iloc[9] == 80
    assert result["C2"].iloc[9] == 90
    assert result["C3"].iloc[9] == 100


def test_incomplete_last_group_same_kernel_different_launch_params(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have incomplete last subgroups.
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, 2 buckets
    Config 2 (Grid=512,  WG=32, LDS=16): 3 dispatches, 2 buckets
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500],
        "Kernel_ID": [1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, 50, None, 70],
        "C2": [None, 20, None, None, 60, None],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6

    # Config 1 (dispatches 1-3): subgroup 0 (1-2) complete, subgroup 1 (3) filled
    # via cross-subgroup ffill; no NaN remains so dispatch 3 is kept as valid.
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20

    # Config 2 (dispatches 4-6): subgroup 0 (4-5) complete, subgroup 1 (6) filled
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 70
    assert result["C2"].iloc[5] == 60


def test_incomplete_last_group_same_kernel_one_incomplete_one_complete(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with one config incomplete, other complete.

    kernel_launch_params policy:
    Config 1 (Grid=1024, WG=64, LDS=32): 3 dispatches, incomplete last
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches, 2 complete rounds
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 512, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 32, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 16, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500, 2700],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, 50, None, 70, None],
        "C2": [None, 20, None, None, 60, None, 80],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 7

    # Config 1 (dispatches 1-3): subgroup 0 (1-2) complete, subgroup 1 (3) filled
    # via cross-subgroup ffill; no NaN remains so dispatch 3 is kept as valid.
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 20

    # Config 2 (dispatches 4-7): 2 complete rounds, no nullification
    assert result["C1"].iloc[3] == 50
    assert result["C2"].iloc[3] == 60
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 70
    assert result["C2"].iloc[5] == 80
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80


def test_complete_last_group_single_kernel(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on a single kernel with complete last group.

    1 kernel, 2 counters, 4 dispatches (2 complete rounds of 2 buckets).
    All imputation happens within subgroups; previous_fill_values fallback
    is never needed.
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4],
        "GPU_ID": [0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400, 1600],
        "End_Timestamp": [1500, 1700, 1900, 2100],
        "Kernel_ID": [1, 1, 1, 1],
        "C1": [10, None, 30, None],
        "C2": [None, 20, None, 40],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 4

    # Subgroup 1 (dispatches 1-2): self-contained imputation
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Subgroup 2 (dispatches 3-4): self-contained, values don't bleed from subgroup 1
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 40
    assert result["C1"].iloc[3] == 30
    assert result["C2"].iloc[3] == 40


def test_complete_last_group_multiple_kernels_both_complete(tmp_path: Path) -> None:
    """
    Test imputation with kernel policy on multiple kernels, both complete.

    kernel_a: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    kernel_b: 6 dispatches, 3 buckets {C1},{C2},{C3}, 2 complete rounds
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
            1024,
        ],
        "Workgroup_Size": [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
        "LDS_Per_Workgroup": [
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
            32,
        ],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
            "kernel_b",
        ],
        "Start_Timestamp": [
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            2200,
            2400,
            2600,
            2800,
            3000,
            3200,
        ],
        "End_Timestamp": [
            1500,
            1700,
            1900,
            2100,
            2300,
            2500,
            2700,
            2900,
            3100,
            3300,
            3500,
            3700,
        ],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "C1": [
            10,
            None,
            None,
            40,
            None,
            None,
            70,
            None,
            None,
            100,
            None,
            None,
        ],
        "C2": [
            None,
            20,
            None,
            None,
            50,
            None,
            None,
            80,
            None,
            None,
            110,
            None,
        ],
        "C3": [
            None,
            None,
            30,
            None,
            None,
            60,
            None,
            None,
            90,
            None,
            None,
            120,
        ],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12

    # kernel_a round 1 (dispatches 1-3)
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C3"].iloc[0] == 30
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20
    assert result["C3"].iloc[1] == 30
    assert result["C1"].iloc[2] == 10
    assert result["C2"].iloc[2] == 20
    assert result["C3"].iloc[2] == 30

    # kernel_a round 2 (dispatches 4-6)
    assert result["C1"].iloc[3] == 40
    assert result["C2"].iloc[3] == 50
    assert result["C3"].iloc[3] == 60
    assert result["C1"].iloc[4] == 40
    assert result["C2"].iloc[4] == 50
    assert result["C3"].iloc[4] == 60
    assert result["C1"].iloc[5] == 40
    assert result["C2"].iloc[5] == 50
    assert result["C3"].iloc[5] == 60

    # kernel_b round 1 (dispatches 7-9)
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80
    assert result["C3"].iloc[6] == 90
    assert result["C1"].iloc[7] == 70
    assert result["C2"].iloc[7] == 80
    assert result["C3"].iloc[7] == 90
    assert result["C1"].iloc[8] == 70
    assert result["C2"].iloc[8] == 80
    assert result["C3"].iloc[8] == 90

    # kernel_b round 2 (dispatches 10-12)
    assert result["C1"].iloc[9] == 100
    assert result["C2"].iloc[9] == 110
    assert result["C3"].iloc[9] == 120
    assert result["C1"].iloc[10] == 100
    assert result["C2"].iloc[10] == 110
    assert result["C3"].iloc[10] == 120
    assert result["C1"].iloc[11] == 100
    assert result["C2"].iloc[11] == 110
    assert result["C3"].iloc[11] == 120


def test_complete_last_group_same_kernel_different_launch_params(
    tmp_path: Path,
) -> None:
    """
    Test imputation with kernel_launch_params on the same kernel
    with different launch params.

    kernel_launch_params policy, both configs have 2 complete rounds.
    Config 1 (Grid=1024, WG=64, LDS=32): 4 dispatches
    Config 2 (Grid=512,  WG=32, LDS=16): 4 dispatches
    """

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6, 7, 8],
        "GPU_ID": [0, 0, 0, 0, 0, 0, 0, 0],
        "Grid_Size": [1024, 1024, 1024, 1024, 512, 512, 512, 512],
        "Workgroup_Size": [64, 64, 64, 64, 32, 32, 32, 32],
        "LDS_Per_Workgroup": [32, 32, 32, 32, 16, 16, 16, 16],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [16, 16, 16, 16, 16, 16, 16, 16],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0, 0, 0],
        "SGPR": [32, 32, 32, 32, 32, 32, 32, 32],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
            "kernel_a",
        ],
        "Start_Timestamp": [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400],
        "End_Timestamp": [1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900],
        "Kernel_ID": [1, 1, 1, 1, 1, 1, 1, 1],
        "C1": [10, None, 30, None, 50, None, 70, None],
        "C2": [None, 20, None, 40, None, 60, None, 80],
    }

    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df, "kernel_launch_params", tmp_path
    )
    result = result.sort_values(by="Dispatch_ID")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 8

    # Config 1 round 1 (dispatches 1-2)
    assert result["C1"].iloc[0] == 10
    assert result["C2"].iloc[0] == 20
    assert result["C1"].iloc[1] == 10
    assert result["C2"].iloc[1] == 20

    # Config 1 round 2 (dispatches 3-4)
    assert result["C1"].iloc[2] == 30
    assert result["C2"].iloc[2] == 40
    assert result["C1"].iloc[3] == 30
    assert result["C2"].iloc[3] == 40

    # Config 2 round 1 (dispatches 5-6)
    assert result["C1"].iloc[4] == 50
    assert result["C2"].iloc[4] == 60
    assert result["C1"].iloc[5] == 50
    assert result["C2"].iloc[5] == 60

    # Config 2 round 2 (dispatches 7-8)
    assert result["C1"].iloc[6] == 70
    assert result["C2"].iloc[6] == 80
    assert result["C1"].iloc[7] == 70
    assert result["C2"].iloc[7] == 80


def test_impute_counters_iteration_multiplex_missing_kernel_name(
    tmp_path: Path,
) -> None:
    """
    Test imputation when the DataFrame is a valid 2-level MultiIndex
    but without the Kernel_Name column raises a KeyError.
    """

    data_no_kernel_name = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
    }
    df_no_kn = pd.DataFrame(data_no_kernel_name)
    with pytest.raises(KeyError):
        utils_analysis.impute_counters_iteration_multiplex(df_no_kn, "kernel", tmp_path)


def test_impute_counters_iteration_multiplex_empty_dataframe(tmp_path: Path) -> None:
    """Test imputation when the DataFrame is a valid MultiIndex but has no data rows."""

    data_empty = {
        "Dispatch_ID": [],
        "GPU_ID": [],
        "Grid_Size": [],
        "Workgroup_Size": [],
        "LDS_Per_Workgroup": [],
        "Scratch_Per_Workitem": [],
        "Arch_VGPR": [],
        "Accum_VGPR": [],
        "SGPR": [],
        "Kernel_Name": [],
        "Start_Timestamp": [],
        "End_Timestamp": [],
        "Kernel_ID": [],
        "C1": [],
        "C2": [],
    }
    df_empty = pd.DataFrame(data_empty)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df_empty, "kernel", tmp_path
    )

    # Empty-group fallback preserves the input schema with zero rows.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_empty.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_all_counters_nan(tmp_path: Path) -> None:
    """
    Test imputation when all counter values are NaN.

    The bucket-identification loop finds no non-empty frozensets, so
    counter_groups stays empty and the group is skipped entirely.
    """

    data_all_nan = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [None, None, None],
        "C2": [None, None, None],
    }
    df_all_nan = pd.DataFrame(data_all_nan)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df_all_nan, "kernel", tmp_path
    )

    # Group was dropped (no valid counters) -- empty schema-aligned frame.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_all_nan.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_no_counter_columns(tmp_path: Path) -> None:
    """
    Test imputation when the DataFrame contains only the 13 non-counter columns.

    counter_columns is empty, so every row yields an empty frozenset
    and the group is skipped.
    """

    data_no_counters = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
    }
    df_no_counters = pd.DataFrame(data_no_counters)
    result = utils_analysis.impute_counters_iteration_multiplex(
        df_no_counters, "kernel", tmp_path
    )

    # Group was dropped (no counter columns exist) -- empty schema-aligned frame.
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df_no_counters.columns)
    assert len(result) == 0


def test_impute_counters_iteration_multiplex_unrecognized_policy(
    tmp_path: Path,
) -> None:
    """
    Test imputation when the policy is unrecognized.
    Any policy other than "kernel" falls through to the else branch
    (same as "kernel_launch_params"). The output must match exactly.
    """

    data_policy = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [1024, 1024, 1024],
        "Workgroup_Size": [64, 64, 64],
        "LDS_Per_Workgroup": [32, 32, 32],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [16, 16, 16],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [32, 32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Kernel_ID": [1, 1, 1],
        "C1": [100, None, None],
        "C2": [None, 500, 300],
    }
    df_policy = pd.DataFrame(data_policy)
    result_invalid = utils_analysis.impute_counters_iteration_multiplex(
        df_policy, "invalid_policy", tmp_path
    )
    result_klp = utils_analysis.impute_counters_iteration_multiplex(
        df_policy, "kernel_launch_params", tmp_path
    )
    assert isinstance(result_invalid, pd.DataFrame)
    pd.testing.assert_frame_equal(
        result_invalid.sort_values(by="Dispatch_ID").reset_index(drop=True),
        result_klp.sort_values(by="Dispatch_ID").reset_index(drop=True),
    )


def test_incomplete_dispatches_nullify_counter_values(tmp_path: Path) -> None:
    """
    After imputation, any dispatch row that still has at least one NaN counter
    value should have ALL counter columns set to NaN (fully nullified).
    Non-counter columns (timestamps, kernel name, etc.) must be preserved so
    that Top Stats (Block 1) timing data remains accurate.

    Scenario:
      kernel_a: 2 dispatches, 3 counter buckets {C1}, {C2}, {C3}.
      Only 2 dispatches are available so the {C3} bucket is never reached:
        - Dispatch 1: C1=10, C2=NaN, C3=NaN
        - Dispatch 2: C1=NaN, C2=20, C3=NaN
      After bfill/ffill imputation:
        - C1 and C2 are filled for both dispatches (C1=10, C2=20)
        - C3 remains NaN for both dispatches (never collected)
      Post-imputation nullification:
        - Both dispatches have C3=NaN -> all counter columns set to NaN
        - Timestamp and Kernel_Name columns are preserved
    """
    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, None],
        "C2": [None, 20],
        "C3": [None, None],
    }
    df = pd.DataFrame(data)
    # C1, C2, C3 form 3 round-robin buckets but the kernel only had 2 dispatches.
    num_counter_bucket = 3
    seed_perfmon_files(tmp_path, count=num_counter_bucket)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID").reset_index(drop=True)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    # Both dispatches: C3 was never collected so it remains NaN after imputation,
    # triggering nullification of all counter columns on both rows.
    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C3"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])
    assert pd.isna(result["C3"].iloc[1])

    # Non-counter columns must still be populated on both dispatches
    # (preserved for Top Stats / Block 1 timing display).
    assert result["Start_Timestamp"].iloc[0] == 1000
    assert result["End_Timestamp"].iloc[0] == 1500
    assert result["Kernel_Name"].iloc[0] == "kernel_a"
    assert result["Start_Timestamp"].iloc[1] == 1200
    assert result["End_Timestamp"].iloc[1] == 1700
    assert result["Kernel_Name"].iloc[1] == "kernel_a"


def test_undersampled_kernel_nullified_against_perfmon_file_count(
    tmp_path: Path,
) -> None:
    """
    A kernel whose dispatch count is below the number of configured perfmon
    files must be nullified even when its visible counter columns are fully
    imputed. This guards the degenerate case where some buckets never reached
    the joined dataframe at all.
    """
    # 5 perfmon buckets configured, kernel only has 2 dispatches; the visible
    # counters look fully populated but bucket coverage is incomplete.
    num_counter_bucket = 5
    seed_perfmon_files(tmp_path, count=num_counter_bucket)

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [1024, 1024],
        "Workgroup_Size": [64, 64],
        "LDS_Per_Workgroup": [32, 32],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [16, 16],
        "Accum_VGPR": [0, 0],
        "SGPR": [32, 32],
        "Kernel_Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200],
        "End_Timestamp": [1500, 1700],
        "Kernel_ID": [1, 1],
        "C1": [10, 30],
        "C2": [20, 40],
    }
    df = pd.DataFrame(data)
    result = utils_analysis.impute_counters_iteration_multiplex(df, "kernel", tmp_path)
    result = result.sort_values(by="Dispatch_ID").reset_index(drop=True)

    assert pd.isna(result["C1"].iloc[0])
    assert pd.isna(result["C2"].iloc[0])
    assert pd.isna(result["C1"].iloc[1])
    assert pd.isna(result["C2"].iloc[1])

    # Timestamps and kernel name preserved for Top Stats.
    assert result["Start_Timestamp"].iloc[0] == 1000
    assert result["Kernel_Name"].iloc[0] == "kernel_a"
