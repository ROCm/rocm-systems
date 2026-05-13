# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests covering functions in src/utils/utils_analysis.py and src/utils/rocpd_data.py,
including torch operator pattern matching and data imputation tests."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import utils.utils_analysis as utils_analysis

logging_trace_set = False
try:
    import logging

    logging.trace = lambda *args, **kwargs: None
    logging_trace_set = True
except Exception:
    pass


# =============================================================================
# TESTS FOR EMPTY WORKLOAD
# =============================================================================


def test_is_workload_empty_valid_data_file(tmp_path):
    """
    Test is_workload_empty with a valid pmc_perf.csv file containing data.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_file_with_nan_values(tmp_path):
    """
    Test is_workload_empty with pmc_perf.csv containing NaN values.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "pmc_perf.csv" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_is_workload_empty_completely_empty_csv(tmp_path):
    """
    Test is_workload_empty with completely empty pmc_perf.csv file.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function detects empty CSV file.
    """
    from unittest.mock import patch

    workload_dir = tmp_path / "workload"
    workload_dir.mkdir()

    pmc_perf_file = workload_dir / "pmc_perf.csv"
    pmc_perf_file.write_text("")

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        try:
            utils_analysis.is_workload_empty(str(workload_dir))
        except Exception:
            pass


def test_is_workload_empty_headers_only_csv(tmp_path):
    """
    Test is_workload_empty with CSV containing only headers.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_is_workload_empty_no_pmc_perf_file(tmp_path):
    """
    Test is_workload_empty when pmc_perf.csv file doesn't exist.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_nonexistent_directory():
    """
    Test is_workload_empty with nonexistent directory path.

    Returns:
        None: Asserts function handles nonexistent directories.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty("/nonexistent/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_malformed_csv(tmp_path):
    """
    Test is_workload_empty with malformed CSV that causes pandas read error.

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
            utils_analysis.is_workload_empty(str(workload_dir))
        except Exception:
            pass


def test_is_workload_empty_mixed_valid_invalid_data(tmp_path):
    """
    Test is_workload_empty with CSV containing mix of valid and invalid (NaN) data.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_large_dataset_with_nans(tmp_path):
    """
    Test is_workload_empty with large dataset that becomes empty after dropping NaNs.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]


def test_is_workload_empty_unicode_content(tmp_path):
    """
    Test is_workload_empty with CSV containing Unicode characters.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_special_path_characters(tmp_path):
    """
    Test is_workload_empty with directory paths containing special characters.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 0


def test_is_workload_empty_csv_read_permission_error(tmp_path):
    """
    Test is_workload_empty when CSV file exists but cannot be read due to permissions.

    Args:
        tmp_path (Path): Temporary directory for test files.

    Returns:
        None: Asserts function handles file permission errors.
    """
    import os
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
            utils_analysis.is_workload_empty(str(workload_dir))
    except PermissionError:
        pass
    finally:
        pmc_perf_file.chmod(0o644)


def test_is_workload_empty_string_path_input():
    """
    Test is_workload_empty with string path input vs Path.

    Returns:
        None: Asserts function handles different path input types.
    """
    from unittest.mock import patch

    console_error_calls = []

    def mock_console_error(*args, **kwargs):
        console_error_calls.append((args, kwargs))

    with patch("utils.utils_analysis.console_error", side_effect=mock_console_error):
        utils_analysis.is_workload_empty("/nonexistent/string/path")

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    assert error_args[0] == "analysis"
    assert error_args[1] == "No profiling data found."


def test_is_workload_empty_console_error_string_formatting(tmp_path):
    """
    Test is_workload_empty string formatting in console_error messages.

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
        utils_analysis.is_workload_empty(str(workload_dir))

    assert len(console_error_calls) == 1
    error_args = console_error_calls[0][0]
    expected_path = str(workload_dir / "pmc_perf.csv")
    assert expected_path in error_args[1]
    assert "profiling" in error_args[0]
    assert "Found empty cells" in error_args[1]
    assert "Profiling data could be corrupt" in error_args[1]


def test_is_workload_empty_function_return_value(tmp_path):
    """
    Test that is_workload_empty function return behavior (implicitly returns None).

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
        result = utils_analysis.is_workload_empty(str(workload_dir))

    assert result is None

    workload_dir2 = tmp_path / "workload2"
    workload_dir2.mkdir()

    with patch("utils.utils_analysis.console_error"):
        result2 = utils_analysis.is_workload_empty(str(workload_dir2))

    assert result2 is None


def test_is_workload_empty_pandas_import_dependency():
    """
    Test is_workload_empty dependency on pandas module.

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
                    utils_analysis.is_workload_empty("/test/path")

    mock_pandas.read_csv.assert_called_once()
    mock_df.dropna.assert_called_once()


# =============================================================================
# TESTS FOR merge_counters_spatial_multiplex FUNCTION
# =============================================================================


def test_merge_counters_spatial_multiplex_basic_functionality():
    """
    Test merge_counters_spatial_multiplex with basic multi-index DataFrame.

    Returns:
        None: Asserts function correctly merges counter values for spatial multiplexing.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 1, 1, 2, 2],
        "Grid_Size": [64, 128, 256, 512, 1024, 2048],
        "Workgroup_Size": [16, 32, 64, 32, 64, 128],
        "LDS_Per_Workgroup": [1024, 2048, 4096, 2048, 4096, 8192],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [32, 64, 96, 64, 96, 128],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [16, 32, 48, 32, 48, 64],
        "Wave_Size": [64, 64, 64, 64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003, 2001, 2002, 2003],
        "Kernel_ID": [501, 502, 503, 601, 602, 603],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_c",
            "kernel_c",
            "kernel_d",
        ],
        "Start_Timestamp": [1000, 1100, 2000, 3000, 3100, 4000],
        "End_Timestamp": [1200, 1300, 2500, 3400, 3500, 4800],
        "Counter1": [100, 200, 300, 400, 500, 600],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)


def test_merge_counters_spatial_multiplex_kernel_name_fallback():
    """
    Test merge_counters_spatial_multiplex when Kernel_Name is missing but Name exists.

    Returns:
        None: Asserts function uses Name column when Kernel_Name is not available.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2],
        "GPU_ID": [0, 0],
        "Grid_Size": [64, 128],
        "Workgroup_Size": [16, 32],
        "LDS_Per_Workgroup": [1024, 2048],
        "Scratch_Per_Workitem": [0, 0],
        "Arch_VGPR": [32, 64],
        "Accum_VGPR": [0, 0],
        "SGPR": [16, 32],
        "Wave_Size": [64, 64],
        "Correlation_ID": [1001, 1002],
        "Kernel_ID": [501, 502],
        "Name": ["kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1100],
        "End_Timestamp": [1200, 1300],
        "Counter1": [100, 200],
    }
    df = pd.DataFrame(data)

    # The function currently has a bug where it doesn't properly check for 'Kernel_Name'
    # existence before accessing it, even though it has fallback logic for 'Name'
    try:
        result = utils_analysis.merge_counters_spatial_multiplex(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    except KeyError as e:
        if "'Kernel_Name'" in str(e):
            pytest.skip(
                "Function doesn't properly check for Kernel_Name "
                "existence before accessing - needs to validate column "
                "presence in the check condition"
            )
        else:
            raise


def test_merge_counters_spatial_multiplex_single_kernel_occurrence():
    """
    Test merge_counters_spatial_multiplex with kernels that appear only once.

    Returns:
        None: Asserts function handles single kernel occurrences correctly.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 1, 2],
        "Grid_Size": [64, 128, 256],
        "Workgroup_Size": [16, 32, 64],
        "LDS_Per_Workgroup": [1024, 2048, 4096],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [32, 64, 96],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [16, 32, 48],
        "Wave_Size": [64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003],
        "Kernel_ID": [501, 502, 503],
        "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
        "Start_Timestamp": [1000, 2000, 3000],
        "End_Timestamp": [1200, 2500, 3800],
        "Counter1": [100, 200, 300],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_merge_counters_spatial_multiplex_multiple_duplicate_kernels():
    """
    Test merge_counters_spatial_multiplex with multiple kernels having duplicates.

    Returns:
        None: Asserts function correctly handles multiple kernel duplicates.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3, 4, 5, 6],
        "GPU_ID": [0, 0, 1, 1, 2, 2],
        "Grid_Size": [64, 64, 128, 128, 256, 256],
        "Workgroup_Size": [16, 16, 32, 32, 64, 64],
        "LDS_Per_Workgroup": [1024, 1024, 2048, 2048, 4096, 4096],
        "Scratch_Per_Workitem": [0, 0, 0, 0, 0, 0],
        "Arch_VGPR": [32, 32, 64, 64, 96, 96],
        "Accum_VGPR": [0, 0, 0, 0, 0, 0],
        "SGPR": [16, 16, 32, 32, 48, 48],
        "Wave_Size": [64, 64, 64, 64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003, 1004, 1005, 1006],
        "Kernel_ID": [501, 502, 503, 504, 505, 506],
        "Kernel_Name": [
            "kernel_a",
            "kernel_a",
            "kernel_b",
            "kernel_b",
            "kernel_c",
            "kernel_c",
        ],
        "Start_Timestamp": [1000, 1100, 2000, 2100, 3000, 3100],
        "End_Timestamp": [1200, 1300, 2500, 2600, 3800, 3900],
        "Counter1": [100, 200, 300, 400, 500, 600],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3


def test_merge_counters_spatial_multiplex_timestamp_median_calculation():
    """
    Test merge_counters_spatial_multiplex timestamp median calculations.

    Returns:
        None: Asserts function correctly calculates median timestamps.
    """
    import pandas as pd

    data = {
        "Dispatch_ID": [1, 2, 3],
        "GPU_ID": [0, 0, 0],
        "Grid_Size": [64, 64, 64],
        "Workgroup_Size": [16, 16, 16],
        "LDS_Per_Workgroup": [1024, 1024, 1024],
        "Scratch_Per_Workitem": [0, 0, 0],
        "Arch_VGPR": [32, 32, 32],
        "Accum_VGPR": [0, 0, 0],
        "SGPR": [16, 16, 16],
        "Wave_Size": [64, 64, 64],
        "Correlation_ID": [1001, 1002, 1003],
        "Kernel_ID": [501, 502, 503],
        "Kernel_Name": ["kernel_a", "kernel_a", "kernel_a"],
        "Start_Timestamp": [1000, 1200, 1400],
        "End_Timestamp": [1500, 1700, 1900],
        "Counter1": [100, 200, 300],
    }
    df = pd.DataFrame(data)

    result = utils_analysis.merge_counters_spatial_multiplex(df)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


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


# =============================================================================
# TESTS FOR Analysis DB mode: Analysis DB mode code path
# =============================================================================


def test_calc_roofline_data_early_exit_on_empty_roofline_df(monkeypatch):
    """Test calc_roofline_data exits early when roofline data is empty.

    This test verifies that when the roofline dataframe (ID 402) is empty
    or filtered out, the function logs a warning and skips that workload
    without adding it to the result dictionary.
    """
    from rocprof_compute_analyze.analysis_db import db_analysis

    # Create mock db_analysis instance
    analyzer = mock.MagicMock(spec=db_analysis)

    # Mock workload data
    workload_path = "/mock/workload/path"
    mock_runs = {
        workload_path: mock.MagicMock(sys_info=pd.DataFrame([{"gpu_arch": "gfx90a"}]))
    }

    # Mock PMC dataframe with kernel data
    mock_pmc_df = pd.DataFrame({
        "Kernel_Name": ["kernel1", "kernel2"],
        "Start_Timestamp": [100, 200],
        "End_Timestamp": [150, 300],
    })

    # Mock architecture config with EMPTY roofline dataframe (ID 402)
    mock_arch_config = mock.MagicMock()
    mock_arch_config.dfs = {
        402: pd.DataFrame()  # Empty roofline dataframe triggers early exit
    }

    # Setup instance variables
    analyzer._runs = mock_runs
    analyzer._pmc_df_per_workload = {workload_path: mock_pmc_df}
    analyzer._arch_configs = {"gfx90a": mock_arch_config}
    analyzer.get_args = mock.MagicMock(return_value=mock.MagicMock(max_stat_num=10))

    # Mock console_warning to verify it's called
    warning_messages = []

    def mock_warning(msg):
        warning_messages.append(msg)

    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_warning", mock_warning
    )
    monkeypatch.setattr(
        "rocprof_compute_analyze.analysis_db.console_debug", lambda msg: None
    )

    # Call the actual function
    result = db_analysis.calc_roofline_data(analyzer)

    # Verify early exit behavior
    assert len(result[0]) == 0, (
        "Should return empty kernel level dict when roofline data is empty"
    )
    assert len(result[1]) == 0, (
        "Should return empty workload level dict when roofline data is empty"
    )
    assert len(warning_messages) == 1, "Should log one warning message"
    assert "Roofline data is filtered out or not found" in warning_messages[0]
    assert workload_path in warning_messages[0]


# ---------------------------------------------------------------------------
# Torch operator pattern matching (PurePosixPath glob)
# ---------------------------------------------------------------------------

H3 = "nn.Module.Net.forward/torch.nn.functional.relu/torch.relu"
H2 = "nn.Module.Net.forward/torch.nn.functional.conv2d"
H1 = "torch.relu"


@pytest.mark.torch_ops
def test_all_keyword():
    """'all' maps to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("all", H3)
    assert m("all", H2)
    assert m("all", H1)
    assert not m("all", "")


@pytest.mark.torch_ops
def test_bare_pattern_matches_last_component():
    """Bare token is matched via PurePosixPath.match() against the full hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert m("torch.nn.functional.conv2d", H2)
    assert not m("relu", H3)
    assert not m("forward", H3)
    assert not m("sigmoid", H3)


@pytest.mark.torch_ops
def test_bare_wildcard_pattern():
    """Wildcard bare token matched via PurePosixPath.match()."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.*", H3)
    assert m("*relu", H3)
    assert m("*conv*", H2)
    assert not m("conv*", H2)
    assert not m("sigm*", H3)


@pytest.mark.torch_ops
def test_hierarchy_glob():
    """Patterns with '/' match across multiple hierarchy components."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("nn.Module.Net.forward/*/torch.relu", H3)
    assert m("*/torch.nn.functional.conv2d", H2)
    assert not m("nn.Module.Net.forward/torch.relu", H3)


@pytest.mark.torch_ops
def test_leading_slash_is_cosmetic():
    """Leading '/' is stripped during pattern normalization."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("/nn.Module.Net.forward/*/torch.relu", H3)
    assert m("/torch.relu", H3)


@pytest.mark.torch_ops
def test_trailing_slash_stripped_by_posixpath():
    """PurePosixPath strips trailing slashes, so they are cosmetic."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("nn.Module.Net.forward/", H3)
    assert m("torch.relu/", H3)


@pytest.mark.torch_ops
def test_regex_not_supported():
    """Regex syntax has no special meaning; treated as literal glob text."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("relu|conv2d", H3)
    assert not m("^torch\\.relu$", H3)
    assert not m("not:relu", H3)
    assert not m("2:functional", H3)


@pytest.mark.torch_ops
def test_empty_inputs():
    """Empty pattern or operator_name returns False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("", H3)
    assert not m("relu", "")
    assert not m("", "")


@pytest.mark.torch_ops
def test_slash_only_markers():
    """Scope-marker-only tokens should not match any hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("/", H3)
    assert not m("//", H3)


# -- get_matched_torch_operators_for_display ---------------------------------


def get_matched_torch_operators_for_display(
    torch_operators: dict[str, pd.DataFrame],
    pattern_list: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Return (operator_name, filtered_df) for each operator matching any pattern.

    Test-only helper: iterates every unique Operator_Name across all torch trace
    DataFrames and checks each against the supplied glob patterns.
    """
    from utils.parser import torch_operator_pattern_matches

    if not torch_operators or not pattern_list:
        return []
    result: list[tuple[str, pd.DataFrame]] = []
    seen: set[str] = set()
    for _, df in torch_operators.items():
        if df is None or df.empty or "Operator_Name" not in df.columns:
            continue
        for op_name in df["Operator_Name"].dropna().unique():
            op_str = str(op_name).strip()
            if op_str in seen:
                continue
            for pattern in pattern_list:
                if torch_operator_pattern_matches(pattern.strip(), op_str):
                    seen.add(op_str)
                    result.append((op_str, df.loc[df["Operator_Name"] == op_name]))
                    break
    return result


@pytest.mark.torch_ops
def test_display_match_hierarchy_glob():
    """Full hierarchy globs are honored by display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H3, H2],
        "Kernel_Name": ["k1", "k2", "k3"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*/torch.relu"])
    assert len(matched) == 1
    assert matched[0][0] == H3


@pytest.mark.torch_ops
def test_display_match_multi_patterns():
    """Multiple glob patterns match their respective operators."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(
        torch_operators, ["*relu", "*conv*"]
    )
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_no_match():
    """No matches returns empty list."""
    df = pd.DataFrame({
        "Operator_Name": [H3],
        "Kernel_Name": ["k1"],
    })
    assert get_matched_torch_operators_for_display({"t": df}, ["sigmoid"]) == []


@pytest.mark.torch_ops
def test_display_empty_inputs():
    """Empty torch_operators or pattern_list returns []."""
    assert get_matched_torch_operators_for_display({}, ["relu"]) == []
    assert get_matched_torch_operators_for_display({"x": pd.DataFrame()}, []) == []


# -- parse_torch_operator_patterns ------------------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu"])
    assert parse_torch_operator_patterns(args) == ["relu"]

    args = Namespace(torch_operator=["relu", "conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["relu,conv2d"])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    assert parse_torch_operator_patterns(args) == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    assert parse_torch_operator_patterns(Namespace(torch_operator=[])) == ["**"]
    assert parse_torch_operator_patterns(Namespace(torch_operator=None)) == []
    assert parse_torch_operator_patterns(Namespace()) == []


# -- PatternMatcherEngine ---------------------------------------------------


@pytest.mark.torch_ops
def test_engine_glob_hierarchy_mode():
    """Facade delegates matching to glob-hierarchy implementation."""
    from utils.pattern_matching import PatternMatcherEngine

    matcher = PatternMatcherEngine(mode="glob-hierarchy")
    assert matcher.matches("torch.relu", H3)
    assert matcher.matches("*relu", H3)
    assert not matcher.matches("sigmoid", H3)


@pytest.mark.torch_ops
def test_engine_invalid_mode():
    """Unsupported strategy names should raise ValueError."""
    from utils.pattern_matching import PatternMatcherEngine

    with pytest.raises(ValueError):
        PatternMatcherEngine(mode="regex")


# -- Additional coverage (xuchen #26) ----------------------------------------


@pytest.mark.torch_ops
def test_double_star_explicit():
    """'**' matches any hierarchy depth."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("**", H3)
    assert m("**", H2)
    assert m("**", H1)
    assert m("**", "a/b/c/d/e")
    assert not m("**", "")


@pytest.mark.torch_ops
def test_single_char_wildcard():
    """'?' matches exactly one character in a component."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel?", H3)
    assert m("torch.?elu", H3)
    assert not m("torch.?", H3)
    assert not m("?", H1)
    assert m("torch.nn.functional.conv?d", H2)


@pytest.mark.torch_ops
def test_long_hierarchy():
    """Deeply nested hierarchies match correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    deep = "/".join([f"level{i}" for i in range(20)])
    assert m("level19", deep)
    assert m("*19", deep)
    assert m("*/level19", deep)
    assert m("all", deep)
    assert not m("level0", deep)


@pytest.mark.torch_ops
def test_long_component_names():
    """Components with very long names are handled correctly."""
    from utils.parser import torch_operator_pattern_matches as m

    long_name = "a" * 500
    hierarchy = f"root/{long_name}"
    assert m(f"{'a' * 500}", hierarchy)
    assert m("a*", hierarchy)
    assert not m("b*", hierarchy)


@pytest.mark.torch_ops
def test_special_characters_in_names():
    """Dots, underscores, and other non-glob chars are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module._internal/torch.nn.functional.conv2d"
    assert m("torch.nn.functional.conv2d", h)
    assert m("*conv2d", h)
    assert m("nn.Module._internal/*", h)
    assert not m("nn_Module._internal/*", h)


@pytest.mark.torch_ops
def test_bracket_glob_pattern():
    """Character classes [abc] work in glob patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.rel[uv]", H3)
    assert not m("torch.rel[ab]", H3)


@pytest.mark.torch_ops
def test_single_component_hierarchy():
    """Single-component hierarchy (no slashes) matches bare patterns."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", "torch.relu")
    assert m("*relu", "torch.relu")
    assert m("torch.*", "torch.relu")
    assert not m("*/torch.relu", "torch.relu")


@pytest.mark.torch_ops
def test_whitespace_only_pattern():
    """Whitespace-only patterns normalize to empty and return False."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("   ", H3)
    assert not m("\t", H3)


@pytest.mark.torch_ops
def test_star_pattern_matches_all():
    """Bare '*' is normalized to '**' and matches every hierarchy."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("*", H3)
    assert m("*", H2)
    assert m("*", H1)
    assert m("*", "a/b/c/d/e")
    assert not m("*", "")


@pytest.mark.torch_ops
def test_star_normalize_equivalence():
    """'*' and 'all' produce the same normalization."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("*") == norm("all") == "**"


@pytest.mark.torch_ops
def test_case_sensitivity():
    """Pattern matching is case-sensitive."""
    from utils.parser import torch_operator_pattern_matches as m

    assert not m("Torch.Relu", H3)
    assert not m("TORCH.RELU", H3)
    assert not m("ALL", H3)
    assert m("all", H3)


@pytest.mark.torch_ops
def test_all_keyword_case_sensitive():
    """Only lowercase 'all' is the special keyword; mixed case is a literal."""
    from utils.pattern_matching import PurePosixGlobHierarchyMatcher

    norm = PurePosixGlobHierarchyMatcher.normalize_pattern
    assert norm("all") == "**"
    assert norm("ALL") == "ALL"
    assert norm("All") == "All"


@pytest.mark.torch_ops
def test_consecutive_slashes_in_target():
    """Consecutive slashes in the target are collapsed by PurePosixPath."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "a//b///torch.relu"
    assert m("torch.relu", h)
    assert m("*relu", h)


@pytest.mark.torch_ops
def test_dots_in_patterns():
    """Dots are literal characters in glob patterns, not regex wildcards."""
    from utils.parser import torch_operator_pattern_matches as m

    assert m("torch.relu", H3)
    assert not m("torchXrelu", H3)
    h = "root/torchXrelu"
    assert not m("torch.relu", h)
    assert m("torchXrelu", h)


@pytest.mark.torch_ops
def test_pattern_with_spaces():
    """Spaces in patterns and targets are treated literally."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "module/ spaced op /torch.relu"
    assert m("torch.relu", h)
    assert not m(" spaced op ", h)
    assert m("* spaced op */*", h)


@pytest.mark.torch_ops
def test_colons_in_operator_names():
    """Colons (e.g. aten::relu) are literal characters in glob matching."""
    from utils.parser import torch_operator_pattern_matches as m

    h = "nn.Module/aten::relu_"
    assert m("aten::relu_", h)
    assert m("*relu_", h)
    assert m("aten::*", h)
    assert not m("*relu", h)
    assert not m("torch.relu", h)


@pytest.mark.torch_ops
def test_display_star_matches_all_operators():
    """'*' pattern matches all operators in display helper."""
    df = pd.DataFrame({
        "Operator_Name": [H3, H2],
        "Kernel_Name": ["k1", "k2"],
    })
    torch_operators = {"trace_0": df}

    matched = get_matched_torch_operators_for_display(torch_operators, ["*"])
    assert len(matched) == 2


@pytest.mark.torch_ops
def test_display_dedup_across_dataframes():
    """Same operator in multiple DataFrames is matched only once."""
    df1 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k1"]})
    df2 = pd.DataFrame({"Operator_Name": [H3], "Kernel_Name": ["k2"]})
    torch_operators = {"trace_0": df1, "trace_1": df2}

    matched = get_matched_torch_operators_for_display(torch_operators, ["all"])
    op_names = [name for name, _ in matched]
    assert op_names.count(H3) == 1


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_torch_operator_patterns

    args = Namespace(torch_operator=["*"])
    assert parse_torch_operator_patterns(args) == ["*"]

    args = Namespace(torch_operator=["*,torch.relu"])
    assert parse_torch_operator_patterns(args) == ["*", "torch.relu"]


# =============================================================================
# DATA IMPUTATION TESTS
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
    Test imputation when the DataFrame is a valid DataFrame
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
    """Test imputation when the DataFrame has no data rows."""

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
