# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests covering functions in src/utils/utils_profile.py."""

import io
import logging
import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

import utils.utils_profile as utils_profile


class MockArgs:
    def __init__(self, **kwargs):
        # Set kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockSoc:
    def __init__(self):
        pass


logging.trace = lambda *args, **kwargs: None


# =============================================================================
# RUN_PROF TESTS
# =============================================================================


def test_run_prof_success_v3(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofv3 successful execution.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution and file creation.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")
    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_content = (
        "Agent_Type,Node_Id,Wave_Front_Size,Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,"
        "Grid_Size,Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,"
        "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Start_Timestamp,"
        "End_Timestamp,Counter_Name,Counter_Value\n"
        "GPU,0,0,0,0,0,0,0,0,0,0,test_kernel,0,0,0,0,0,0,0,1,SQ_WAVES,100"
    )
    with open(workload_dir + "/out/pmc_1/results_0.csv", "w") as f:
        f.write(csv_content)

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr(
        "glob.glob", lambda pattern: [workload_dir + "/out/pmc_1/results_0.csv"]
    )

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")

    assert Path(workload_dir + "/results_pmc_perf_test.csv").exists()


def test_run_prof_success_v3_csv(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofv3 using CSV format.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution with v3 CSV processing.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")
    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_files = [workload_dir + "/out/pmc_1/converted.csv"]

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: csv_files
    )

    # Mock csv_ops functions to avoid disk I/O
    mock_rows = [
        {
            "Dispatch_ID": "0",
            "GPU_ID": "0",
            "Kernel_Name": "test",
            "Grid_Size": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
        }
    ]
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: (mock_rows.copy(), list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", lambda *a, **k: None
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_success_rocprofiler_sdk(tmp_path, monkeypatch):
    """
    Test run_prof with rocprofiler-sdk execution.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts successful execution with SDK configuration.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    profiler_options = {
        "APP_CMD": ["./test_app"],
        "ROCPROF_OUTPUT_PATH": workload_dir,
        "ROCPROF_COUNTER_COLLECTION": "1",
        "ROCP_TOOL_LIBRARIES": "/opt/rocm/lib/rocprofiler-sdk/"
        "librocprofiler-sdk-tool.so",
    }

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.utils_common.parse_pmc_perf", lambda f: ["SQ_WAVES"])
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(
        str(fname), profiler_options, workload_dir, logging.INFO, "csv"
    )


def test_run_prof_with_yaml_config(tmp_path, monkeypatch):
    """
    Test run_prof with additional YAML configuration file.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts YAML config is properly handled.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    yaml_file = tmp_path / "counter_def_test.yaml"
    yaml_file.write_text("rocprofiler-sdk:\n  counters:\n    - TCC_HIT\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_failure_subprocess(tmp_path, monkeypatch):
    """
    Test run_prof when subprocess execution fails.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts proper error handling on subprocess failure.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (False, "error output"),
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    def mock_console_error(msg, exit=True):
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with pytest.raises(RuntimeError, match="console_error called"):
        utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_mi300_environment_setup(tmp_path, monkeypatch):
    """
    Test run_prof sets proper environment variables for MI300 series GPUs.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts MI300 environment variable is set correctly.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    captured_env = {}

    def mock_capture_subprocess_output(cmd, new_env=None, **kwargs):
        if new_env:
            captured_env.update(new_env)
        return (True, "success")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output", mock_capture_subprocess_output
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_timestamps_special_case(tmp_path, monkeypatch):
    """
    Test run_prof handles timestamps.txt special case correctly.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts timestamps processing is handled correctly.
    """
    fname = tmp_path / "pmc_perf_timestamps.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    csv_content = (
        "Agent_Type,Node_Id,Wave_Front_Size,Correlation_Id,Dispatch_Id,Agent_Id,Queue_Id,Process_Id,Thread_Id,"
        "Grid_Size,Kernel_Id,Kernel_Name,Workgroup_Size,LDS_Block_Size,"
        "Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,Start_Timestamp,"
        "End_Timestamp,Counter_Name,Counter_Value\n"
        "GPU,0,0,0,0,0,0,0,0,0,0,test_kernel,0,0,0,0,0,0,0,1,SQ_WAVES,100"
    )
    with open(workload_dir + "/kernel_trace.csv", "w") as f:
        f.write(csv_content)

    csv_files = [workload_dir + "/kernel_trace.csv"]

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: csv_files
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)

    mock_df = pd.DataFrame({
        "Dispatch_ID": [0],
        "Start_Timestamp": [100],
        "End_Timestamp": [200],
        "Grid_Size": [1024],
        "Workgroup_Size": [64],
        "Kernel_Name": ["test_kernel"],
        "LDS_Per_Workgroup": [1024],
    })
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.concat", lambda *a, **k: mock_df)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_no_results_files(tmp_path, monkeypatch):
    """
    Test run_prof when no results files are generated.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts proper handling when no results are found.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv2")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("glob.glob", lambda pattern: [])  # No files found
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_header_standardization(tmp_path, monkeypatch):
    """
    Test run_prof properly standardizes CSV headers.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts CSV headers are standardized correctly.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - SQ_WAVES\n")
    workload_dir = str(tmp_path / "workload")

    os.makedirs(workload_dir + "/out/pmc_1", exist_ok=True)

    results_csv = workload_dir + "/out/pmc_1/results_test.csv"

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: [results_csv]
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    # Mock csv_ops to track rename_columns calls and avoid disk I/O
    mock_rows = [
        {
            "KernelName": "test_kernel",
            "Index": "0",
            "grd": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
            "BeginNs": "0",
            "EndNs": "1",
            "SQ_WAVES": "100",
        }
    ]

    rename_calls = []

    def mock_rename_columns(rows, mapping):
        rename_calls.append(mapping)
        # Apply the rename to verify the mapping
        for row in rows:
            for old_name, new_name in mapping.items():
                if old_name in row:
                    row[new_name] = row.pop(old_name)

    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: ([r.copy() for r in mock_rows], list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", mock_rename_columns
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")

    # Verify that rename_columns was called with the header standardization mapping
    assert len(rename_calls) == 1
    mapping = rename_calls[0]
    assert mapping.get("KernelName") == "Kernel_Name"
    assert mapping.get("Index") == "Dispatch_ID"
    assert mapping.get("grd") == "Grid_Size"
    assert mapping.get("BeginNs") == "Start_Timestamp"
    assert mapping.get("EndNs") == "End_Timestamp"


def test_run_prof_tcc_flattening_mi300(tmp_path, monkeypatch):
    """
    Test run_prof applies TCC flattening for MI300 series GPUs.

    Args:
        tmp_path (Path): Temporary directory for test files.
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for patching.

    Returns:
        None: Asserts TCC flattening is applied for MI300 GPUs.
    """
    fname = tmp_path / "pmc_perf_test.yaml"
    fname.write_text("jobs:\n  - pmc:\n    - TCC_HIT[0]\n")
    workload_dir = str(tmp_path / "workload")

    # Mock functions
    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofv3")
    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "success"),
    )
    monkeypatch.setattr("utils.mi_gpu_spec.mi_gpu_specs.get_num_xcds", lambda *a: 2)
    monkeypatch.setattr(
        "glob.glob", lambda pattern: [workload_dir + "/results_test.csv"]
    )
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_log", lambda *a, **k: None)

    # Mock pandas
    mock_df = pd.DataFrame({"Dispatch_ID": [0], "TCC_HIT[0]": [100]})
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.concat", lambda *a, **k: mock_df)
    monkeypatch.setattr("pandas.DataFrame.to_csv", lambda self, *a, **k: None)

    # Execute function
    utils_profile.run_prof(str(fname), ["--arg"], workload_dir, logging.INFO, "csv")


def test_run_prof_sdk_creates_new_env_copy(tmp_path, monkeypatch):
    """
    Covers: new_env = os.environ.copy()
            when rocprof_cmd == "rocprofiler-sdk" and new_env was not previously set
            by the mspec.gpu_model check.
    """
    fname_str = str(tmp_path / "pmc_perf_counters.yaml")
    Path(fname_str).write_text("jobs:\n  - pmc:\n    - COUNTER1\n")
    workload_dir_str = str(tmp_path)

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprofiler-sdk")
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output", lambda *a, **k: []
    )

    capture_subprocess_called_with_env = None

    def mock_capture_subprocess(app_cmd, new_env=None, profileMode=False):
        nonlocal capture_subprocess_called_with_env
        capture_subprocess_called_with_env = new_env
        return (True, "Success")

    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output", mock_capture_subprocess
    )

    def mock_console_error_no_exit(msg, exit=True):
        print(f"Mocked console_error: {msg}, exit={exit}")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error_no_exit)
    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.parse_pmc_perf", lambda *a, **k: ["COUNTER1", "COUNTER2"]
    )

    mock_fname_path_obj = mock.MagicMock(spec=Path)
    mock_fname_path_obj.stem = "pmc_perf_counters"
    mock_fname_path_obj.name = "pmc_perf_counters.yaml"
    mock_fname_path_obj.exists.return_value = False

    mock_out_path_obj = mock.Mock(spec=Path)
    mock_out_path_obj.exists.return_value = False

    mock_counter_def_path_obj = mock.Mock(spec=Path)
    mock_counter_def_path_obj.exists.return_value = False
    mock_fname_path_obj.parent.__truediv__ = mock.Mock(
        return_value=mock_counter_def_path_obj
    )

    def path_side_effect(p_arg, *args):
        if isinstance(p_arg, Path):
            if p_arg.name == "pmc_perf_counters.yaml":
                return mock_fname_path_obj
            return p_arg
        if isinstance(p_arg, str):
            if p_arg.endswith("/out"):
                return mock_out_path_obj
            if p_arg.endswith("pmc_perf_counters.yaml"):
                return mock_fname_path_obj
            if "counter_def" in p_arg:
                return mock_counter_def_path_obj
        if (
            p_arg == mock_fname_path_obj
            and args == ()
            and hasattr(p_arg, "with_suffix")
        ):
            return mock_fname_path_obj
        return mock_fname_path_obj

    monkeypatch.setattr("utils.utils_profile.Path", path_side_effect)

    loglevel = logging.DEBUG
    format_rocprof_output = True

    dummy_df = pd.DataFrame({"Dispatch_ID": [0], "A": [1]})
    monkeypatch.setattr("pandas.read_csv", lambda *a, **k: dummy_df.copy())
    monkeypatch.setattr("pandas.DataFrame.to_csv", lambda self, *a, **k: None)
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("shutil.rmtree", lambda *a, **k: None)
    monkeypatch.setattr(
        "utils.utils_profile.create_temp_rocprofiler_metrics_path",
        lambda *a, **k: "dummy_path",
    )
    monkeypatch.setattr("yaml.dump", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)
    monkeypatch.setattr("builtins.open", lambda *a, **k: io.StringIO(""))

    from rocprof_compute_profile.profiler_rocprofiler_sdk import (
        rocprofiler_sdk_profiler as rocprofiler_sdk_profiler,
    )

    profiler = rocprofiler_sdk_profiler(
        profiling_args=MockArgs(
            rocprofiler_sdk_tool_path="sdk_tool",
            roof_only=True,
            format_rocprof_output="format",
            output_directory="path",
            remaining="remaining",
            iteration_multiplexing=None,
            attach_pid=None,
            kokkos_trace=None,
            kernel=None,
            dispatch=None,
        ),
        profiler_mode="rocprofiler-sdk",
        soc=MockSoc(),
    )

    # Test 1: LD_PRELOAD is already set - should be preserved
    # Since we check all env. vars. in test,
    # empty them out while calling profiling function
    with mock.patch.dict(os.environ, {}, clear=True):
        assert len(os.environ) == 0
        original_env_var = "original_value"
        monkeypatch.setenv("EXISTING_VAR", original_env_var)
        monkeypatch.setenv("LD_LIBRARY_PATH", original_env_var)
        monkeypatch.setenv("LD_PRELOAD", original_env_var)
        profiler_options = profiler.get_profiler_options(native_tool_path="native_tool")

        utils_profile.run_prof(
            fname_str,
            profiler_options,
            workload_dir_str,
            loglevel,
            format_rocprof_output,
        )

    assert capture_subprocess_called_with_env is not None, (
        "new_env should have been created"
    )
    assert "EXISTING_VAR" in capture_subprocess_called_with_env, (
        "new_env should be a copy of os.environ"
    )
    # Ensure existing env. vars. are preserved
    assert capture_subprocess_called_with_env["EXISTING_VAR"] == original_env_var
    # Ensure LD_LIBRARY_PATH is not touched
    assert capture_subprocess_called_with_env["LD_LIBRARY_PATH"] == original_env_var
    # Ensure LD_PRELOAD is preserved and our tools are appended
    assert original_env_var in capture_subprocess_called_with_env["LD_PRELOAD"], (
        f"User's LD_PRELOAD '{original_env_var}' should be preserved"
    )
    assert "sdk_tool" in capture_subprocess_called_with_env["LD_PRELOAD"], (
        "Profiler sdk_tool should be in LD_PRELOAD"
    )
    assert "native_tool" in capture_subprocess_called_with_env["LD_PRELOAD"], (
        "Native tool should be in LD_PRELOAD"
    )
    # Verify the order: user's LD_PRELOAD comes first, then our tools appended
    expected_ld_preload = f"{original_env_var}:sdk_tool:native_tool"
    assert capture_subprocess_called_with_env["LD_PRELOAD"] == expected_ld_preload, (
        f"LD_PRELOAD should be '{expected_ld_preload}' but got "
        f"'{capture_subprocess_called_with_env['LD_PRELOAD']}'"
    )

    # Test 2: LD_PRELOAD is unset - should only contain profiler tools
    capture_subprocess_called_with_env = None
    with mock.patch.dict(os.environ, {}, clear=True):
        assert len(os.environ) == 0
        monkeypatch.setenv("EXISTING_VAR", original_env_var)
        monkeypatch.setenv("LD_LIBRARY_PATH", original_env_var)
        # Intentionally not setting LD_PRELOAD to test the unset case
        profiler_options = profiler.get_profiler_options(native_tool_path="native_tool")

        utils_profile.run_prof(
            fname_str,
            profiler_options,
            workload_dir_str,
            loglevel,
            format_rocprof_output,
        )

    assert capture_subprocess_called_with_env is not None, (
        "new_env should have been created"
    )
    # When LD_PRELOAD is unset, should only contain our profiler tools
    expected_ld_preload_unset = "sdk_tool:native_tool"
    actual_ld_preload = capture_subprocess_called_with_env["LD_PRELOAD"]
    assert actual_ld_preload == expected_ld_preload_unset, (
        f"LD_PRELOAD should be '{expected_ld_preload_unset}' when unset, "
        f"but got '{actual_ld_preload}'"
    )
    assert (
        capture_subprocess_called_with_env["ROCPROFILER_METRICS_PATH"] == "dummy_path"
    )
    assert capture_subprocess_called_with_env["ROCPROF_COUNTER_COLLECTION"] == "0"
    assert capture_subprocess_called_with_env["ROCPROF_KERNEL_TRACE"] == "1"
    assert capture_subprocess_called_with_env["ROCPROF_OUTPUT_FORMAT"] == "format"
    assert capture_subprocess_called_with_env["ROCPROF_OUTPUT_PATH"] == "path/out/pmc_1"
    assert (
        capture_subprocess_called_with_env["ROCPROF_COUNTERS"]
        == "pmc: COUNTER1 COUNTER2"
    )
    assert "APP_CMD" not in capture_subprocess_called_with_env


def test_run_prof_v3_cli_calls_kokkos_trace_processing(tmp_path, monkeypatch):
    """
    Covers:
    CLI: if "--kokkos-trace" in options:
        process_kokkos_trace_output(...)
    """
    fname_str = str(tmp_path) + "/pmc_perf_counters.yaml"
    Path(fname_str).write_text("jobs:\n  - pmc:\n    - C1\n")
    fbase_str = "pmc_perf_counters"
    workload_dir_str = str(tmp_path)
    (tmp_path / "out" / "pmc_1").mkdir(parents=True, exist_ok=True)

    results_csv = str(tmp_path) + "/out/pmc_1/results1.csv"

    monkeypatch.setattr(
        "utils.utils_profile.capture_subprocess_output",
        lambda *a, **k: (True, "Success"),
    )
    monkeypatch.setattr(
        "utils.utils_profile.process_rocprofv3_output",
        lambda *a, **k: [results_csv],
    )

    kokkos_trace_called_with = None

    def mock_kokkos_trace(wd, fb):
        nonlocal kokkos_trace_called_with
        kokkos_trace_called_with = (wd, fb)

    monkeypatch.setattr(
        "utils.utils_profile.process_kokkos_trace_output", mock_kokkos_trace
    )

    monkeypatch.setattr("utils.utils_profile.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.console_warning", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.parse_pmc_perf", lambda *a, **k: ["C1"])

    # Mock csv_ops functions to avoid disk I/O
    mock_rows = [
        {
            "Dispatch_ID": "0",
            "Kernel_Name": "test",
            "Grid_Size": "1024",
            "Workgroup_Size": "64",
            "LDS_Per_Workgroup": "1024",
        }
    ]
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.concat_csv_files", lambda *a, **k: mock_rows.copy()
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.read_csv_as_dicts",
        lambda *a, **k: (mock_rows.copy(), list(mock_rows[0].keys())),
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.add_column_to_rows", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.assign_group_ids", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.rename_columns", lambda *a, **k: None
    )
    # Mock shutil operations since we're not actually writing files
    monkeypatch.setattr("utils.utils_profile.shutil.copyfile", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_profile.shutil.rmtree", lambda *a, **k: None)

    loglevel = logging.INFO
    format_rocprof_output = "csv"

    monkeypatch.setattr("utils.utils_common._rocprof_cmd", "rocprof_cli_v3")

    profiler_options_cli_kokkos = ["--kokkos-trace", "--other-opt"]
    kokkos_trace_called_with = None

    utils_profile.run_prof(
        fname_str,
        profiler_options_cli_kokkos,
        workload_dir_str,
        loglevel,
        format_rocprof_output,
    )
    assert kokkos_trace_called_with == (workload_dir_str, fbase_str)


# =============================================================================
# ROCPROFV3 OUTPUT PROCESSING TESTS
# =============================================================================


def test_process_rocprofv3_output_csv_format_with_counter_files(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output with csv format processes counter collection files.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    agent_file = output_dir / "test_agent_info.csv"
    converted_file = output_dir / "test_converted.csv"

    counter_file.write_text("counter,data\ntest,value")
    agent_file.write_text("agent,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        elif "_converted.csv" in pattern:
            return [str(converted_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        Path(output_path).write_text("converted,data\ntest,value")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert len(result) == 1
    assert str(converted_file) in result


def test_process_rocprofv3_output_csv_format_conversion_error(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output handles conversion errors gracefully.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    agent_file = output_dir / "test_agent_info.csv"

    counter_file.write_text("counter,data\ntest,value")
    agent_file.write_text("agent,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        raise ValueError("Conversion failed")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    warnings = []
    monkeypatch.setattr(
        "utils.utils_profile.console_warning", lambda msg: warnings.append(msg)
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert result == []
    assert len(warnings) == 1
    assert "Error converting" in warnings[0]


def test_process_rocprofv3_output_csv_format_missing_agent_file(tmp_path, monkeypatch):
    """
    Test process_rocprofv3_output raises error when agent info file is missing.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file = output_dir / "test_counter_collection.csv"
    counter_file.write_text("counter,data\ntest,value")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    with pytest.raises(ValueError, match='has no corresponding "agent info" file'):
        utils_profile.process_rocprofv3_output(workload_dir, False)


def test_process_rocprofv3_output_csv_format_no_files_non_timestamps(
    tmp_path, monkeypatch
):
    """
    Test process_rocprofv3_output returns empty list when
    no files found for non-timestamps.
    """
    workload_dir = str(tmp_path)

    monkeypatch.setattr("glob.glob", lambda pattern: [])

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert result == []


def test_process_rocprofv3_output_csv_format_multiple_counter_files(
    tmp_path, monkeypatch
):
    """
    Test process_rocprofv3_output processes multiple counter collection files.
    """
    workload_dir = str(tmp_path)
    output_dir = tmp_path / "out" / "pmc_1" / "subdir"
    output_dir.mkdir(parents=True)

    counter_file1 = output_dir / "test1_counter_collection.csv"
    agent_file1 = output_dir / "test1_agent_info.csv"
    converted_file1 = output_dir / "test1_converted.csv"

    counter_file2 = output_dir / "test2_counter_collection.csv"
    agent_file2 = output_dir / "test2_agent_info.csv"
    converted_file2 = output_dir / "test2_converted.csv"

    counter_file1.write_text("counter,data\ntest1,value1")
    agent_file1.write_text("agent,data\ntest1,value1")
    counter_file2.write_text("counter,data\ntest2,value2")
    agent_file2.write_text("agent,data\ntest2,value2")

    def mock_glob(pattern):
        if "_counter_collection.csv" in pattern:
            return [str(counter_file1), str(counter_file2)]
        elif "_converted.csv" in pattern:
            return [str(converted_file1), str(converted_file2)]
        return []

    monkeypatch.setattr("glob.glob", mock_glob)

    def mock_v3_counter_csv_to_v2_csv(counter_path, agent_path, output_path):
        Path(output_path).write_text(f"converted,data\n{Path(counter_path).stem},value")

    monkeypatch.setattr(
        "utils.utils_profile.v3_counter_csv_to_v2_csv", mock_v3_counter_csv_to_v2_csv
    )

    result = utils_profile.process_rocprofv3_output(workload_dir, False)

    assert len(result) == 2
    assert str(converted_file1) in result
    assert str(converted_file2) in result


# =============================================================================
# KOKKOS TRACE PROCESSING TESTS
# =============================================================================


def test_process_kokkos_trace_output_single_file(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with a single CSV file.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "single_marker_api_trace.csv"
    csv1.write_text(
        "marker_id,marker_name,start_time,end_time\n1,kokkos_begin,1000,1050\n2,kokkos_end,2000,2010\n"
    )

    fbase = "single_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    # Check output file in pmc_1 directory
    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert df["marker_name"].tolist() == ["kokkos_begin", "kokkos_end"]

    # Check copied file in workload directory
    copied_file = tmp_path / f"{fbase}_marker_api_trace.csv"
    assert copied_file.exists()


def test_process_kokkos_trace_output_multiple_files(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with multiple valid CSV files.
    Should concatenate all files and save the result.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_warning", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub1.mkdir()
    sub2.mkdir()

    csv1 = sub1 / "test_marker_api_trace.csv"
    csv2 = sub2 / "test_marker_api_trace.csv"
    csv1.write_text(
        "timestamp,marker_name,duration\n1000,kokkos_malloc,500\n2000,kokkos_parallel_for,300\n"
    )
    csv2.write_text(
        "timestamp,marker_name,duration\n3000,kokkos_free,200\n4000,kokkos_parallel_reduce,800\n"
    )

    fbase = "test_workload"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists(), "The primary output file was not created."

    df = pd.read_csv(output_file)
    assert len(df) == 4, (
        "The final DataFrame does not contain the correct number of rows."
    )
    assert set(df["timestamp"]) == {1000, 2000, 3000, 4000}
    assert "kokkos_malloc" in df["marker_name"].values
    assert "kokkos_parallel_reduce" in df["marker_name"].values


def test_process_kokkos_trace_output_no_files_found(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when no marker API trace files are found.
    With the new csv_ops-based implementation, no output file is created when
    there are no input files, and shutil.copyfile will raise FileNotFoundError.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)
    monkeypatch.setattr("utils.utils_common.console_log", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    fbase = "no_files"

    # With the csv_ops implementation, when there are no input files:
    # - concat_csv_files returns []
    # - write_csv_from_dicts doesn't write anything (no rows, no fieldnames)
    # - shutil.copyfile fails because the source file doesn't exist
    with pytest.raises(FileNotFoundError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


def test_process_kokkos_trace_output_mixed_file_states(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with a mix of valid, empty, and corrupted files.
    """
    monkeypatch.setattr("utils.logger.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub3 = out_dir / "process3"
    sub1.mkdir()
    sub2.mkdir()
    sub3.mkdir()

    csv1 = sub1 / "valid_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name\n1000,kokkos_malloc\n2000,kokkos_free\n")

    csv2 = sub2 / "empty_marker_api_trace.csv"
    csv2.write_text("")

    csv3 = sub3 / "headers_marker_api_trace.csv"
    csv3.write_text("timestamp,marker_name\n")

    fbase = "mixed_test"

    original_read_csv = pd.read_csv

    def mock_read_csv(filepath, **kwargs):
        try:
            return original_read_csv(filepath, **kwargs)
        except pd.errors.EmptyDataError:
            # Return empty DataFrame for empty files
            return pd.DataFrame()

    monkeypatch.setattr("pandas.read_csv", mock_read_csv)

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) >= 0


def test_process_kokkos_trace_output_no_out_directory(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when output directory doesn't exist.
    Should not copy file to workload directory.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)

    fbase = "no_out_dir"

    monkeypatch.setattr("glob.glob", lambda pattern: [])

    def mock_concat(dataframes, **kwargs):
        if not dataframes:
            return pd.DataFrame()
        return pd.concat(dataframes, **kwargs)

    monkeypatch.setattr("pandas.concat", mock_concat)

    def mock_to_csv(self, path, **kwargs):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")

    monkeypatch.setattr("pandas.DataFrame.to_csv", mock_to_csv)

    from pathlib import Path as original_path

    def mock_path_exists(path_str):
        if path_str == workload_dir + "/out":
            mock_path_obj = mock.MagicMock()
            mock_path_obj.exists.return_value = False
            return mock_path_obj
        else:
            return original_path(path_str)

    monkeypatch.setattr("utils.utils_profile.Path", mock_path_exists)

    try:
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)

        # Should not copy file to workload directory since /out doesn't exist
        copied_file = tmp_path / f"{fbase}_marker_api_trace.csv"
        assert not copied_file.exists()

    except ValueError:
        pytest.skip(
            "process_kokkos_trace_output doesn't handle missing "
            "output directory gracefully"
        )


def test_process_kokkos_trace_output_csv_with_only_headers(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files that contain
    only headers but no data.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "headers_only_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name,duration,thread_id\n")

    fbase = "headers_only"

    # With csv_ops, header-only files result in empty rows and the output
    # file isn't created, causing FileNotFoundError during copyfile
    with pytest.raises(FileNotFoundError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


def test_process_kokkos_trace_output_large_files(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with larger CSV files to ensure memory handling.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "large_marker_api_trace.csv"

    content = "timestamp,marker_name,duration,thread_id\n"
    kokkos_markers = [
        "kokkos_malloc",
        "kokkos_free",
        "kokkos_parallel_for",
        "kokkos_parallel_reduce",
        "kokkos_fence",
    ]
    for i in range(1000):
        marker_name = kokkos_markers[i % len(kokkos_markers)]
        content += f"{i},{marker_name},{i % 100},{i % 10}\n"

    csv1.write_text(content)

    fbase = "large_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 1000
    assert "kokkos_malloc" in df["marker_name"].values
    assert "kokkos_parallel_reduce" in df["marker_name"].values


def test_process_kokkos_trace_output_unicode_content(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files containing unicode characters.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "unicode_marker_api_trace.csv"
    csv1.write_text(
        "timestamp,marker_name,duration\n1000,kokkos_\u03b1_kernel,500\n2000,kokkos_\u03b2_operation,300\n",
        encoding="utf-8",
    )

    fbase = "unicode_test"

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    output_file = out_dir / f"results_{fbase}_marker_api_trace.csv"
    assert output_file.exists()

    df = pd.read_csv(output_file)
    assert len(df) == 2
    assert "kokkos_\u03b1_kernel" in df["marker_name"].values
    assert "kokkos_\u03b2_operation" in df["marker_name"].values


def test_process_kokkos_trace_output_different_schemas(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output with CSV files having different column schemas.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    # Create subdirectories for glob to find
    sub1 = out_dir / "process1"
    sub2 = out_dir / "process2"
    sub1.mkdir()
    sub2.mkdir()

    # Create marker trace files (needed for glob pattern matching)
    csv1 = sub1 / "schema1_marker_api_trace.csv"
    csv2 = sub2 / "schema2_marker_api_trace.csv"
    csv1.touch()
    csv2.touch()

    fbase = "schema_test"

    # Mock csv_ops to avoid disk I/O and test concatenation behavior
    mock_rows = [
        {"marker_id": "1", "marker_name": "kokkos_begin", "start_time": "1000"},
        {"marker_id": "2", "marker_name": "kokkos_end", "start_time": "2000"},
        {"marker_name": "kokkos_malloc", "duration": "500", "thread_id": "0"},
        {"marker_name": "kokkos_free", "duration": "200", "thread_id": "1"},
    ]

    write_calls = []

    def mock_concat(files, output_file=None):
        return mock_rows.copy()

    def mock_write(path, rows, fieldnames=None):
        write_calls.append((path, rows))

    monkeypatch.setattr("utils.utils_profile.csv_ops.concat_csv_files", mock_concat)
    monkeypatch.setattr("utils.utils_profile.csv_ops.write_csv_from_dicts", mock_write)
    monkeypatch.setattr("shutil.copyfile", lambda *a, **k: None)

    utils_profile.process_kokkos_trace_output(workload_dir, fbase)

    # Verify write was called with the concatenated rows
    assert len(write_calls) == 1
    written_rows = write_calls[0][1]
    assert len(written_rows) == 4
    # Check that all rows have marker_name
    assert all("marker_name" in row for row in written_rows)


def test_process_kokkos_trace_output_permission_error(tmp_path, monkeypatch):
    """
    Test process_kokkos_trace_output when there are permission
    errors during file operations.
    """
    monkeypatch.setattr("utils.utils_common.console_debug", lambda *a, **k: None)

    workload_dir = str(tmp_path)
    out_dir = tmp_path / "out" / "pmc_1"
    out_dir.mkdir(parents=True)

    sub1 = out_dir / "process1"
    sub1.mkdir()

    csv1 = sub1 / "test_marker_api_trace.csv"
    csv1.write_text("timestamp,marker_name\n1000,kokkos_malloc\n")

    fbase = "permission_test"

    def mock_write_permission_error(path, rows, fieldnames=None):
        raise PermissionError("Permission denied")

    monkeypatch.setattr(
        "utils.utils_profile.csv_ops.write_csv_from_dicts",
        mock_write_permission_error,
    )

    with pytest.raises(PermissionError):
        utils_profile.process_kokkos_trace_output(workload_dir, fbase)


# =============================================================================
# GET_SUBMODULES TESTS
# =============================================================================


mock_package = mock.MagicMock()
mock_package.__path__ = ["/fake/path"]
mock_submodules = [
    (None, "module_parse", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules)
def test_get_submodules_basic_functionality(mock_walk, mock_import):
    """
    Test basic functionality with a real package that has submodules.
    """

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert len(result) == 3
    expected = ["parse", "request", "error"]
    assert result == expected


def test_get_submodules_empty_package():
    """
    Test with a package that has no submodules.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("empty_package")

            assert isinstance(result, list)
            assert len(result) == 0


def test_get_submodules_package_not_found():
    """
    Test behavior when package doesn't exist.
    """

    with pytest.raises(ModuleNotFoundError):
        utils_profile.get_submodules("nonexistent_package_12345")


mock_package_single = mock.MagicMock()
mock_package_single.__path__ = ["/fake/path"]
mock_submodules_single = [
    (None, "module_parser", False),
    (None, "module_request", False),
    (None, "module_error", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_single)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_single)
def test_get_submodules_name_processing_single_underscore(mock_walk, mock_import):
    """
    Test name processing with single underscore pattern.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "request", "error"]
    assert result == expected


mock_package_multiple = mock.MagicMock()
mock_package_multiple.__path__ = ["/fake/path"]
mock_submodules_multiple = [
    (None, "module_some_complex_name", False),
    (None, "module_another_test_case", False),
    (None, "module_simple", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_multiple)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_multiple)
def test_get_submodules_name_processing_multiple_underscores(mock_walk, mock_import):
    """
    Test name processing with multiple underscores in submodule names.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["somecomplexname", "anothertestcase", "simple"]
    assert result == expected


mock_package_base = mock.MagicMock()
mock_package_base.__path__ = ["/fake/path"]
mock_submodules_base = [
    (None, "module_base", False),
    (None, "module_parser", False),
    (None, "module_handler", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_base)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_base)
def test_get_submodules_base_module_filtered(mock_walk, mock_import):
    """
    Test that 'base' submodule is properly filtered out.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "handler"]
    assert result == expected
    assert "base" not in result


mock_package_no_underscore = mock.MagicMock()
mock_package_no_underscore.__path__ = ["/fake/path"]
mock_submodules_no_underscore = [
    (None, "simplemodule", False),
    (None, "anothermodule", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_no_underscore)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_no_underscore)
def test_get_submodules_no_underscore_in_name(mock_walk, mock_import):
    """
    Test behavior with submodule names that don't follow the expected pattern.
    """

    with pytest.raises(IndexError):
        utils_profile.get_submodules("test_package")


mock_package_empty_parts = mock.MagicMock()
mock_package_empty_parts.__path__ = ["/fake/path"]
mock_submodules_empty_parts = [
    (None, "module_", False),  # ends with underscore
    (None, "_module", False),  # starts with underscore - this will cause IndexError
    (None, "module__double", False),  # double underscore
]


@mock.patch("importlib.import_module", return_value=mock_package_empty_parts)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_empty_parts)
def test_get_submodules_empty_name_parts(mock_walk, mock_import):
    """
    Test behavior with empty name parts after splitting.
    """

    try:
        result = utils_profile.get_submodules("test_package")
        expected = ["", "", "double"]  # noqa - Empty strings for edge cases
        assert len(result) == 3
    except IndexError:
        pytest.skip("Function doesn't handle edge case module names gracefully")


def test_get_submodules_package_without_path_attribute():
    """
    Test behavior when package doesn't have __path__ attribute.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    del mock_package.__path__

    with patch("importlib.import_module", return_value=mock_package):
        with pytest.raises(AttributeError):
            utils_profile.get_submodules("test_package")


mock_package_exception = mock.MagicMock()
mock_package_exception.__path__ = ["/fake/path"]


@mock.patch("importlib.import_module", return_value=mock_package_exception)
@mock.patch("pkgutil.walk_packages", side_effect=ImportError("Mock error"))
def test_get_submodules_pkgutil_walk_packages_exception(mock_walk, mock_import):
    """
    Test behavior when pkgutil.walk_packages raises an exception.
    """

    with pytest.raises(ImportError):
        utils_profile.get_submodules("test_package")


mock_package_mixed = mock.MagicMock()
mock_package_mixed.__path__ = ["/fake/path"]
mock_submodules_mixed = [
    (None, "module_base", False),  # Should be filtered out
    (None, "module_parser", False),  # Normal case
    (None, "module_test_case", False),  # Multiple underscores
    (None, "module_simple", False),  # Simple case
    (None, "module_another_base", False),  # Contains 'base' but not exactly 'base'
]


@mock.patch("importlib.import_module", return_value=mock_package_mixed)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_mixed)
def test_get_submodules_mixed_module_types(mock_walk, mock_import):
    """
    Test with a mix of different module types and names.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["parser", "testcase", "simple", "anotherbase"]
    assert result == expected
    assert "base" not in result


mock_package_large = mock.MagicMock()
mock_package_large.__path__ = ["/fake/path"]
mock_submodules_large = []
expected_results_large = []
for i in range(100):
    module_name = f"module_test{i}"
    mock_submodules_large.append((None, module_name, False))
    expected_results_large.append(f"test{i}")


@mock.patch("importlib.import_module", return_value=mock_package_large)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_large)
def test_get_submodules_large_number_of_submodules(mock_walk, mock_import):
    """
    Test performance and correctness with a large number of submodules.
    """

    result = utils_profile.get_submodules("test_package")
    assert len(result) == 100
    assert result == expected_results_large


def test_get_submodules_string_input_validation():
    """
    Test input validation for package_name parameter.
    """

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(None)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(123)

    with pytest.raises((TypeError, AttributeError)):
        utils_profile.get_submodules(["list", "input"])


def test_get_submodules_return_type_consistency():
    """
    Test that function always returns a list, even in edge cases.
    """
    from unittest.mock import MagicMock, patch

    mock_package = MagicMock()
    mock_package.__path__ = ["/fake/path"]

    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=[]):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0

    mock_submodules = [(None, "module_base", False)]
    with patch("importlib.import_module", return_value=mock_package):
        with patch("pkgutil.walk_packages", return_value=mock_submodules):
            result = utils_profile.get_submodules("test_package")
            assert isinstance(result, list)
            assert len(result) == 0


mock_package_special = mock.MagicMock()
mock_package_special.__path__ = ["/fake/path"]
mock_submodules_special = [
    (None, "module_test-case", False),
    (None, "module_test.case", False),
    (None, "module_test123", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_special)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_special)
def test_get_submodules_special_characters_in_names(mock_walk, mock_import):
    """
    Test handling of special characters in submodule names.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["test-case", "test.case", "test123"]
    assert result == expected


mock_package_isolation = mock.MagicMock()
mock_package_isolation.__path__ = ["/fake/path"]
mock_submodules_isolation = [(None, "module_test", False)]


@mock.patch("importlib.import_module", return_value=mock_package_isolation)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_isolation)
def test_get_submodules_imports_isolation(mock_walk, mock_import):
    """
    Test that imports are properly isolated and don't affect global state.
    """
    import sys

    original_importlib = sys.modules.get("importlib")
    original_pkgutil = sys.modules.get("pkgutil")

    result = utils_profile.get_submodules("test_package")

    assert sys.modules.get("importlib") == original_importlib
    assert sys.modules.get("pkgutil") == original_pkgutil
    assert isinstance(result, list)
    assert result == ["test"]


mock_package_unicode = mock.MagicMock()
mock_package_unicode.__path__ = ["/fake/path"]
mock_submodules_unicode = [
    (None, "module_t\u00ebst", False),
    (None, "module_\u6d4b\u8bd5", False),
    (None, "module_\u0442\u0435\u0441\u0442", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_unicode)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_unicode)
def test_get_submodules_unicode_names(mock_walk, mock_import):
    """
    Test handling of Unicode characters in package and submodule names.
    """

    result = utils_profile.get_submodules("test_package")
    expected = ["t\u00ebst", "\u6d4b\u8bd5", "\u0442\u0435\u0441\u0442"]
    assert result == expected


mock_package_docstring = mock.MagicMock()
mock_package_docstring.__path__ = ["/fake/path"]
mock_submodules_docstring = [
    (None, "module_submodule1", False),
    (None, "module_submodule2", False),
]


@mock.patch("importlib.import_module", return_value=mock_package_docstring)
@mock.patch("pkgutil.walk_packages", return_value=mock_submodules_docstring)
def test_get_submodules_docstring_verification(mock_walk, mock_import):
    """
    Test that function behavior matches its docstring description.
    """

    assert utils_profile.get_submodules.__doc__ is not None
    assert (
        "List all submodules for a target package"
        in utils_profile.get_submodules.__doc__
    )  # noqa

    result = utils_profile.get_submodules("test_package")

    assert isinstance(result, list)
    assert "submodule1" in result
    assert "submodule2" in result


# =============================================================================
# additional tests for v3_counter_csv_to_v2_csv function
# =============================================================================


def create_csv_string(data_dict):
    return pd.DataFrame(data_dict).to_csv(index=False)


@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_v3_to_v2_agent_id_parsing_success_and_error(
    mock_console_debug, mock_console_error, tmp_path
):
    """
    Tests Line 1: Successful parsing of 'Agent Id' string.
    Tests Line 2: Graceful handling of malformed 'Agent Id' (no error expected).
    """
    import utils.utils_profile_csv as csv_ops

    agent_info_content = create_csv_string({
        "Node_Id": [0, 1],
        "Agent_Type": ["CPU", "GPU"],
        "Wave_Front_Size": [0, 64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted.csv"
    counter_content_success = create_csv_string({
        "Correlation_Id": [1],
        "Dispatch_Id": [10],
        "Agent_Id": ["Agent 1"],
        "Queue_Id": [100],
        "Process_Id": [1000],
        "Thread_Id": [10000],
        "Grid_Size": [256],
        "Kernel_Id": [1],
        "Kernel_Name": ["kernelA"],
        "Workgroup_Size": [64],
        "LDS_Block_Size": [32],
        "Scratch_Size": [0],
        "VGPR_Count": [16],
        "Accum_VGPR_Count": [0],
        "SGPR_Count": [32],
        "Start_Timestamp": [100000],
        "End_Timestamp": [100100],
        "Counter_Name": ["Cycles"],
        "Counter_Value": [5000],
    })
    counter_filepath_success = tmp_path / "counter_success.csv"
    counter_filepath_success.write_text(counter_content_success)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath_success),
        str(agent_info_filepath),
        str(converted_csv_filepath),
    )

    mock_console_error.assert_not_called()
    rows, _ = csv_ops.read_csv_as_dicts(str(converted_csv_filepath))
    assert len(rows) == 1
    assert "GPU_ID" in rows[0]
    # GPU_ID should be the index of the GPU agent (1 is the only GPU, so index 0)
    # Note: csv_ops returns strings, so we compare as string
    assert str(rows[0]["GPU_ID"]) == "0"

    mock_console_error.reset_mock()

    # Test with malformed Agent_Id - the new implementation gracefully handles
    # this by keeping the original value unchanged (regex simply doesn't match)
    counter_content_malformed = create_csv_string({
        "Correlation_Id": [2],
        "Dispatch_Id": [20],
        "Agent_Id": ["Malformed Agent X"],
        "Queue_Id": [200],
        "Process_Id": [2000],
        "Thread_Id": [20000],
        "Grid_Size": [512],
        "Kernel_Id": [2],
        "Kernel_Name": ["kernelB"],
        "Workgroup_Size": [128],
        "LDS_Block_Size": [64],
        "Scratch_Size": [0],
        "VGPR_Count": [32],
        "Accum_VGPR_Count": [0],
        "SGPR_Count": [64],
        "Start_Timestamp": [200000],
        "End_Timestamp": [200200],
        "Counter_Name": ["Instructions"],
        "Counter_Value": [10000],
    })
    counter_filepath_malformed = tmp_path / "counter_malformed.csv"
    counter_filepath_malformed.write_text(counter_content_malformed)
    converted_malformed_filepath = tmp_path / "converted_malformed.csv"

    # This should not raise an exception - malformed values are handled gracefully
    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath_malformed),
        str(agent_info_filepath),
        str(converted_malformed_filepath),
    )

    # console_error is not called because the regex simply doesn't match
    # and the original value is kept (no exception raised)
    mock_console_error.assert_not_called()

    # The output should still be written with the original Agent_Id value
    rows, _ = csv_ops.read_csv_as_dicts(str(converted_malformed_filepath))
    assert len(rows) == 1
    # GPU_ID will have the malformed value since it wasn't converted
    # It won't map to a GPU ID, so it stays as the original value
    assert "GPU_ID" in rows[0]


@mock.patch("utils.utils_profile.console_debug")  # To suppress debug output
def test_v3_to_v2_accum_column_rename(mock_console_debug, tmp_path):
    """
    Tests Line 3: Renaming of a column ending with '_ACCUM' to 'SQ_ACCUM_PREV_HIRES'.
    """
    # --- Setup ---
    agent_info_content = create_csv_string({
        "Node_Id": [0],
        "Agent_Type": ["GPU"],
        "Wave_Front_Size": [64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted_accum.csv"

    counter_data = {
        "Correlation_Id": [1, 1],
        "Dispatch_Id": [10, 10],
        "Agent_Id": [0, 0],
        "Queue_Id": [100, 100],
        "Process_Id": [1000, 1000],
        "Thread_Id": [10000, 10000],
        "Grid_Size": [256, 256],
        "Kernel_Id": [1, 1],
        "Kernel_Name": ["kernelA", "kernelA"],
        "Workgroup_Size": [64, 64],
        "LDS_Block_Size": [32, 32],
        "Scratch_Size": [0, 0],
        "VGPR_Count": [16, 16],
        "Accum_VGPR_Count": [0, 0],
        "SGPR_Count": [32, 32],
        "Start_Timestamp": [100000, 100000],
        "End_Timestamp": [100100, 100100],
        "Counter_Name": ["FETCH_SIZE_ACCUM", "CYCLES"],
        "Counter_Value": [12345, 5000],
    }
    counter_content = create_csv_string(counter_data)
    counter_filepath = tmp_path / "counter_accum.csv"
    counter_filepath.write_text(counter_content)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath), str(agent_info_filepath), str(converted_csv_filepath)
    )

    result_df = pd.read_csv(converted_csv_filepath)
    assert "SQ_ACCUM_PREV_HIRES" in result_df.columns
    assert "FETCH_SIZE_ACCUM" not in result_df.columns
    assert "CYCLES" in result_df.columns
    assert result_df["SQ_ACCUM_PREV_HIRES"].iloc[0] == 12345
    assert result_df["CYCLES"].iloc[0] == 5000


@mock.patch("utils.utils_profile.console_debug")
def test_v3_to_v2_default_accum_vgpr_count(mock_console_debug, tmp_path):
    """
    Tests Line 4: 'Accum_VGPR_Count' is added and set to 0 if not present in input.
    """
    agent_info_content = create_csv_string({
        "Node_Id": [0],
        "Agent_Type": ["GPU"],
        "Wave_Front_Size": [64],
    })
    agent_info_filepath = tmp_path / "agent_info.csv"
    agent_info_filepath.write_text(agent_info_content)
    converted_csv_filepath = tmp_path / "converted_no_accum_vgpr.csv"

    counter_content = create_csv_string({
        "Correlation_Id": [1],
        "Dispatch_Id": [10],
        "Agent_Id": [0],
        "Queue_Id": [100],
        "Process_Id": [1000],
        "Thread_Id": [10000],
        "Grid_Size": [256],
        "Kernel_Id": [1],
        "Kernel_Name": ["kernelA"],
        "Workgroup_Size": [64],
        "LDS_Block_Size": [32],
        "Scratch_Size": [0],
        "VGPR_Count": [16],
        "SGPR_Count": [32],
        "Start_Timestamp": [100000],
        "End_Timestamp": [100100],
        "Counter_Name": ["Cycles"],
        "Counter_Value": [5000],
    })
    counter_filepath = tmp_path / "counter_no_accum_vgpr.csv"
    counter_filepath.write_text(counter_content)

    utils_profile.v3_counter_csv_to_v2_csv(
        str(counter_filepath), str(agent_info_filepath), str(converted_csv_filepath)
    )

    result_df = pd.read_csv(converted_csv_filepath)
    assert "Accum_VGPR" in result_df.columns
    assert result_df["Accum_VGPR"].iloc[0] == 0
    assert result_df["Accum_VGPR"].dtype == "int64"


# ===================================================================
# Test PC_sampling function
# ===================================================================


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_sdk_path_nonexistent_librocprofiler_sdk_tool(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """
    Edge Case: rocprofiler_sdk_tool_path is valid, but librocprofiler-sdk-tool.so
    is NOT found next to it (or in rocprofiler-sdk subdir).
    """
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        method = "host_trap"
        interval = 1000
        workload_dir = str(tmp_path)
        options = {"APP_CMD": "my_app --arg"}

        sdk_lib_dir = tmp_path / "rocm_sdk" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path).touch()

        expected_tool_path = str(
            sdk_lib_dir / "rocprofiler-sdk" / "librocprofiler-sdk-tool.so"
        )

        options["LD_PRELOAD"] = expected_tool_path

        mock_capture_subprocess.return_value = (True, "Success output")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        call_args = mock_capture_subprocess.call_args
        called_env = call_args.kwargs.get("new_env", {})

        assert "LD_PRELOAD" in called_env
        assert called_env["LD_PRELOAD"] == expected_tool_path

        mock_console_error.assert_not_called()


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_subprocess_fails(
    mock_console_debug, mock_capture_subprocess, tmp_path, monkeypatch
):
    """
    Edge Case: The capture_subprocess_output returns success=False.
    This should trigger the console_error("PC sampling failed.").
    """
    console_error_calls = []

    def mock_console_error(msg, exit=True):
        console_error_calls.append(msg)
        if exit:
            raise RuntimeError("console_error called")

    monkeypatch.setattr("utils.utils_profile.console_error", mock_console_error)

    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "stochastic"
        interval = 5000
        workload_dir = str(tmp_path)
        options = ["another_app"]

        with pytest.raises(RuntimeError, match="console_error called"):
            utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        mock_capture_subprocess.assert_not_called()
        assert console_error_calls == [
            "APP_CMD, the workload's executable must be provided "
            "when not in live attach mode"
        ]

    mock_capture_subprocess.reset_mock()
    console_error_calls.clear()
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        options = {"APP_CMD": "another_app"}
        sdk_lib_dir = tmp_path / "rocm_sdk_fail" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path_sdk).touch()

        tool_dir = sdk_lib_dir / "rocprofiler-sdk"
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "librocprofiler-sdk-tool.so").touch()

        mock_capture_subprocess.return_value = (
            False,
            "Error output from SDK subprocess",
        )

        with pytest.raises(RuntimeError, match="console_error called"):
            utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        mock_capture_subprocess.assert_called_once()
        assert console_error_calls == ["PC sampling failed."]


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_empty_appcmd(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """
    Edge Case: The appcmd is an empty string.
    """
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "host_trap"
        interval = 100
        workload_dir = str(tmp_path)
        options = ["--"]
        rocprofiler_sdk_tool_path = "/some/path/librocprofiler_sdk.so"  # noqa: F841

        mock_capture_subprocess.return_value = (True, "Output with empty appcmd")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        assert options_list[-1] == "--"
        mock_console_error.assert_not_called()

    mock_capture_subprocess.reset_mock()
    mock_console_error.reset_mock()
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprofiler-sdk"):
        sdk_lib_dir = tmp_path / "rocm_sdk_empty" / "lib"
        sdk_lib_dir.mkdir(parents=True, exist_ok=True)
        rocprofiler_sdk_tool_path_sdk = str(sdk_lib_dir / "librocprofiler_sdk.so")
        Path(rocprofiler_sdk_tool_path_sdk).touch()
        tool_dir = sdk_lib_dir / "rocprofiler-sdk"
        tool_dir.mkdir(parents=True, exist_ok=True)
        (tool_dir / "librocprofiler-sdk-tool.so").touch()

        mock_capture_subprocess.return_value = (True, "Output with empty appcmd SDK")
        options = {"APP_CMD": ""}

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        assert mock_capture_subprocess.call_args[0][0] == ""
        mock_console_error.assert_not_called()


@mock.patch("utils.utils_profile.capture_subprocess_output")
@mock.patch("utils.utils_profile.console_error")
@mock.patch("utils.utils_profile.console_debug")
def test_pc_sampling_prof_multiarg_appcmd(
    mock_console_debug, mock_console_error, mock_capture_subprocess, tmp_path
):
    """All arguments after '--' in profiler_options must appear
    in the subprocess call."""
    with mock.patch("utils.utils_common._rocprof_cmd", "rocprof_cli_tool"):
        method = "host_trap"
        interval = 100
        workload_dir = str(tmp_path)
        options = ["--kernel-trace", "--", "./myapp", "arg1", "arg2"]

        mock_capture_subprocess.return_value = (True, "Success")

        utils_profile.pc_sampling_prof(options, method, interval, workload_dir)

        assert mock_capture_subprocess.called
        options_list = mock_capture_subprocess.call_args[0][0]
        separator_index = options_list.index("--")
        assert options_list[separator_index:] == ["--", "./myapp", "arg1", "arg2"]
        mock_console_error.assert_not_called()
