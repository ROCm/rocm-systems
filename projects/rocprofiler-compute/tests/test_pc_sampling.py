# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path
from types import SimpleNamespace

import common
import pandas as pd
import pytest

from utils.parser import load_pc_sampling_data
from utils.pc_sampling_analysis import load_pc_sample_records

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1


PC_SAMPLING_HOST_TRAP_FILES = sorted([
    "ps_file_results.json",
    "sysinfo.csv",
])

PC_SAMPLING_STOCHASTIC_FILES = sorted([
    "ps_file_results.json",
    "sysinfo.csv",
])


def _assert_pc_sampling_files(file_dict, expected):
    """Assert the PC sampling output file-set, matching the native collector's
    ``<pid>_code_obj_info.json``.
    """
    keys = list(file_dict.keys())
    code_obj = [k for k in keys if k.endswith("_code_obj_info.json")]
    assert len(code_obj) == 1, (
        f"expected exactly one *_code_obj_info.json, got {code_obj}"
    )
    remaining = sorted(k for k in keys if k not in code_obj)
    assert remaining == sorted(expected)


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return "Given PC sampling configuration is not supported" in output


def _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir):
    if is_pc_sampling_not_supported(f"{stdout}\n{stderr}"):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict, PC_SAMPLING_HOST_TRAP_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    common.require_pc_sampling_gpu(is_stochastic=True)
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict, PC_SAMPLING_STOCHASTIC_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" not in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_warning_pc_sampling_with_counters(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a multi-rank warning is printed when running with --block 21
    and another block (PC sampling with counters mode requires multiple passes)
    with multi-rank.
    """
    common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_profile_then_analyze(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    End-to-end: profile with PC sampling (host_trap), then
    run analysis on the profiling output.
    """
    common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict, PC_SAMPLING_HOST_TRAP_FILES)

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out

    workload_path = Path(workload_dir)

    kernel_top_csv = workload_path / "pmc_kernel_top.csv"
    assert kernel_top_csv.exists()
    kernel_top_header = kernel_top_csv.read_text().splitlines()[0]
    assert "Kernel_Name" in kernel_top_header
    assert "Count" in kernel_top_header
    assert "Percent" in kernel_top_header

    dispatch_info_csv = workload_path / "pmc_dispatch_info.csv"
    assert dispatch_info_csv.exists()
    dispatch_info_header = dispatch_info_csv.read_text().splitlines()[0]
    assert "Dispatch_ID" in dispatch_info_header
    assert "Kernel_Name" in dispatch_info_header
    assert "GPU_ID" in dispatch_info_header

    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--block",
            "21",
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "0.1 Top Kernels" in captured.out
    assert "0.2 Dispatch List" in captured.out
    assert "21. PC Sampling" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_with_sol_block(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
    monkeypatch,
):
    """
    PC sampling with counter collection (--block 21 2): profiling produces the
    expected artifacts and analyze renders both counter and PC sampling panels.
    """
    common.require_pc_sampling_gpu()
    monkeypatch.setenv("ROCPROF", "rocprofiler-sdk")

    options = [
        "--experimental",
        "--pc-sampling",
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
    ]

    workload_dir = common.get_output_dir()

    code, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=False,
        capture_output=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = common.check_csv_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict, PC_SAMPLING_HOST_TRAP_FILES)

    assert common.check_file_pattern("- '21'", f"{workload_dir}/profiling_config.yaml")
    assert common.check_file_pattern("- '2'", f"{workload_dir}/profiling_config.yaml")

    # Analyze with a single kernel so the detailed PC sampling table renders.
    code = binary_handler_analyze_rocprof_compute(
        [
            "analyze",
            "--path",
            workload_dir,
            "--kernel",
            "0",
        ],
    )
    assert code == 0

    captured = capsys.readouterr()
    assert "2.1 System Speed-of-Light" in captured.out
    assert "21. PC Sampling" in captured.out
    # The "instruction" column header only renders when the table has rows.
    assert "instruction" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)


def _kernel_top_workload() -> SimpleNamespace:
    """Workload stub with dfs[1] populated for load_pc_sampling_data tests."""
    return SimpleNamespace(
        filter_kernel_ids=[],
        dfs={
            1: pd.DataFrame({
                "Kernel_Name": ["kernel_a", "kernel_b", "kernel_c"],
                "Count": [2, 1, 1],
                "Sum(ns)": [900, 800, 200],
            }),
        },
    )


def test_load_pc_sampling_data_missing_or_empty_sources_return_empty() -> None:
    """Absent tool data and empty buffer records both yield empty frames."""
    workload = SimpleNamespace(filter_kernel_ids=[])

    assert load_pc_sampling_data(workload, "none", "count", None).empty
    assert load_pc_sampling_data(workload, "missing", "count", None).empty

    workload.filter_kernel_ids = [0, 1, 2]
    assert load_pc_sampling_data(workload, "test", "count", None).empty

    empty_records = load_pc_sample_records({
        "buffer_records": {
            "pc_sample_stochastic": [],
            "pc_sample_host_trap": [],
            "kernel_dispatch": [],
        },
    })
    assert empty_records.empty


def test_load_pc_sampling_data_out_of_bounds_kernel_warns(monkeypatch) -> None:
    """An out-of-bounds kernel index warns and returns empty."""
    mock_warning = common.patch_console(monkeypatch, "utils.parser", "warning")[
        "warning"
    ]
    workload = _kernel_top_workload()
    tool_data = {
        "buffer_records": {"pc_sample_stochastic": [{}], "pc_sample_host_trap": []}
    }

    workload.filter_kernel_ids = [99]
    result = load_pc_sampling_data(workload, "test", "count", tool_data)

    mock_warning.assert_called()
    call_args_str = str(mock_warning.call_args)
    assert "out of bounds" in call_args_str or "99" in call_args_str
    assert result.empty


def test_load_pc_sampling_data_single_kernel_uses_workload_dfs(monkeypatch) -> None:
    """A single-kernel filter reads the kernel name from workload.dfs[1]."""
    per_kernel_calls = []

    def record_per_kernel(*args, **kwargs):
        per_kernel_calls.append((args, kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(
        "utils.parser.load_pc_sampling_data_per_kernel", record_per_kernel
    )
    workload = _kernel_top_workload()
    tool_data = {
        "buffer_records": {"pc_sample_stochastic": [{}], "pc_sample_host_trap": []}
    }

    workload.filter_kernel_ids = [1]
    load_pc_sampling_data(workload, "test", "count", tool_data)

    if per_kernel_calls:
        assert "kernel_b" in str(per_kernel_calls[0])
