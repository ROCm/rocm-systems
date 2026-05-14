# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

import os
from pathlib import Path

import common
import pytest

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1

_, soc = common.gpu_soc()

if soc is None:
    pytest.skip("GPU not supported", allow_module_level=True)

os.environ["ROCPROF"] = "rocprofiler-sdk"

PC_SAMPLING_HOST_TRAP_FILES = sorted([
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_host_trap.csv",
    "ps_file_results.json",
    "sysinfo.csv",
])

PC_SAMPLING_STOCHASTIC_FILES = sorted([
    "ps_file_agent_info.csv",
    "ps_file_kernel_trace.csv",
    "ps_file_pc_sampling_stochastic.csv",
    "ps_file_results.json",
    "sysinfo.csv",
])


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return "Given PC sampling configuration is not supported" in output


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    common.skip_unsupported_pc_sampling_soc()

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    common.skip_unsupported_pc_sampling_soc(is_stochastic=True)

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
        "--pc-sampling-interval",
        "1048576",
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

    output = f"{stdout}\n{stderr}"
    if is_pc_sampling_not_supported(output):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")

    assert code == 0
    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_STOCHASTIC_FILES)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    common.skip_unsupported_pc_sampling_soc()

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

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
    common.skip_unsupported_pc_sampling_soc()

    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = [
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

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
):
    """
    End-to-end: profile with PC sampling (host_trap), then
    run analysis on the profiling output.
    """
    common.skip_unsupported_pc_sampling_soc()

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = common.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

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


def test_pc_sampling_with_sol_block(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --block 2
    (PC sampling with counter collection)
    """
    common.skip_unsupported_pc_sampling_soc()

    options = [
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = common.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = common.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    assert common.check_file_pattern("- '21'", f"{workload_dir}/profiling_config.yaml")
    assert common.check_file_pattern("- '2'", f"{workload_dir}/profiling_config.yaml")

    common.clean_output_dir(config["cleanup"], workload_dir)
