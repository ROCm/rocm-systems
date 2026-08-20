# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from pathlib import Path

import common
import pytest

from tests.integration import common as integration_common

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1


def _assert_pc_sampling_files(file_dict):
    """Assert PID-prefixed PC sampling and code-object output files."""
    file_names = file_dict.keys()
    code_obj = [
        file_name
        for file_name in file_names
        if file_name.endswith("_code_obj_info.json")
    ]
    assert len(code_obj) == 1, (
        f"expected exactly one *_code_obj_info.json, got {code_obj}"
    )
    pc_sampling_results = [
        file_name
        for file_name in file_names
        if file_name.endswith("_ps_file_results.json")
    ]
    assert len(pc_sampling_results) == 1, (
        f"expected exactly one *_ps_file_results.json, got {pc_sampling_results}"
    )

    code_obj_pid = code_obj[0].split("_", maxsplit=1)[0]
    pc_sampling_pid = pc_sampling_results[0].split("_", maxsplit=1)[0]
    assert pc_sampling_pid.isdigit()
    assert pc_sampling_pid == code_obj_pid

    dynamic_files = {*code_obj, *pc_sampling_results}
    remaining = file_names - dynamic_files
    assert remaining == {"sysinfo.csv"}


def is_pc_sampling_not_supported(output):
    """
    To be called with the stdout + stderr after profiling.
    Check whether profiling output said PC sampling is not supported on the machine
    """
    return any(
        marker in output
        for marker in (
            # rocprof-compute's own pre-flight check against the agent configs
            "is not supported on any of the agents on this system",
            # rocprofiler-sdk, when it accepts the run and then rejects the config
            "Given PC sampling configuration is not supported",
        )
    )


def _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir):
    if is_pc_sampling_not_supported(f"{stdout}\n{stderr}"):
        common.clean_output_dir(config["cleanup"], workload_dir)
        pytest.skip("PC sampling is not supported")


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    integration_common.require_pc_sampling_gpu()
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
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute, monkeypatch):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    integration_common.require_pc_sampling_gpu(is_stochastic=True)
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
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    integration_common.require_pc_sampling_gpu()
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
        stream=True,
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
    integration_common.require_pc_sampling_gpu()
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
        stream=True,
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
    integration_common.require_pc_sampling_gpu()
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
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_non_pmc_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

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
    integration_common.require_pc_sampling_gpu()
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
        stream=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    _skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    file_dict = integration_common.check_csv_files(workload_dir, num_devices, 1)
    _assert_pc_sampling_files(file_dict)

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
