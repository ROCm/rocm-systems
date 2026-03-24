##############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import os

import pytest
import test_utils

config = {}
config["app_1"] = ["./tests/vcopy", "-n", "1048576", "-b", "256", "-i", "3"]
config["app_mat_mul_max"] = ["./tests/mat_mul_max"]
config["cleanup"] = True
config["COUNTER_LOGGING"] = False
config["METRIC_COMPARE"] = False

num_devices = 1

soc = test_utils.gpu_soc()

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


def test_pc_sampling_host_trap(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method host_trap.
    """
    if soc == "MI100":
        assert True
        return

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = test_utils.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = test_utils.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_stochastic(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --pc-sampling-method stochastic.
    """
    if soc == "MI100" or soc == "MI200":
        assert True
        return

    options = [
        "--block",
        "21",
        "--pc-sampling-method",
        "stochastic",
        "--pc-sampling-interval",
        "1048576",
    ]

    workload_dir = test_utils.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = test_utils.check_non_pmc_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_STOCHASTIC_FILES)

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_pc_sampling_only(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no multi-rank warning is printed when running with only
    --block 21 (PC sampling only mode requires a single pass) with multi-rank.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    workload_dir = test_utils.get_output_dir()

    options = ["--block", "21"]

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

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_multi_rank_warning_pc_sampling_with_counters(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a multi-rank warning is printed when running with --block 21
    and another block (PC sampling with counters mode requires multiple passes)
    with multi-rank.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")

    workload_dir = test_utils.get_output_dir()

    options = ["--block", "21", "2"]

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

    test_utils.clean_output_dir(config["cleanup"], workload_dir)


def test_pc_sampling_with_sol_block(binary_handler_profile_rocprof_compute):
    """
    Test that PC sampling works with --block 21 and --block 2
    (PC sampling with counter collection)
    """
    if soc == "MI100":
        assert True
        return

    options = [
        "--block",
        "21",
        "2",
        "--pc-sampling-method",
        "host_trap",
        "--pc-sampling-interval",
        "256",
    ]

    workload_dir = test_utils.get_output_dir()

    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_mat_mul_max",
    )

    file_dict = test_utils.check_csv_files(workload_dir, num_devices, 1)
    assert sorted(list(file_dict.keys())) == sorted(PC_SAMPLING_HOST_TRAP_FILES)

    assert test_utils.check_file_pattern(
        "- '21'", f"{workload_dir}/profiling_config.yaml"
    )
    assert test_utils.check_file_pattern(
        "- '2'", f"{workload_dir}/profiling_config.yaml"
    )

    test_utils.clean_output_dir(config["cleanup"], workload_dir)
