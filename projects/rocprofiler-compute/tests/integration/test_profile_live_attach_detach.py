# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for live attach/detach profiling."""

import os
import subprocess
import time
from pathlib import Path

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    attach_detach_interval_msec_no_delay,
    config,
    num_kernels,
)


def workload_env(pytestconfig):
    """Build the workload subprocess environment for live attach tests.

    Sets ROCP_TOOL_ATTACH and prepends the rocprofiler-sdk tool library
    directory to LD_LIBRARY_PATH so that rocattach can dlopen the tool
    library by basename inside the target process.
    """
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"
    sdk_tool_path = pytestconfig.getoption("--rocprofiler-sdk-tool-path", default=None)
    if sdk_tool_path:
        sdk_lib_dir = str(Path(sdk_tool_path).parent)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{sdk_lib_dir}:{existing}".rstrip(":")
    return env


def test_live_attach_detach_block(
    binary_handler_profile_rocprof_compute,
    pytestconfig,
):
    options = [
        "--block",
        "3.1.1",
        "4.1.1",
        "5.1.1",
    ]
    workload_dir = common.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = workload_env(pytestconfig)

    process_workload = None

    try:
        # Start workload
        process_workload = subprocess.Popen(config["app_hip_dynamic_shared"], env=env)
        time.sleep(5)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_no_delay,
        }

        # Run profiler (might fail / timeout / throw)
        binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=True,
            roof=False,
            app_name="app_hip_dynamic_shared",
            attach_detach_para=attach_detach,
        )

    finally:
        if process_workload and process_workload.poll() is None:
            print(f"[finally] killing workload pid={process_workload.pid}")
            process_workload.kill()
            process_workload.wait()
        # Clean up any stale rocprof-attach processes to prevent interference
        # with subsequent tests.
        subprocess.run(
            ["pkill", "-9", "-f", "rocprof-attach"],
            capture_output=True,
        )

    # Validate results
    integration_common.check_csv_files(workload_dir, 1, num_kernels)
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_live_attach_detach_pc_sampling(
    binary_handler_profile_rocprof_compute,
    pytestconfig,
):
    integration_common.skip_unsupported_pc_sampling_soc(is_stochastic=True)

    options = ["--experimental", "--pc-sampling"]
    workload_dir = common.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = workload_env(pytestconfig)

    process_workload = None

    try:
        # Start workload
        process_workload = subprocess.Popen(config["app_hip_dynamic_shared"], env=env)
        time.sleep(15)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_no_delay,
        }

        # Profiling step (may fail)
        code, stdout, stderr = binary_handler_profile_rocprof_compute(
            config,
            workload_dir,
            options,
            check_success=False,
            capture_output=True,
            stream=True,
            roof=False,
            app_name="app_hip_dynamic_shared",
            attach_detach_para=attach_detach,
        )

    finally:
        if process_workload and process_workload.poll() is None:
            print(f"[finally] killing workload pid={process_workload.pid}")
            process_workload.kill()
            process_workload.wait()
        # Clean up any stale rocprof-attach processes to prevent interference
        # with subsequent tests.
        subprocess.run(
            ["pkill", "-9", "-f", "rocprof-attach"],
            capture_output=True,
        )

    integration_common.skip_if_pc_sampling_unsupported(stdout, stderr, workload_dir)

    assert code == 0
    common.clean_output_dir(config["cleanup"], workload_dir)
