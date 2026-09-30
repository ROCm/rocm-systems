# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for live attach/detach profiling."""

import inspect
import os
import subprocess
import time

import common

from tests.integration import common as integration_common
from tests.integration.common import (
    attach_detach_interval_msec_no_delay,
    config,
    num_kernels,
    validate,
)


def test_live_attach_detach_block(
    binary_handler_profile_rocprof_compute,
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
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

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
    file_dict = integration_common.check_csv_files(workload_dir, 1, num_kernels)
    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)


def test_live_attach_detach_pc_sampling(
    binary_handler_profile_rocprof_compute,
):
    integration_common.skip_unsupported_pc_sampling_soc(is_stochastic=True)

    options = ["--experimental", "--pc-sampling"]
    workload_dir = common.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

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

    common.clean_output_dir(config["cleanup"], workload_dir)
