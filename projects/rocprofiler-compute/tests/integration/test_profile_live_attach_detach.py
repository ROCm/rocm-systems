# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode live attach/detach integration tests.

Extracted verbatim from tests/test_profile_general.py during the test-suite
refactor; no logic changes.
"""

import csv
import inspect
import os
import re
import socket
import subprocess
import sqlite3
import sys
import time
from pathlib import Path

import common
import numpy as np
import pandas as pd
import pytest
import yaml

from _profile_helpers import (
    AVAILABLE_SETS,
    CSVS,
    DEFAULT_ABS_DIFF,
    DEFAULT_REL_DIFF,
    GPU_ARCH,
    GPU_MODEL,
    MAX_REOCCURING_COUNT,
    METRIC_THRESHOLDS,
    MockMachineSpecs,
    MockProfiler,
    MockSoc,
    ROOF_ONLY_FILES,
    SLURM_RANK_VAR,
    SLURM_SIZE_VAR,
    are_deterministic_counters_equal,
    are_stochastic_counters_similar,
    arch_config,
    attach_detach_interval_msec_no_delay,
    attach_detach_interval_msec_with_delay,
    baseline_compare_metric,
    clear_rank_env,
    config,
    counter_compare,
    get_available_sets_for_arch,
    gpu_arch,
    is_rdna35_halo_soc,
    mock_generate_machine_specs,
    mock_load_soc_specs,
    num_devices,
    num_kernels,
    skip_unsupported_roofline_soc,
    soc,
    validate,
)

@pytest.mark.live_attach_detach
@pytest.mark.parametrize("profile_format", ["rocpd", "csv"])
def test_live_attach_detach_block(
    binary_handler_profile_rocprof_compute, profile_format
):
    options = [
        "--block",
        "3.1.1",
        "4.1.1",
        "5.1.1",
        "--format-rocprof-output",
        profile_format,
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
    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(inspect.stack()[0][3], workload_dir, file_dict)
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.skip(
    reason="Temporarily disabled: \
                  waiting for SDK fix for no outputfile with thread sleeping"
)
@pytest.mark.live_attach_detach
def test_live_attach_detach_block_thread_sleep(binary_handler_profile_rocprof_compute):
    options = ["--block", "3.1.1", "4.1.1", "5.1.1"]
    workload_dir = common.get_output_dir()

    # TODO: temp fix for sdk defautly disable attach/detach,
    # remove after it sets default to enable
    env = os.environ.copy()
    env["ROCP_TOOL_ATTACH"] = "1"

    process_workload = None

    try:
        # Start workload with sleep mode enabled
        process_workload = subprocess.Popen(
            [*config["app_hip_dynamic_shared"], "--enable-sleep"], env=env
        )
        time.sleep(5)  # Give workload time to start

        attach_detach = {
            "attach_pid": process_workload.pid,
            "attach-duration-msec": attach_detach_interval_msec_with_delay,
        }

        # Main profiling call (can fail or hang)
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

    # Validate output
    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    # Check profiling_config.yaml block entries
    config_file = f"{workload_dir}/profiling_config.yaml"
    assert common.check_file_pattern("- 3.1.1", config_file)
    assert common.check_file_pattern("- 4.1.1", config_file)
    assert common.check_file_pattern("- 5.1.1", config_file)
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.live_attach_detach
def test_live_attach_detach_singlepass_launch_stats(
    binary_handler_profile_rocprof_compute,
):
    options = ["--set", "launch_stats"]
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

    # Validate CSVs & output correctness
    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    # Check that launch-stat sets were applied
    config_file = f"{workload_dir}/profiling_config.yaml"
    for tag in [
        "7.1.0",
        "7.1.1",
        "7.1.2",
        "7.1.5",
        "7.1.6",
        "7.1.7",
        "7.1.8",
        "7.1.9",
    ]:
        assert common.check_file_pattern(f"- {tag}", config_file)

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.live_attach_detach
def test_live_attach_detach_pc_sampling(
    binary_handler_profile_rocprof_compute,
):
    common.skip_unsupported_pc_sampling_soc(is_stochastic=True)

    options = ["-b", "21"]
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
