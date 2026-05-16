# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode section / list-metrics integration tests.

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

@pytest.mark.section
def test_lds_section(binary_handler_profile_rocprof_compute):
    lds_block = "3" if is_rdna35_halo_soc() else "12"
    options = ["--block", lds_block]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{lds_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(common.check_file_pattern("SQ_INSTS_LDS", str(f)) for f in results_files)
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_instmix_memchart_section(binary_handler_profile_rocprof_compute):
    instmix_block = "7" if is_rdna35_halo_soc() else "10"
    options = ["--block", instmix_block, "3"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{instmix_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    assert common.check_file_pattern("- '3'", f"{workload_dir}/profiling_config.yaml")
    instmix_counter = "SQ_INSTS_FLAT" if is_rdna35_halo_soc() else "TA_FLAT_WAVEFRONTS"
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        common.check_file_pattern(instmix_counter, str(f)) for f in results_files
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        common.check_file_pattern("SQC_TC_DATA_READ_REQ", str(f)) for f in results_files
    )
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_lds_sol_section(binary_handler_profile_rocprof_compute):
    lds_sol_block = "3" if is_rdna35_halo_soc() else "12.1"
    options = ["--block", lds_sol_block]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{lds_sol_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    lds_sol_counter = (
        "SQC_LDS_IDX_ACTIVE" if is_rdna35_halo_soc() else "SQ_ACTIVE_INST_LDS"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(
        common.check_file_pattern(lds_sol_counter, str(f)) for f in results_files
    )
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_instmix_section_global_write_kernel(binary_handler_profile_rocprof_compute):
    instmix_block = "7" if is_rdna35_halo_soc() else "10"
    options = ["-k", "global_write", "--block", instmix_block]
    custom_config = dict(config)
    custom_config["kernel_name_1"] = "global_write"
    custom_config["app_1"] = ["./tests/vmem"]
    num_kernels = 1

    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        custom_config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, 1, num_kernels)
    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    assert common.check_file_pattern(
        f"- '{instmix_block}'", f"{workload_dir}/profiling_config.yaml"
    )
    assert common.check_file_pattern(
        "- global_write", f"{workload_dir}/profiling_config.yaml"
    )
    kernel_counter = (
        "SQ_INSTS_FLAT_STORE" if is_rdna35_halo_soc() else "TA_FLAT_WAVEFRONTS"
    )
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(common.check_file_pattern(kernel_counter, str(f)) for f in results_files)
    results_files = Path(workload_dir).glob("results_*.csv")
    assert any(common.check_file_pattern("global_write", str(f)) for f in results_files)
    results_files = Path(workload_dir).glob("results_*.csv")
    assert not any(
        common.check_file_pattern("global_read", str(f)) for f in results_files
    )
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_list_metrics(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_list_metrics_with_block(binary_handler_profile_rocprof_compute):
    options = ["--list-metrics", "gfx90a", "--block", "10"]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.section
def test_list_available_metrics(binary_handler_profile_rocprof_compute, capsys):
    options = ["--list-available-metrics"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)

    # Test output
    output = capsys.readouterr().out
    assert "0 -> Top Stats" in output
    assert "1 -> System Info" in output

@pytest.mark.section
def test_list_available_metrics_with_block(
    binary_handler_profile_rocprof_compute, capsys
):
    options = ["--list-available-metrics", "--block", "10"]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    # Should return code 1 since --block cannot be used with --list-available-metrics
    assert code == 1
    # workload dir should not exist
    assert not Path(workload_dir).exists()
    common.clean_output_dir(config["cleanup"], workload_dir)
