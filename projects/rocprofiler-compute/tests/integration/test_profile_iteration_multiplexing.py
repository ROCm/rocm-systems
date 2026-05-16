# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode iteration-multiplexing integration tests.

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

@pytest.mark.iteration_multiplexing_1
def test_profiler_options(binary_handler_profile_rocprof_compute):
    options = ["--no-native-tool", "--iteration-multiplexing"]
    workload_dir = common.get_output_dir()
    code = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=False, roof=False
    )
    assert code == 1

@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, num_devices, num_kernels)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel(binary_handler_profile_rocprof_compute):
    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, num_devices, num_kernels)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.iteration_multiplexing_1
def test_iteration_multiplexing_kernel_launch_params(
    binary_handler_profile_rocprof_compute,
):
    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = common.get_output_dir()
    _ = binary_handler_profile_rocprof_compute(
        config, workload_dir, options, check_success=True, roof=False
    )

    file_dict = common.check_csv_files(workload_dir, num_devices, num_kernels)
    assert sorted(list(file_dict.keys())) == CSVS

    validate(
        inspect.stack()[0][3],
        workload_dir,
        file_dict,
    )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.iteration_multiplexing_2
@pytest.mark.xfail(
    reason="Multiple profiling workloads mapped to the same GPU corrupts the counters"
)
def test_iteration_multiplexing_deterministic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    # These metrics should cover the deterministic counters being checked
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "6.1.5", "6.1.6", "7.2.2", "10.1"]
    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "6.1.5",
        "6.1.6",
        "7.2.2",
        "10.1",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir_klp) / "pmc_perf.csv")

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
    assert len(roofline_df) >= num_devices

    common.clean_output_dir(config["cleanup"], workload_dir_klp)

@pytest.mark.iteration_multiplexing_stochastic
def test_iteration_multiplexing_stochastic_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    skip_unsupported_roofline_soc()

    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    # These metrics should cover the L1 cache stochastic counters
    # Block 4 (roofline) included to verify roofline counters under multiplexing
    options = ["--block", "4", "16.1", "16.3"]
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = [
        "--block",
        "4",
        "16.1",
        "16.3",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir_klp = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir_klp,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir_klp, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir_klp])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir_klp) / "pmc_perf.csv")

    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

    assert os.path.exists(f"{workload_dir_klp}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir_klp}/roofline.csv")
    assert len(roofline_df) >= num_devices

    common.clean_output_dir(config["cleanup"], workload_dir_klp)


# Not part of automated test runs since testing all counters is expensive

def test_iteration_multiplexing_all_counter_accuracy(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    workload_dir = common.get_output_dir(param_id="no_iter_mplx")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_no_multiplexing = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel"]
    workload_dir = common.get_output_dir(param_id="iter_mplx_kernel")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    options = ["--iteration-multiplexing", "kernel_launch_params"]
    workload_dir = common.get_output_dir(param_id="iter_mplx_params")
    _ = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_iter",
    )
    common.check_csv_files(workload_dir, num_devices, num_kernels)
    binary_handler_analyze_rocprof_compute(["analyze", "--path", workload_dir])
    counters_kernel_launch_params = pd.read_csv(Path(workload_dir) / "pmc_perf.csv")
    common.clean_output_dir(config["cleanup"], workload_dir)

    assert are_deterministic_counters_equal(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )
    assert are_stochastic_counters_similar(
        [counters_kernel, counters_kernel_launch_params], counters_no_multiplexing
    )

@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_insufficient_dispatches(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    capsys,
):
    """Verify graceful degradation when dispatches are too few for full
    counter coverage under iteration multiplexing.
    """
    options = [
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_insufficient")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=False,
        app_name="app_laplace_eqn_insufficient",
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    captured = capsys.readouterr()
    assert "missing counter data" in captured.out

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.iteration_multiplexing_2
def test_iteration_multiplexing_data_types(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
):
    """Verify roofline analysis with different data types (FP32 and FP16)
    on iteration-multiplexed profiling data.
    """
    skip_unsupported_roofline_soc()

    options = [
        "--block",
        "4",
        "--iteration-multiplexing",
        "kernel_launch_params",
    ]
    workload_dir = common.get_output_dir(param_id="iter_mplx_dtypes")
    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options,
        check_success=True,
        roof=True,
        app_name="app_laplace_eqn",
    )

    assert os.path.exists(f"{workload_dir}/roofline.csv")
    roofline_df = pd.read_csv(f"{workload_dir}/roofline.csv")
    assert len(roofline_df) >= num_devices

    code_fp32 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP32",
    ])
    assert code_fp32 == 0

    code_fp16 = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "--roofline-data-type",
        "FP16",
    ])
    assert code_fp16 == 0

    common.clean_output_dir(config["cleanup"], workload_dir)
