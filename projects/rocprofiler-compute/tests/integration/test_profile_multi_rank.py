# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Profile-mode multi-rank integration tests.

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

@pytest.mark.multi_rank
def test_multi_rank_profiling_no_mpi_comm(binary_handler_profile_rocprof_compute):
    """
    Test multi-rank profiling of a non-MPI application.

    The fixture launches the profiling command with mpirun.
    """
    num_ranks = 2

    workload_dir = common.get_output_dir()

    binary_handler_profile_rocprof_compute(config, workload_dir, num_ranks=num_ranks)

    # Check output for each rank
    for rank in range(num_ranks):
        rank_dir = Path(workload_dir) / str(rank)
        assert rank_dir.exists(), f"Rank directory {rank_dir} does not exist"

        file_dict = common.check_csv_files(str(rank_dir), num_devices, num_kernels)
        if soc == "MI100":
            assert sorted(list(file_dict.keys())) == CSVS
        elif soc == "MI200":
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI300" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI350" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        else:
            print(f"Testing isn't supported yet for {soc}")
            assert 0

        validate(
            inspect.stack()[0][3],
            str(rank_dir),
            file_dict,
        )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.multi_rank
def test_multi_rank_profiling_mpi_comm(
    binary_handler_profile_rocprof_compute,
):
    """
    Test multi-rank profiling of an MPI application.

    The fixture launches the profiling command with mpirun.
    """
    # Skip test if mpi_aware_laplace_eqn is not available
    app_path = config.get("app_mpi_aware_laplace_eqn", [None])[0]
    if not (app_path and Path(app_path).exists()):
        pytest.skip(
            f"mpi_aware_laplace_eqn not found, skipping {inspect.stack()[0][3]}"
        )

    num_ranks = 2

    workload_dir = common.get_output_dir()

    options = ["--iteration-multiplexing"]

    binary_handler_profile_rocprof_compute(
        config, workload_dir, options, app_name="app_mpi_aware_laplace_eqn", num_ranks=2
    )

    # Check output for each rank
    for rank in range(num_ranks):
        rank_dir = Path(workload_dir) / str(rank)
        assert rank_dir.exists(), f"Rank directory {rank_dir} does not exist"

        file_dict = common.check_csv_files(str(rank_dir), num_devices, num_kernels)

        if soc == "MI100":
            assert sorted(list(file_dict.keys())) == CSVS
        elif soc == "MI200":
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI300" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        elif "MI350" in soc:
            assert sorted(list(file_dict.keys())) == CSVS
        else:
            print(f"Testing isn't supported yet for {soc}")
            assert 0

        validate(
            inspect.stack()[0][3],
            str(rank_dir),
            file_dict,
        )

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.multi_rank
def test_wrapped_mpi(binary_handler_profile_rocprof_compute):
    """
    Test that using MPI launchers (mpirun, mpiexec, srun, orterun) after '--'
    raises an error.
    """
    config["wrapped_mpi"] = ["mpirun", "-n", "2", "./tests/occupancy"]

    workload_dir = common.get_output_dir()

    returncode = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options=[],
        check_success=False,
        app_name="wrapped_mpi",
    )

    # Should fail with exit code 1
    assert returncode == 1

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.multi_rank
def test_multi_rank_warning_application_replay(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that a warning is printed when running a multi-rank application
    in application replay mode.
    """
    # Set MPI environment variables to simulate multi-rank
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    _, stdout, stderr = binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        app_name="app_1",
        capture_output=True,
        check_success=False,
    )

    # Check that warning message is in output
    output = stdout + stderr
    assert "Multi-rank application detected" in output
    assert "Application replay mode" in output
    assert "--iteration-multiplexing" in output
    assert "--block" not in output
    assert "--set" in output

    common.clean_output_dir(config["cleanup"], workload_dir)

@pytest.mark.multi_rank
def test_multi_rank_no_warning_with_iteration_multiplexing(
    binary_handler_profile_rocprof_compute, monkeypatch
):
    """
    Test that no application replay warning is printed when running a
    multi-rank application with iteration multiplexing enabled.
    """
    monkeypatch.setenv("OMPI_COMM_WORLD_RANK", "0")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "2")

    workload_dir = common.get_output_dir()

    options = ["--iteration-multiplexing"]

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
    assert "Application replay mode" not in output

    common.clean_output_dir(config["cleanup"], workload_dir)
