# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for multi-rank (MPI) profiling."""

import inspect
from pathlib import Path

import common
import pandas as pd
import pytest

from tests.integration import common as integration_common
from tests.integration.common import (
    CSVS,
    config,
    num_devices,
    num_kernels,
)


def assert_gpu_ids_are_agent_node_ids(rank_dir):
    """Each process's dispatch GPU ids are node ids from its own agents CSV."""
    dispatch_csvs = sorted(rank_dir.glob("dispatch_*.csv.gz"))
    assert dispatch_csvs, f"No native dispatch CSV in {rank_dir}"
    for dispatch_csv in dispatch_csvs:
        agents_csv = rank_dir / dispatch_csv.name.replace("dispatch_", "agents_", 1)
        assert agents_csv.is_file(), f"No agents CSV beside {dispatch_csv.name}"
        node_ids = set(pd.read_csv(agents_csv)["node_id"])
        # Subset: agents lists every GPU, not just the ones dispatched to.
        assert set(pd.read_csv(dispatch_csv)["gpu_id"]) <= node_ids


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

        file_dict = integration_common.check_csv_files(
            str(rank_dir), num_devices, num_kernels
        )
        assert sorted(list(file_dict.keys())) == CSVS
        assert_gpu_ids_are_agent_node_ids(rank_dir)

    common.clean_output_dir(config["cleanup"], workload_dir)


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

        file_dict = integration_common.check_csv_files(
            str(rank_dir), num_devices, num_kernels
        )

        assert sorted(list(file_dict.keys())) == CSVS

    common.clean_output_dir(config["cleanup"], workload_dir)


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
