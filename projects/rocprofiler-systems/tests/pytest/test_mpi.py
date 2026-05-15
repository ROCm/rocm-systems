# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
MPI tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.mpi]

# =============================================================================
# MPI fixtures
# =============================================================================


@pytest.fixture
def mpip_env() -> dict[str, str]:
    """Environment variables for MPI tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_FILE_OUTPUT": "ON",
        "ROCPROFSYS_USE_MPIP": "ON",
        "ROCPROFSYS_DEBUG": "OFF",
        "ROCPROFSYS_VERBOSE": "2",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_DL_VERBOSE": "2",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
    }


@pytest.fixture
def mpip_flat_env(flat_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for MPI flat MPIP test."""
    env = flat_env.copy()
    env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
    env["ROCPROFSYS_STRICT_CONFIG"] = "OFF"
    env["ROCPROFSYS_USE_MPIP"] = "ON"
    return env


@pytest.fixture
def mpi_flat_env(flat_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for MPI flat test."""
    env = flat_env.copy()
    env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
    return env


@pytest.fixture
def mpip_all2all_env(mpip_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for MPI all2all test."""
    env = mpip_env.copy()
    env["ROCPROFSYS_DEBUG"] = "ON"
    env["ROCPROFSYS_VERBOSE"] = "3"
    env["ROCPROFSYS_DL_VERBOSE"] = "3"
    return env


# =============================================================================
# MPI Tests
# =============================================================================


class TestMPI(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode", ["baseline", "sampling", "binary_rewrite", "sys_run"]
    )
    def test(self, mode):
        REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "return",
            "args",
            "--min-instructions",
            "0",
        ]
        REWRITE_PASS_REGEX = [r"perfetto-trace-0\.proto", r"wall_clock-0\.txt"]
        REWRITE_FAIL_REGEX = [
            r"Outputting.*(perfetto-trace|trip_count|sampling_percent|sampling_cpu_clock|sampling_wall_clock|wall_clock)-[0-9][0-9]+.(json|txt|proto)"
        ]
        ENV = {"ROCPROFSYS_VERBOSE": "1"}

        result = self.run_test(
            mode,
            "mpi-example",
            env=ENV,
            rewrite_args=REWRITE_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            rewrite_fail_regex=REWRITE_FAIL_REGEX,
        )

    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite", "sys_run"])
    def test_perfetto_merge(self, mode):
        REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "--min-instructions",
            "0",
        ]
        REWRITE_PASS_REGEX = [r"Successfully executed: .+rocprof-sys-merge-output.sh.*"]
        REWRITE_FAIL_REGEX = ["Script not found", "Failed to execute"]
        ENV = {
            "ROCPROFSYS_VERBOSE": "1",
            "ROCPROFSYS_TRACE_LEGACY": "ON",
            "ROCPROFSYS_PERFETTO_COMBINE_TRACES": "ON",
        }
        result = self.run_test(
            mode,
            "mpi-example",
            env=ENV,
            rewrite_args=REWRITE_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            rewrite_fail_regex=REWRITE_FAIL_REGEX,
        )


class TestMPIP(RocprofsysTest):
    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    @pytest.mark.parametrize(
        "target",
        [
            "mpi-all2all",
            "mpi-allgather",
            "mpi-allreduce",
            "mpi-bcast",
            "mpi-reduce",
            "mpi-scatter-gather",
            "mpi-send-recv",
        ],
    )
    def test(self, mode, target, mpip_env, mpip_all2all_env):
        REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "--min-instructions",
            "0",
        ]
        RUN_ARGS = ["30"]

        result = self.run_test(
            mode,
            target,
            env=mpip_env if target != "mpi-all2all" else mpip_all2all_env,
            rewrite_args=REWRITE_ARGS,
            run_args=RUN_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(result)

    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite", "sys_run"])
    def test_flat(self, mode, mpip_flat_env):
        REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "args",
            "--min-instructions",
            "0",
        ]
        REWRITE_PASS_REGEX = [
            r">>> mpi-example.inst",
            r">>> MPI_Init_thread",
            r">>> pthread_create",
            r">>> MPI_Comm_size",
            r">>> MPI_Comm_rank",
            r">>> MPI_Barrier",
            r">>> MPI_Alltoall",
        ]

        result = self.run_test(
            mode,
            "mpi-example",
            env=mpip_flat_env,
            rewrite_args=REWRITE_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
        )
