# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
UCX tests.

UCX tests require MPI examples since UCX is MPI's transport layer.
The first test checks UCX availability - if UCX is not available,
all subsequent tests in this module are skipped.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

# UCX tests require MPI examples since UCX is MPI's transport layer
pytestmark = [
    pytest.mark.ucx,
    pytest.mark.mpi,
    pytest.mark.mpi_implementation("openmpi"),
]

# =============================================================================
# UCX fixtures
# =============================================================================


@pytest.fixture
def ucx_base_env() -> dict[str, str]:
    """UCX environment."""
    return {
        "ROCPROFSYS_USE_UCX": "ON",
        "ROCPROFSYS_DL_VERBOSE": "3",
        "ROCPROFSYS_PERFETTO_BACKEND": "inprocess",
        "ROCPROFSYS_PERFETTO_FILL_POLICY": "ring_buffer",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_MPI_INIT": "OFF",
        "OMPI_MCA_pml": "ucx",  # Use UCX point-to-point messaging layer
        "OMPI_MCA_osc": "ucx",  # Use UCX one-sided communications
        "OMPI_MCA_pml_ucx_tls": "tcp,self",  # Force TCP and self (not sysv/posix/cma which bypass UCX functions)
        "OMPI_MCA_pml_ucx_devices": "any",  # Accept any device (not just InfiniBand/Mellanox)
        "OMPI_MCA_btl": "^vader,sm",  # Disable shared memory BTLs to force communication through UCX
        "UCX_TLS": "tcp,self",  # Tell UCX to use TCP for inter-process, self for intra-process
        "OMPI_MCA_pml_base_verbose": "100",  # Show which PML is selected
        "UCX_LOG_LEVEL": "info",  # Enable UCX logging to show transport usage
    }


@pytest.fixture
def ucx_active_messages_env(ucx_base_env) -> dict[str, str]:
    """UCX active messages environment."""
    env = ucx_base_env.copy()
    env.update(
        {
            "OMPI_MCA_btl": "^vader,tcp,openib,uct",
        }
    )
    return env


@pytest.fixture
def ucx_mpip_env(ucx_base_env) -> dict[str, str]:
    """UCX MPIP environment."""
    env = ucx_base_env.copy()
    env.update(
        {
            "ROCPROFSYS_USE_MPIP": "ON",
        }
    )
    return env


@pytest.fixture
def ucx_env(ucx_base_env) -> dict[str, str]:
    """UCX environment."""
    env = ucx_base_env.copy()
    env.update(
        {
            "ROCPROFSYS_TRACE_LEGACY": "ON",
            "ROCPROFSYS_PERFETTO_COMBINE_TRACES": "ON",
        }
    )
    return env


# =============================================================================
# UCX tests
# =============================================================================


class TestUCX(RocprofsysTest):
    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    @pytest.mark.parametrize(
        "target",
        [
            "mpi-all2all",
            "mpi-allgather",
            "mpi-allreduce",
            "mpi-scatter-gather",
            "mpi-send-recv",
        ],
    )
    def test(self, mode, target, ucx_env):
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
        REWRITE_PASS_REGEX = [
            r"ucx_gotcha|category::ucx|Successfully executed: .+rocprof-sys-merge-output\.sh.*"
        ]
        SYS_RUN_PASS_REGEX = [r"ucx_gotcha|category::ucx|Using UCX|pml.*ucx"]

        result = self.run_test(
            mode,
            target,
            env=ucx_env,
            rewrite_args=REWRITE_ARGS,
            run_args=RUN_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            sys_run_pass_regex=SYS_RUN_PASS_REGEX,
        )
        if mode == "sys_run":
            self.assert_perfetto(
                result,
                perfetto_file="merged.proto",
                categories=["ucx"],
                counter_names=["UCX Comm Recv", "UCX Comm Send"],
            )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    def test_perfetto(self, mode, ucx_env):
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
        REWRITE_PASS_REGEX = [
            r"ucx_gotcha|category::ucx|Successfully executed: .+rocprof-sys-merge-output\.sh.*"
        ]
        REWRITE_FAIL_REGEX = [r"Script not found|Failed to execute"]
        SYS_RUN_PASS_REGEX = [r"ucx_gotcha|category::ucx|Using UCX|pml.*ucx"]

        result = self.run_test(
            mode, "mpi-example", env=ucx_env, rewrite_args=REWRITE_ARGS, mpi_ranks=2
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            rewrite_fail_regex=REWRITE_FAIL_REGEX,
            sys_run_pass_regex=SYS_RUN_PASS_REGEX,
        )
        if mode == "sys_run":
            self.assert_perfetto(
                result,
                perfetto_file="merged.proto",
                categories=["ucx"],
                counter_names=["UCX Comm Recv", "UCX Comm Send"],
            )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    def test_mpip(self, mode, ucx_mpip_env):
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
        REWRITE_PASS_REGEX = [r"ucx_gotcha|category::ucx"]
        RUN_ARGS = ["30"]

        result = self.run_test(
            mode,
            "mpi-all2all",
            env=ucx_mpip_env,
            rewrite_args=REWRITE_ARGS,
            run_args=RUN_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
        )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    @pytest.mark.parametrize("msg_size", ["1024", "4096", "16384"])
    def test_bcast(self, mode, msg_size, ucx_env):
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
        REWRITE_PASS_REGEX = [r"ucx_gotcha|category::ucx"]

        result = self.run_test(
            mode,
            "mpi-bcast",
            env=ucx_env,
            rewrite_args=REWRITE_ARGS,
            run_args=[msg_size],
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
        )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    def test_active_message(self, mode, ucx_active_messages_env):
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
        REWRITE_PASS_REGEX = [r"ucx_gotcha|category::ucx"]
        RUN_ARGS = ["64"]

        result = self.run_test(
            mode,
            "mpi-allreduce",
            env=ucx_active_messages_env,
            rewrite_args=REWRITE_ARGS,
            run_args=RUN_ARGS,
            mpi_ranks=2,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
        )
