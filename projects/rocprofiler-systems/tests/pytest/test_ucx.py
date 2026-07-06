# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
        "ROCPROFSYS_LOG_LEVEL": "debug",  # Required for rocprof-sys UCX regex validation output
        "OMPI_MCA_pml": "ucx",  # Use UCX point-to-point messaging layer
        "OMPI_MCA_osc": "ucx",  # Use UCX one-sided communications
        "OMPI_MCA_pml_ucx_tls": "tcp,self",  # Force TCP and self (not sysv/posix/cma which bypass UCX functions)
        "OMPI_MCA_pml_ucx_devices": "any",  # Accept any device (not just InfiniBand/Mellanox)
        "OMPI_MCA_btl": "^vader,sm",  # Disable shared memory BTLs to force communication through UCX
        "UCX_TLS": "tcp,self",  # Tell UCX to use TCP for inter-process, self for intra-process
        "FI_PROVIDER": "^psm3",  # Stop libfabric from loading the psm3 provider
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
    UCX_PASS_REGEX = [
        "Shutting down UCX tracing",  # Emitted by rocprof-sys (requires debug logging)
        r"pml.*ucx",  # Emitted by program when UCX logging is enabled
    ]

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
        BINARY_REWRITE_ARGS = [
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
            env=ucx_env,
            binary_rewrite_args=BINARY_REWRITE_ARGS,
            run_args=RUN_ARGS,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.UCX_PASS_REGEX,
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
        BINARY_REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "--min-instructions",
            "0",
        ]
        BINARY_REWRITE_FAIL_REGEX = [r"Script not found|Failed to execute"]

        result = self.run_test(
            mode,
            "mpi-send-recv",
            env=ucx_env,
            binary_rewrite_args=BINARY_REWRITE_ARGS,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.UCX_PASS_REGEX,
            binary_rewrite_fail_regex=BINARY_REWRITE_FAIL_REGEX,
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
        BINARY_REWRITE_ARGS = [
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
        RUN_ARGS = ["30"]

        result = self.run_test(
            mode,
            "mpi-all2all",
            env=ucx_mpip_env,
            binary_rewrite_args=BINARY_REWRITE_ARGS,
            run_args=RUN_ARGS,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.UCX_PASS_REGEX,
        )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    @pytest.mark.parametrize("msg_size", ["1024", "4096", "16384"])
    def test_bcast(self, mode, msg_size, ucx_env):
        BINARY_REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "--min-instructions",
            "0",
        ]

        result = self.run_test(
            mode,
            "mpi-bcast",
            env=ucx_env,
            binary_rewrite_args=BINARY_REWRITE_ARGS,
            run_args=[msg_size],
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.UCX_PASS_REGEX,
        )

    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    def test_active_message(self, mode, ucx_active_messages_env):
        BINARY_REWRITE_ARGS = [
            "-e",
            "-v",
            "2",
            "--label",
            "file",
            "line",
            "--min-instructions",
            "0",
        ]
        RUN_ARGS = ["64"]

        result = self.run_test(
            mode,
            "mpi-allreduce",
            env=ucx_active_messages_env,
            binary_rewrite_args=BINARY_REWRITE_ARGS,
            run_args=RUN_ARGS,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=self.UCX_PASS_REGEX,
        )
