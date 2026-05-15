# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for RCCL
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest
from pathlib import Path

pytestmark = [
    pytest.mark.rccl,
    pytest.mark.mpi,
    pytest.mark.gpu,
]

# =============================================================================
# RCCL fixtures
# =============================================================================


@pytest.fixture
def rccl_env() -> dict[str, str]:
    """Environment variables for RCCL tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "OFF",
        "ROCPROFSYS_TRACE_CACHED": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_USE_RCCLP": "ON",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
    }


@pytest.fixture
def rccl_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for RCCL rocpd tests."""
    rules_dir = validation_rules_dir / "rccl"
    return [
        rules_dir / "rccl-comm-rules.json",
    ]


# =============================================================================
# RCCL tests
# =============================================================================


# RCCL test binaries
RCCL_TARGETS = [
    "all_reduce_perf",
    "all_gather_perf",
    "broadcast_perf",
    "reduce_scatter_perf",
    "reduce_perf",
    "alltoall_perf",
    "scatter_perf",
    "gather_perf",
    "sendrecv_perf",
    "alltoallv_perf",
]


@pytest.mark.parametrize(
    "rccl_target",
    RCCL_TARGETS,
    ids=[t.replace("_", "-") for t in RCCL_TARGETS],
)
class TestRCCL(RocprofsysTest):
    REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "-i",
        "8",
        "--label",
        "file",
        "line",
        "return",
        "args",
    ]
    RUNTIME_ARGS = [
        "-e",
        "-v",
        "1",
        "-i",
        "8",
        "--label",
        "file",
        "line",
        "return",
        "args",
        "-ME",
        "sysdeps",
        "--log-file",
        "rccl-test.log",
    ]
    RUN_ARGS = [
        "-t",
        "1",
        "-g",
        "1",
        "-i",
        "10",
        "-w",
        "2",
        "-m",
        "2",
        "-p",
        "-c",
        "1",
        "-z",
        "-s",
        "1",
    ]

    @pytest.mark.parametrize(
        "mode",
        [
            "sampling",
            "binary_rewrite",
            pytest.param("sys_run", marks=pytest.mark.rocpd("rccl_env")),
            pytest.param("runtime_instrument", marks=pytest.mark.slow),
        ],
    )
    def test(self, mode, rccl_target, rccl_env, rccl_rocpd_rules):
        result = self.run_test(
            mode,
            rccl_target,
            env=rccl_env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            run_args=self.RUN_ARGS,
            timeout=300,
            mpi_ranks=1,
        )
        self.assert_regex(result)
        if mode == "sys_run":
            self.assert_perfetto(
                result,
                categories=["rocm_rccl_api"],
                counter_names=["RCCL Comm"],
            )
            self.assert_rocpd(result, rules_files=rccl_rocpd_rules)
