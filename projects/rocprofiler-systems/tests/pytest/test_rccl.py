# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for RCCL

MPI is unsupported for RCCL tests.
"""

from __future__ import annotations
import pytest
from pathlib import Path

pytestmark = [pytest.mark.rccl, pytest.mark.disable("all")]

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
def rccl_rocpd_env() -> dict[str, str]:
    """Environment variables for RCCL rocpd tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "OFF",
        "ROCPROFSYS_TRACE_CACHED": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_USE_RCCLP": "ON",
        "ROCPROFSYS_USE_ROCPD": "ON",
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
@pytest.mark.gpu
class TestRCCL:

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

    def test_sampling(
        self,
        rccl_target: str,
        run_test,
        rccl_env: dict[str, str],
        assert_regex,
        assert_perfetto,
    ):
        result = run_test(
            "sampling",
            target=rccl_target,
            env=rccl_env,
            run_args=self.RUN_ARGS,
            timeout=300,
        )
        assert_regex(result)
        assert_perfetto(
            result,
            categories=["rocm_rccl_api"],
            counter_names=["RCCL Comm"],
        )

    def test_binary_rewrite(
        self,
        rccl_target: str,
        run_test,
        rccl_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "binary_rewrite",
            target=rccl_target,
            env=rccl_env,
            run_args=self.RUN_ARGS,
            rewrite_args=self.REWRITE_ARGS,
            timeout=300,
        )
        assert_regex(result)

    @pytest.mark.slow
    def test_runtime_instrument(
        self,
        rccl_target: str,
        run_test,
        rccl_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "runtime_instrument",
            target=rccl_target,
            env=rccl_env,
            run_args=self.RUN_ARGS,
            instrument_args=self.RUNTIME_ARGS,
            timeout=300,
        )
        assert_regex(result)

    def test_sys_run(
        self,
        rccl_target: str,
        run_test,
        rccl_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target=rccl_target,
            env=rccl_env,
            run_args=self.RUN_ARGS,
            timeout=300,
        )
        assert_regex(result)

    @pytest.mark.rocpd("rccl_rocpd_env")
    def test_rocpd(
        self,
        rccl_target: str,
        run_test,
        rccl_rocpd_env: dict[str, str],
        rccl_rocpd_rules: list[Path],
        assert_regex,
        assert_perfetto,
        assert_rocpd,
    ):
        result = run_test(
            "sys_run",
            target=rccl_target,
            env=rccl_rocpd_env,
            run_args=self.RUN_ARGS,
            timeout=300,
        )
        assert_regex(result)
        assert_perfetto(
            result,
            categories=["rocm_rccl_api"],
            counter_names=["RCCL Comm"],
        )
        assert_rocpd(result, rules_files=rccl_rocpd_rules)
