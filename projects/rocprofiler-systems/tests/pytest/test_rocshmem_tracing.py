# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for rocSHMEM API tracing (rocm_rocshmem_api).

Validates that rocprofiler-systems captures host-stream rocSHMEM API calls
and surfaces them as ``rocm_rocshmem_api`` spans in the Perfetto trace.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.rocshmem,
    pytest.mark.mpi,
    pytest.mark.gpu,
]

# Binary built by the rocprofiler-sdk test suite that exercises every
# host-stream API listed in rocshmem/src/api_trace.cc.
_ROCSHMEM_DEMO = "rocshmem-demo"

# The 9 host-stream APIs registered by the rocSHMEM dispatch table.
EXPECTED_OPERATIONS = [
    "barrier_all_on_stream",
    "quiet_on_stream",
    "sync_all_on_stream",
    "alltoallmem_on_stream",
    "broadcastmem_on_stream",
    "getmem_on_stream",
    "putmem_on_stream",
    "putmem_signal_on_stream",
    "signal_wait_until_on_stream",
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rocshmem_env() -> dict[str, str]:
    """Environment variables for rocSHMEM API tracing tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "OFF",
        "ROCPROFSYS_TRACE_CACHED": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_PID": "OFF",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,rocshmem_api",
    }


@pytest.fixture(scope="session")
def rocshmem_demo_available(rocprof_config) -> tuple[bool, str]:
    """Return (True, "") if rocshmem-demo is present, else (False, reason)."""
    try:
        rocprof_config.get_target_executable(_ROCSHMEM_DEMO)
        return True, ""
    except FileNotFoundError as exc:
        return False, str(exc)


# =============================================================================
# Tests
# =============================================================================


class TestRocSHMEMTracing(RocprofsysTest):
    """rocSHMEM API tracing via ROCPROFSYS_ROCM_DOMAINS=rocshmem_api."""

    @pytest.mark.parametrize(
        "mode",
        [
            "sampling",
            pytest.param("sys_run", marks=pytest.mark.rocpd("rocshmem_env")),
        ],
    )
    def test_host_stream_apis(
        self,
        mode,
        rocshmem_env,
        rocshmem_demo_available,
        assert_perfetto,
        assert_rocpd,
        assert_regex,
    ):
        available, reason = rocshmem_demo_available
        if not available:
            pytest.skip(f"rocshmem-demo not found: {reason}")

        result = self.run_test(
            mode,
            _ROCSHMEM_DEMO,
            env=rocshmem_env,
            launcher="mpi",
            num_procs=2,
        )
        assert_regex(result)

        if mode == "sys_run":
            assert_perfetto(
                result,
                categories=["rocm_rocshmem_api"],
                # Check for each of the 9 traced APIs as label substrings.
                # rocprofiler-sdk uses the bare operation name
                # (e.g. "barrier_all_on_stream") as the span label; using
                # substrings is robust against any "rocshmem_" prefix that
                # different build configurations may emit.
                label_substrings=EXPECTED_OPERATIONS,
                skip_on_fail=True,
            )
            assert_rocpd(
                result,
                categories=["rocm_rocshmem_api"],
                label_substrings=EXPECTED_OPERATIONS,
                skip_on_fail=True,
            )
