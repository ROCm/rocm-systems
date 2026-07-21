# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for rocSHMEM API tracing (rocm_rocshmem_api).

Validates that rocprofiler-systems captures host-stream rocSHMEM API calls
and surfaces them as ``rocm_rocshmem_api`` spans in the Perfetto trace.
"""

from __future__ import annotations
import os
import subprocess
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.rocshmem,
    pytest.mark.mpi,
    pytest.mark.gpu,
]

_ROCSHMEM_DEMO = "rocshmem"

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
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }


@pytest.fixture(scope="session")
def rocshmem_demo_available(rocprof_config) -> tuple[bool, str]:
    """Return (True, "") if rocshmem is present and functional on this
    system's GPU, else (False, reason).

    A probe subprocess run (without rocprofiler-systems instrumentation) is
    performed to detect runtime failures such as rocshmem_init() aborting on
    unsupported GPU hardware.  Any non-zero exit code — including signals such
    as SIGABRT (134) — is treated as "not available" so the test skips rather
    than fails.
    """
    try:
        exe = rocprof_config.get_target_executable(_ROCSHMEM_DEMO)
    except FileNotFoundError as exc:
        return False, str(exc)

    env = {
        **os.environ,
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    try:
        probe = subprocess.run(
            ["mpirun", "-np", "2", exe],
            env=env,
            timeout=60,
            capture_output=True,
        )
        if probe.returncode != 0:
            stderr = probe.stderr.decode(errors="replace").strip()
            return False, (
                f"rocSHMEM runtime not supported on this system "
                f"(probe exit {probe.returncode})" + (f": {stderr}" if stderr else "")
            )
    except subprocess.TimeoutExpired:
        return False, "rocshmem probe timed out (rocSHMEM may hang on init)"
    except FileNotFoundError as exc:
        return False, f"mpirun not found: {exc}"

    return True, ""


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
            pytest.skip(f"rocshmem not found: {reason}")

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
                label_substrings=EXPECTED_OPERATIONS,
            )
            assert_rocpd(
                result,
                categories=["rocm_rocshmem_api"],
                label_substrings=EXPECTED_OPERATIONS,
            )
