# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for rocSHMEM API tracing (rocm_rocshmem_api).

Validates that rocprofiler-systems captures host-stream rocSHMEM API calls
and surfaces them as ``rocm_rocshmem_api`` spans in the Perfetto trace.
"""

from __future__ import annotations
from pathlib import Path
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.rocshmem,
    pytest.mark.mpi,
    pytest.mark.gpu,
    pytest.mark.rocprofiler_sdk_min_version("1.3.4"),
]

_ROCSHMEM_DEMO = "rocshmem-test"

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
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,rocshmem_api",
    }


@pytest.fixture
def rocshmem_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    """Validation rules for rocSHMEM rocpd database checks."""
    return [validation_rules_dir / "rocshmem" / "validation-rules.json"]


@pytest.mark.class_name("rocshmem-tracing")
@pytest.mark.parametrize(
    "mode",
    [
        "sampling",
        pytest.param("sys_run", marks=pytest.mark.rocpd("rocshmem_env")),
    ],
)
class TestRocSHMEMTracing(RocprofsysTest):
    def test_host_stream_apis(self, mode, rocshmem_env, rocshmem_rocpd_rules):
        result = self.run_test(
            mode,
            _ROCSHMEM_DEMO,
            env=rocshmem_env,
            launcher="mpi",
            num_procs=2,
            check_target_arch=True,
        )
        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_perfetto(
                result,
                categories=["rocm_rocshmem_api"],
                label_substrings=EXPECTED_OPERATIONS,
            )
            self.assert_rocpd(
                result,
                rules_files=rocshmem_rocpd_rules,
            )
