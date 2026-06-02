# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unified-memory report tests using the unified-memory example.

Validates the user-facing unified-memory text and JSON outputs generated from
KFD page fault and page migration events.
"""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xnack,
    pytest.mark.hip,
    pytest.mark.unified_memory,
]


@pytest.fixture
def unified_memory_environment() -> dict[str, str]:
    """Environment variables for unified-memory report tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,kfd_events",
        "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING": "ON",
        "ROCPROFSYS_USE_AMD_SMI": "OFF",
        "HSA_XNACK": "1",
    }


@pytest.mark.class_name("unified-memory")
class TestUnifiedMemory(RocprofsysTest):
    """Validate unified-memory reports generated from the HIP example."""

    run_args = ["-s", "32", "-p", "256", "-i", "4"]

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_output(self, mode, unified_memory_environment):
        """Run unified-memory and validate text/JSON report generation."""
        result = self.run_test(
            mode,
            target="unified-memory",
            env=unified_memory_environment,
            run_args=self.run_args,
            check_target_arch=True,
        )

        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            pass_regex=[r"6 tests completed"],
        )

        self.assert_unified_memory_output(
            result,
            subtest_name="Unified-memory output validation",
            pass_regex=["All validation checks passed"],
        )
