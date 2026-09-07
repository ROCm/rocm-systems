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
from rocprofsys import GPUInfo, TestResult

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.xnack,
    pytest.mark.hip,
    pytest.mark.unified_memory,
    # rocprofiler-sdk < 1.2.2 can abort on undefined KFD node IDs; product disables KFD domains.
    pytest.mark.rocprofiler_sdk_min_version("1.2.2"),
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

    UM_DEFAULT_TEST_PASS_REGEX = ["9 tests completed"]

    run_args = ["-s", "32", "-p", "256", "-i", "4"]

    def _assert_valid_reports(self, result: TestResult, prefix: str) -> None:
        """Validate workload completion and unified-memory report files."""
        self.assert_regex(
            result,
            subtest_name=f"{prefix} unified-memory completion check",
            pass_regex=self.UM_DEFAULT_TEST_PASS_REGEX,
        )
        self.assert_unified_memory_output(
            result,
            subtest_name=f"{prefix} unified-memory output validation",
            pass_regex=["All validation checks passed"],
        )

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

        self._assert_valid_reports(result, "Manual")

    @pytest.mark.timeout(120)
    def test_preset_generates_valid_reports(self, gpu_info: GPUInfo) -> None:
        """Validate preset reports and implicit KFD enablement.

        Only HSA_XNACK is supplied directly; AMD SMI keeps its default.
        """
        result = self.run_test(
            "sys_run",
            target="unified-memory",
            env={"HSA_XNACK": "1"},
            sys_run_args=["--preset=trace-unified-memory"],
            run_args=self.run_args,
            check_target_arch=True,
        )

        self._assert_valid_reports(result, "Preset")

        counter_names = ["Unified Memory Page Faults"]
        if "apu" not in gpu_info.categories:
            counter_names.append("Unified Memory Migration Throughput")
        self.assert_perfetto(
            result,
            subtest_name="Preset unified-memory Perfetto counter validation",
            counter_names=counter_names,
        )
