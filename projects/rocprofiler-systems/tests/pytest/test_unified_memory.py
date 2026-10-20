# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unified-memory report tests using the unified-memory example.

Validates the user-facing unified-memory text and JSON outputs generated from
KFD page fault and page migration events.
"""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest, assert_resolved_settings

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
            pass_regex=self.UM_DEFAULT_TEST_PASS_REGEX,
        )

        self.assert_unified_memory_output(
            result,
            subtest_name="Unified-memory output validation",
            pass_regex=["All validation checks passed"],
        )


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("unified-memory-cli")
class TestUnifiedMemoryCli(RocprofsysTest):
    """CLI enables unified-memory profiling and redirects its reports.

    Seeds ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING=OFF so
    --use-unified-memory-profiling is what turns it on (the shared fixture
    defaults it ON for the env-driven test above).
    """

    run_args = ["-s", "32", "-p", "256", "-i", "4"]
    CLI_FLAGS_PREFIX = ["--use-unified-memory-profiling", "--unified-memory-output-path"]

    def test_profiling_flag_and_output_path(
        self, unified_memory_environment, test_output_dir
    ):
        # Under test_output_dir so assert_unified_memory_output still finds it.
        reports_dir = test_output_dir / "um-reports"
        flag_args = [*self.CLI_FLAGS_PREFIX, str(reports_dir)]
        env = dict(unified_memory_environment)
        env["ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING"] = "OFF"

        result = self.run_test(
            "sampling",
            target="unified-memory",
            env=env,
            run_args=self.run_args,
            sampling_args=flag_args,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            pass_regex=["9 tests completed"],
        )
        assert_resolved_settings(
            result,
            {
                "ROCPROFSYS_USE_UNIFIED_MEMORY_PROFILING": True,
                "ROCPROFSYS_UNIFIED_MEMORY_OUTPUT_PATH": str(reports_dir),
            },
        )

        requested = sorted(reports_dir.glob("unified_memory*.txt")) + sorted(
            reports_dir.glob("unified_memory*.json")
        )
        if not requested:
            pytest.fail(
                "--unified-memory-output-path did not write unified_memory*.txt "
                f"and unified_memory*.json under {reports_dir}"
            )

        stray = sorted(result.output_dir.glob("unified_memory*.txt")) + sorted(
            result.output_dir.glob("unified_memory*.json")
        )
        if stray:
            pytest.fail(
                "--unified-memory-output-path was ignored: reports were still "
                f"written to the default output location: {[str(p) for p in stray]}"
            )

        self.assert_unified_memory_output(
            result,
            subtest_name="Unified-memory output validation at the requested path",
            pass_regex=["All validation checks passed"],
        )
