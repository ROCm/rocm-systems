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
@pytest.mark.class_name("unified-memory-output-path")
class TestUnifiedMemoryOutputPath(RocprofsysTest):
    """--unified-memory-output-path redirects the reports off the trace output.

    Tests:  the unified-memory reports are written where the flag asks, and
            only there.
    Input:  --use-unified-memory-profiling --unified-memory-output-path <dir>
            on unified-memory -s 32 -p 256 -i 4.
    Expect: unified_memory*.txt and unified_memory*.json under <dir>, none
            left in the default output directory, and the reports validate.

    With the flag unset the reports land next to the active trace backend
    output. The assertion that makes this test mean something is the
    second one: the reports must be gone from that default location, not
    merely present in the requested one.
    """

    run_args = ["-s", "32", "-p", "256", "-i", "4"]

    def test_redirects_reports(self, unified_memory_environment, test_output_dir):
        # Kept under test_output_dir so assert_unified_memory_output, which
        # searches result.output_dir recursively, still finds the reports.
        reports_dir = test_output_dir / "um-reports"
        result = self.run_test(
            "sampling",
            target="unified-memory",
            env=unified_memory_environment,
            run_args=self.run_args,
            sampling_args=[
                "--use-unified-memory-profiling",
                "--unified-memory-output-path",
                str(reports_dir),
            ],
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            subtest_name="Unified-memory completion check",
            # Only the workload's own completion line. Where the reports landed
            # is checked below against the filesystem; the echoed
            # ROCPROFSYS_UNIFIED_MEMORY_OUTPUT_PATH would just repeat the
            # argument back.
            pass_regex=["9 tests completed"],
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
