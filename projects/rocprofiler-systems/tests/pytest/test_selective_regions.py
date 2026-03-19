# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for selective region tracing and pause/resume integration.

Validates that:
- roctxProfilerPause/Resume correctly excludes kernels from traces
- ROCPROFSYS_TRACE_REGION filters tracing to specific roctx regions
- Pause/resume interacts correctly with region filtering at various boundaries
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu, pytest.mark.selective_regions, pytest.mark.ci_enable]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def selective_region_env() -> dict[str, str]:
    """Environment variables for selective region tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api",
    }


# =============================================================================
# Test Class: Pause/Resume
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
class TestPauseResume(RocprofsysTest):
    """Tests for roctxProfilerPause/Resume without region filtering.

    Code flow:
        CodeBlock_Z (profiled), CodeBlock_A (profiled),
        pause, CodeBlock_B (NOT profiled), resume,
        CodeBlock_C (profiled), CodeBlock_D (profiled)
    """

    def test(self, mode, selective_region_env):
        result = self.run_test(
            mode,
            "pause_resume",
            env=selective_region_env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause/Resume kernel presence",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_Z", "CodeBlock_A", "CodeBlock_C", "CodeBlock_D"],
            fail_regex=["CodeBlock_B"],
        )


# =============================================================================
# Test Class: Selective Region (no pause/resume)
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
class TestSelectiveRegion(RocprofsysTest):
    """Tests for selective region tracing without pause/resume.

    Code flow:
        CodeBlock_A (outside),
        Region 1: CodeBlock_B, Region 2: CodeBlock_C, CodeBlock_D,
        Region 3: CodeBlock_E,
        Region 1: CodeBlock_F,
        CodeBlock_G (outside)
    """

    def test_no_filter(self, mode, selective_region_env):
        """No ROCPROFSYS_TRACE_REGION — all regions traced."""
        result = self.run_test(
            mode,
            "selective_region",
            env=selective_region_env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="All kernels present",
            categories=["rocm_kernel_dispatch"],
            pass_regex=[
                "CodeBlock_A", "CodeBlock_B", "CodeBlock_C", "CodeBlock_D",
                "CodeBlock_E", "CodeBlock_F", "CodeBlock_G",
            ],
        )
        self.assert_perfetto(
            result,
            subtest_name="All regions present",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1", "Region 2", "Region 3"],
        )

    def test_region_1_filter(self, mode, selective_region_env):
        """ROCPROFSYS_TRACE_REGION='Region 1' — only Region 1 content traced.

        Region 1 spans: CodeBlock_B, CodeBlock_C (nested Region 2), CodeBlock_D,
                        CodeBlock_F (second Region 1 open)
        Outside Region 1: CodeBlock_A (before), CodeBlock_E (Region 3), CodeBlock_G (after)
        """
        env = selective_region_env.copy()
        env["ROCPROFSYS_TRACE_REGION"] = "Region 1"
        result = self.run_test(
            mode,
            "selective_region",
            env=env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Region 1 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_B", "CodeBlock_C", "CodeBlock_D", "CodeBlock_F"],
            fail_regex=["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Region 1 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1", "Region 2"],
            fail_regex=["Region 3"],
        )

    def test_region_2_and_3_filter(self, mode, selective_region_env):
        """ROCPROFSYS_TRACE_REGION='Region 2,Region 3' — only Region 2+3 content traced.

        Region 2 spans: CodeBlock_C (nested inside Region 1)
        Region 3 spans: CodeBlock_E
        Outside: CodeBlock_A, B, D, F, G and Region 1
        """
        env = selective_region_env.copy()
        env["ROCPROFSYS_TRACE_REGION"] = "Region 2,Region 3"
        result = self.run_test(
            mode,
            "selective_region",
            env=env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Region 2+3 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_C", "CodeBlock_E"],
            fail_regex=[
                "CodeBlock_A", "CodeBlock_B", "CodeBlock_D",
                "CodeBlock_F", "CodeBlock_G",
            ],
        )
        self.assert_perfetto(
            result,
            subtest_name="Region 2+3 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 2", "Region 3"],
            fail_regex=["Region 1"],
        )


# =============================================================================
# Test Class: Selective Region + Pause 1
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
class TestSelectiveRegionPause1(RocprofsysTest):
    """Pause and Resume both occur INSIDE the target region.

    Code flow:
        CodeBlock_Z (outside), Region 1 start,
        CodeBlock_A (profiled), pause, CodeBlock_B (paused), resume,
        CodeBlock_C (profiled), Region 1 stop, CodeBlock_D (outside)
    """

    def test_no_filter(self, mode, selective_region_env):
        """Without filter, pause/resume still apply so B is absent."""
        result = self.run_test(
            mode,
            "selective_region_pause_1",
            env=selective_region_env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause inside region (no filter) kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_Z", "CodeBlock_A", "CodeBlock_C", "CodeBlock_D"],
            fail_regex=["CodeBlock_B"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause inside region (no filter) markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )

    def test_region_1_filter(self, mode, selective_region_env):
        """With Region 1 filter: Z and D outside, B paused — only A and C profiled."""
        env = selective_region_env.copy()
        env["ROCPROFSYS_TRACE_REGION"] = "Region 1"
        result = self.run_test(
            mode,
            "selective_region_pause_1",
            env=env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause inside Region 1 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_A", "CodeBlock_C"],
            fail_regex=["CodeBlock_Z", "CodeBlock_B", "CodeBlock_D"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause inside Region 1 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )


# =============================================================================
# Test Class: Selective Region + Pause 2
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
class TestSelectiveRegionPause2(RocprofsysTest):
    """Pause occurs BEFORE the target region.

    Code flow:
        pause, CodeBlock_Z, Region 1 start,
        CodeBlock_A, CodeBlock_B, resume, CodeBlock_C,
        Region 1 stop, CodeBlock_D
    """

    def test_no_filter(self, mode, selective_region_env):
        """Without filter, pause is global: Z, A, B paused. Only C and D profiled."""
        result = self.run_test(
            mode,
            "selective_region_pause_2",
            env=selective_region_env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause before region (no filter) kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_C", "CodeBlock_D"],
            fail_regex=["CodeBlock_Z", "CodeBlock_A", "CodeBlock_B"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause before region (no filter) markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )

    def test_region_1_filter(self, mode, selective_region_env):
        """With Region 1 filter: pause outside is invalid, A/B/C profiled, Z/D outside."""
        env = selective_region_env.copy()
        env["ROCPROFSYS_TRACE_REGION"] = "Region 1"
        result = self.run_test(
            mode,
            "selective_region_pause_2",
            env=env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause before Region 1 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_A", "CodeBlock_B", "CodeBlock_C"],
            fail_regex=["CodeBlock_Z", "CodeBlock_D"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause before Region 1 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )


# =============================================================================
# Test Class: Selective Region + Pause 3
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
class TestSelectiveRegionPause3(RocprofsysTest):
    """Pause occurs INSIDE the region, resume occurs OUTSIDE after region stop.

    Code flow:
        Region 1 start, CodeBlock_A, pause, CodeBlock_C,
        Region 1 stop, CodeBlock_D, resume
    """

    def test_no_filter(self, mode, selective_region_env):
        """Without filter, pause is global: C and D paused. Only A profiled."""
        result = self.run_test(
            mode,
            "selective_region_pause_3",
            env=selective_region_env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause inside, resume outside (no filter) kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_A"],
            fail_regex=["CodeBlock_C", "CodeBlock_D"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause inside, resume outside (no filter) markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )

    def test_region_1_filter(self, mode, selective_region_env):
        """With Region 1 filter: A profiled, C paused, D outside. Only A profiled."""
        env = selective_region_env.copy()
        env["ROCPROFSYS_TRACE_REGION"] = "Region 1"
        result = self.run_test(
            mode,
            "selective_region_pause_3",
            env=env,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Pause inside Region 1, resume outside filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_A"],
            fail_regex=["CodeBlock_C", "CodeBlock_D"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Pause inside Region 1, resume outside filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region 1"],
        )
