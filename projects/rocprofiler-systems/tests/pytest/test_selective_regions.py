# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for selective region tracing and pause/resume integration.

Validates that:
- roctxProfilerPause/Resume correctly excludes kernels from traces
- ROCPROFSYS_SELECTED_REGIONS filters tracing to specific roctx regions
- Pause/resume interacts correctly with region filtering at various boundaries
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.selective_regions,
    pytest.mark.timeout(120),
    pytest.mark.rocm,
]

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def selective_region_env() -> dict[str, str]:
    """Environment variables for selective region tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api",
    }


@pytest.fixture
def no_marker_env() -> dict[str, str]:
    """Environment variables for tests without marker_api (ConditionB only).

    When marker_api is NOT in ROCM_DOMAINS, pause/resume is IGNORED.
    Region filtering via ROCPROFSYS_SELECTED_REGIONS still works.
    """
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch",
    }


# =============================================================================
# Test Class: Pause/Resume
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
@pytest.mark.parametrize("marker_api", ["enabled", "disabled"])
@pytest.mark.class_name("pause-resume")
class TestPauseResume(RocprofsysTest):
    """Tests for roctxProfilerPause/Resume without region filtering.

    Code flow:
        CodeBlock_Z (profiled), CodeBlock_A (profiled),
        pause, CodeBlock_B (NOT profiled), resume,
        CodeBlock_C (profiled), CodeBlock_D (profiled)

    When marker_api is disabled, pause/resume is IGNORED — ALL kernels profiled.
    """

    def test(self, mode, marker_api, selective_region_env, no_marker_env):
        if marker_api == "enabled":
            env = selective_region_env.copy()
            subtest_name = "Pause/Resume kernel presence"
            pass_regex = ["CodeBlock_Z", "CodeBlock_A", "CodeBlock_C", "CodeBlock_D"]
            fail_regex = ["CodeBlock_B"]
        else:
            env = no_marker_env.copy()
            subtest_name = "Pause/Resume ignored without marker_api"
            pass_regex = [
                "CodeBlock_Z",
                "CodeBlock_A",
                "CodeBlock_B",
                "CodeBlock_C",
                "CodeBlock_D",
            ]
            fail_regex = []

        result = self.run_test(
            mode,
            "pause_resume",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name=subtest_name,
            categories=["rocm_kernel_dispatch"],
            pass_regex=pass_regex,
            fail_regex=fail_regex,
        )


# =============================================================================
# Test Class: Selective Region (no pause/resume)
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
@pytest.mark.class_name("selective-region")
class TestSelectiveRegion(RocprofsysTest):
    """Tests for selective region tracing without pause/resume.

    Code flow:
        CodeBlock_A (outside),
        Region1: CodeBlock_B, Region2: CodeBlock_C, CodeBlock_D,
        Region3: CodeBlock_E,
        Region1: CodeBlock_F,
        CodeBlock_G (outside)
    """

    def test_no_filter(self, mode, selective_region_env):
        """No ROCPROFSYS_SELECTED_REGIONS — all regions traced."""
        result = self.run_test(
            mode,
            "selective_region",
            env=selective_region_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="All kernels present",
            categories=["rocm_kernel_dispatch"],
            pass_regex=[
                "CodeBlock_A",
                "CodeBlock_B",
                "CodeBlock_C",
                "CodeBlock_D",
                "CodeBlock_E",
                "CodeBlock_F",
                "CodeBlock_G",
            ],
        )
        self.assert_perfetto(
            result,
            subtest_name="All regions present",
            categories=["rocm_marker_api"],
            pass_regex=["Region1", "Region2", "Region3"],
        )

    def test_region_1_filter(self, mode, selective_region_env):
        """ROCPROFSYS_SELECTED_REGIONS='Region1' — only Region1 content traced.

        Region1 spans: CodeBlock_B, CodeBlock_C (nested Region2), CodeBlock_D,
                        CodeBlock_F (second Region1 open)
        Outside Region1: CodeBlock_A (before), CodeBlock_E (Region3), CodeBlock_G (after)
        """
        env = selective_region_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            mode,
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Region1 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_B", "CodeBlock_C", "CodeBlock_D", "CodeBlock_F"],
            fail_regex=["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"],
        )
        self.assert_perfetto(
            result,
            subtest_name="Region1 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region1", "Region2"],
            fail_regex=["Region3"],
        )

    def test_region_2_and_3_filter(self, mode, selective_region_env):
        """ROCPROFSYS_SELECTED_REGIONS='Region2,Region3' — only Region2+3 content traced.

        Region2 spans: CodeBlock_C (nested inside Region1)
        Region3 spans: CodeBlock_E
        Outside: CodeBlock_A, B, D, F, G and Region1
        """
        env = selective_region_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region2,Region3"
        result = self.run_test(
            mode,
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Region2+3 filtered kernels",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_C", "CodeBlock_E"],
            fail_regex=[
                "CodeBlock_A",
                "CodeBlock_B",
                "CodeBlock_D",
                "CodeBlock_F",
                "CodeBlock_G",
            ],
        )
        self.assert_perfetto(
            result,
            subtest_name="Region2+3 filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region2", "Region3"],
            fail_regex=["Region1"],
        )


# =============================================================================
# Test Class: Selective Region + Pause
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
@pytest.mark.parametrize(
    "target",
    [
        pytest.param("selective_region_pause_1", id="inside"),
        pytest.param("selective_region_pause_2", id="before"),
        pytest.param("selective_region_pause_3", id="outside"),
    ],
)
@pytest.mark.class_name("selective-region-pause")
class TestSelectiveRegionPause(RocprofsysTest):
    """Tests for pause/resume interaction with selective region filtering.

    Target 1: Pause and Resume both occur INSIDE the target region.
        Code flow: CodeBlock_Z (outside), Region1 start,
        CodeBlock_A (profiled), pause, CodeBlock_B (paused), resume,
        CodeBlock_C (profiled), Region1 stop, CodeBlock_D (outside)

    Target 2: Pause occurs BEFORE the target region.
        Code flow: pause, CodeBlock_Z, Region1 start,
        CodeBlock_A, CodeBlock_B, resume, CodeBlock_C,
        Region1 stop, CodeBlock_D

    Target 3: Pause occurs INSIDE the region, resume occurs OUTSIDE after region stop.
        Code flow: Region1 start, CodeBlock_A, pause, CodeBlock_C,
        Region1 stop, CodeBlock_D, resume
    """

    def test_no_filter(self, mode, target, selective_region_env):
        """Without filter, pause/resume apply globally."""
        result = self.run_test(
            mode,
            target,
            env=selective_region_env,
            check_target_arch=True,
        )
        self.assert_regex(result)

        if target == "selective_region_pause_1":
            subtest_name = "Pause inside region (no filter) kernels"
            pass_regex = ["CodeBlock_Z", "CodeBlock_A", "CodeBlock_C", "CodeBlock_D"]
            fail_regex = ["CodeBlock_B"]
        elif target == "selective_region_pause_2":
            subtest_name = "Pause before region (no filter) kernels"
            pass_regex = ["CodeBlock_C", "CodeBlock_D"]
            fail_regex = ["CodeBlock_Z", "CodeBlock_A", "CodeBlock_B"]
        else:  # selective_region_pause_3
            subtest_name = "Pause inside, resume outside (no filter) kernels"
            pass_regex = ["CodeBlock_A"]
            fail_regex = ["CodeBlock_C", "CodeBlock_D"]

        self.assert_perfetto(
            result,
            subtest_name=subtest_name,
            categories=["rocm_kernel_dispatch"],
            pass_regex=pass_regex,
            fail_regex=fail_regex,
        )
        self.assert_perfetto(
            result,
            subtest_name=f"{subtest_name} markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region1"],
        )

    def test_filtered(self, mode, target, selective_region_env):
        """With Region1 filter: region filtering combined with pause/resume."""
        env = selective_region_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            mode,
            target,
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)

        if target == "selective_region_pause_1":
            subtest_name = "Pause inside Region1 filtered kernels"
            pass_regex = ["CodeBlock_A", "CodeBlock_C"]
            fail_regex = ["CodeBlock_Z", "CodeBlock_B", "CodeBlock_D"]
        elif target == "selective_region_pause_2":
            subtest_name = "Pause before Region1 filtered kernels"
            pass_regex = ["CodeBlock_A", "CodeBlock_B", "CodeBlock_C"]
            fail_regex = ["CodeBlock_Z", "CodeBlock_D"]
        else:  # selective_region_pause_3
            subtest_name = "Pause inside Region1, resume outside filtered kernels"
            pass_regex = ["CodeBlock_A"]
            fail_regex = ["CodeBlock_C", "CodeBlock_D"]

        self.assert_perfetto(
            result,
            subtest_name=subtest_name,
            categories=["rocm_kernel_dispatch"],
            pass_regex=pass_regex,
            fail_regex=fail_regex,
        )
        self.assert_perfetto(
            result,
            subtest_name=f"{subtest_name} markers",
            categories=["rocm_marker_api"],
            pass_regex=["Region1"],
        )

    def test_no_marker(self, mode, target, no_marker_env):
        """With Region1 filter but no marker_api: pause/resume ignored."""
        env = no_marker_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            mode,
            target,
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)

        if target == "selective_region_pause_1":
            subtest_name = "Pause inside Region1 ignored (no marker_api)"
            pass_regex = ["CodeBlock_A", "CodeBlock_B", "CodeBlock_C"]
            fail_regex = ["CodeBlock_Z", "CodeBlock_D"]
        elif target == "selective_region_pause_2":
            subtest_name = "Pause before Region1 ignored (no marker_api)"
            pass_regex = ["CodeBlock_A", "CodeBlock_B", "CodeBlock_C"]
            fail_regex = ["CodeBlock_Z", "CodeBlock_D"]
        else:  # selective_region_pause_3
            subtest_name = "Pause inside Region1 ignored (no marker_api)"
            pass_regex = ["CodeBlock_A", "CodeBlock_C"]
            fail_regex = ["CodeBlock_D"]

        self.assert_perfetto(
            result,
            subtest_name=subtest_name,
            categories=["rocm_kernel_dispatch"],
            pass_regex=pass_regex,
            fail_regex=fail_regex,
        )


# =============================================================================
# Test Class: Selective Region without marker_api (ConditionB only)
# =============================================================================


@pytest.mark.parametrize("mode", ["sys_run", "sampling"])
@pytest.mark.class_name("selective-region-no-marker")
class TestSelectiveRegionNoMarker(RocprofsysTest):
    """Tests for region filtering with ConditionB only (no marker_api).

    Region filtering works via ROCPROFSYS_SELECTED_REGIONS even without marker_api.
    Pause/resume is IGNORED.

    Code flow:
        CodeBlock_A (outside),
        Region1: CodeBlock_B, Region2: CodeBlock_C, CodeBlock_D,
        Region3: CodeBlock_E,
        Region1: CodeBlock_F,
        CodeBlock_G (outside)
    """

    def test_region_1_filter(self, mode, no_marker_env):
        """ROCPROFSYS_SELECTED_REGIONS='Region1' without marker_api."""
        env = no_marker_env.copy()
        env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
        result = self.run_test(
            mode,
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Region1 filtered kernels (no marker_api)",
            categories=["rocm_kernel_dispatch"],
            pass_regex=["CodeBlock_B", "CodeBlock_C", "CodeBlock_D", "CodeBlock_F"],
            fail_regex=["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"],
        )
