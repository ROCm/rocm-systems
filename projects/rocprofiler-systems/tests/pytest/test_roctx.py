# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests for the ROCTX marker API integration with rocprofiler-systems.
Equivalent to rocprof-sys-roctx-tests.cmake
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest


# =============================================================================
# rocTX fixtures
# =============================================================================


@pytest.fixture
def roctx_env() -> dict[str, str]:
    """Environment variables for rocTX tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "ON",
        "ROCPROFSYS_TRACE_CACHED": "OFF",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch",
    }


@pytest.fixture
def roctx_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for rocTX tests."""
    rules_dir = validation_rules_dir / "roctx"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "amd-smi-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# ============================================================================
# Test Class: rocTX Tests
# ============================================================================


@pytest.mark.gpu
class TestRoctx:
    """Tests for rocTX marker API."""

    def roctx_legacy_labels(self) -> list[str]:
        return [
            "roctxMark_GPU_workload",
            "roctxRangePush_run_profiling",
            "roctxRangeStart_GPU_Compute",
            "roctxRangeStart_GPU_Compute",
            "roctxRangePush_HIP_Kernel",
            "roctxRangePush_HIP_Kernel",
            "roctxGetThreadId",
            "roctxMark_RoctxProfilerPause_End",
            "roctxMark_Thread_Start",
            "roctxMark_End",
            "roctxMark_Finished_GPU",
        ]

    def roctx_legacy_count(self) -> list[int]:
        return [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def roctx_legacy_depth(self) -> list[int]:
        return [1, 1, 2, 0, 3, 1, 2, 2, 0, 0, 1]

    def roctx_cached_labels(self) -> list[str]:
        return [
            "roctxMark_GPU_workload",
            "roctxRangePush_HIP_Kernel",
            "roctxRangeStart_GPU_Compute",
            "roctxGetThreadId",
            "roctxMark_RoctxProfilerPause_End",
            "roctxMark_Thread_Start",
            "roctxMark_End",
            "roctxRangePush_run_profiling",
            "roctxMark_Finished_GPU",
        ]

    def roctx_cached_count(self) -> list[int]:
        return [1, 2, 2, 1, 1, 1, 1, 1, 1]

    def roctx_cached_depth(self) -> list[int]:
        return [1, 1, 1, 1, 1, 2, 1, 1, 1]

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_baseline(
        self,
        roctx_env: dict[str, str],
        run_test,
        assert_regex,
    ):
        result = run_test("baseline", target="roctx", env=roctx_env, timeout=120)
        assert_regex(result)

    @pytest.mark.rocpd("roctx_env")
    def test_sampling(
        self,
        run_test,
        roctx_env: dict[str, str],
        roctx_rules: list[Path],
        assert_regex,
        assert_perfetto,
        assert_rocpd,
    ):
        env = roctx_env.copy()
        categories = ["rocm_marker_api"]
        if env["ROCPROFSYS_TRACE_LEGACY"] == "ON":
            labels = self.roctx_legacy_labels()
            counts = self.roctx_legacy_count()
            depths = self.roctx_legacy_depth()
        else:
            labels = self.roctx_cached_labels()
            counts = self.roctx_cached_count()
            depths = self.roctx_cached_depth()

        result = run_test("sampling", target="roctx", env=env, timeout=120)

        assert_regex(result)
        assert_perfetto(
            result,
            subtest_name="Perfetto counter validation",
            categories=categories,
            labels=labels,
            counts=counts,
            depths=depths,
        )
        assert_rocpd(
            result,
            rules_files=roctx_rules,
        )

    def test_binary_rewrite(
        self,
        run_test,
        roctx_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "binary_rewrite",
            target="roctx",
            rewrite_args=self.REWRITE_ARGS,
            env=roctx_env,
            timeout=120,
        )
        assert_regex(result)

    def test_sys_run(
        self,
        run_test,
        roctx_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target="roctx",
            env=roctx_env,
            timeout=120,
        )
        assert_regex(result)
