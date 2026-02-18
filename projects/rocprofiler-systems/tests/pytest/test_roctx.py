# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the ROCTx marker API integration with rocprofiler-systems.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu, pytest.mark.roctx, pytest.mark.ci_enable]

# =============================================================================
# ROCTx fixtures
# =============================================================================


@pytest.fixture
def roctx_env() -> dict[str, str]:
    """Environment variables for rocTX tests."""
    return {
        "ROCPROFSYS_TRACE_LEGACY": "ON",
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
# Test Class: ROCTx Tests
# ============================================================================


class TestROCTx(RocprofsysTest):
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

    @pytest.mark.parametrize("mode", ["baseline", "binary_rewrite", "sys_run"])
    def test(self, mode, roctx_env):
        result = self.run_test(
            mode,
            "roctx",
            env=roctx_env,
            rewrite_args=self.REWRITE_ARGS,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)

    @pytest.mark.ci_disable("assert_rocpd")
    @pytest.mark.rocpd("roctx_env")
    def test_sampling(
        self,
        roctx_env: dict[str, str],
        roctx_rules: list[Path],
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

        result = self.run_test(
            "sampling", target="roctx", env=env, check_target_arch=True, timeout=120
        )

        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto counter validation",
            categories=categories,
            labels=labels,
            counts=counts,
            depths=depths,
        )
        self.assert_rocpd(
            result,
            rules_files=roctx_rules,
        )
