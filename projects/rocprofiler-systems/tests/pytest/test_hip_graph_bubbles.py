# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the HIP graph capture example.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import RocprofsysTest
from rocprofsys import GPUInfo

pytestmark = [
    pytest.mark.hip_graph_bubbles,
    pytest.mark.gpu,
    pytest.mark.ci_enable,
]

# Configurations mirror the upstream rocprofiler-sdk hip-graph-bubbles tests
# (projects/rocprofiler-sdk/tests/rocprofv3/hip-graph-bubbles-test/CMakeLists.txt):
# (suffix, num_kernels, num_iterations, array_size, progress_interval, timeout).
_HIP_GRAPH_BUBBLES_CONFIGS = [
    ("k256-i20", "256", "20", "256", "25", 60),
    ("k256-i200", "256", "200", "256", "25", 120),
    # ("k2000-i200", "2000", "200", "256", "25", 180), # takes too long for CI
]


def _hip_graph_bubbles_rocm_events(gpu_info: GPUInfo) -> str:
    # non-instinct (e.g. Navi) targets -> SQ_WAVES; otherwise full counter set.
    if "instinct" in gpu_info.categories:
        return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU"
    return "SQ_WAVES"


@pytest.fixture
def hip_graph_bubbles_env(gpu_info: GPUInfo) -> dict[str, str]:
    """Environment variables for HIP graph tests (ROCPD is enabled for sys_run via
    the @pytest.mark.rocpd marker, which sets ROCPROFSYS_USE_ROCPD=ON on this dict)."""
    return {
        "ROCPROFSYS_ROCM_EVENTS": _hip_graph_bubbles_rocm_events(gpu_info),
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch",
        "ROCPROFSYS_USE_AMD_SMI": "OFF",
    }


@pytest.fixture
def hip_graph_bubbles_rules(validation_rules_dir: Path) -> list[Path]:
    """ROCPD rules aligned with tests/rocpd-validation-rules/hip-graph-bubbles/."""
    return [
        validation_rules_dir / "default-rules.json",
        validation_rules_dir / "hip-graph-bubbles" / "graph-bubbles-rules.json",
    ]


@pytest.mark.class_name("hip-graph-bubbles")
class TestHipGraphBubbles(RocprofsysTest):
    """Exercise hip-graph-bubbles under rocprofiler-systems.

    Runs each configuration in two modes:
      * baseline - run the executable directly (no instrumentation)
      * sys_run  - run under rocprof-sys-run with ROCPD output validated
    """

    @pytest.mark.rocpd("hip_graph_bubbles_env")
    @pytest.mark.parametrize("mode", ["baseline", "sys_run"])
    @pytest.mark.parametrize(
        "suffix,num_kernels,num_iterations,array_size,progress_interval,timeout",
        _HIP_GRAPH_BUBBLES_CONFIGS,
    )
    def test(
        self,
        mode: str,
        suffix: str,
        num_kernels: str,
        num_iterations: str,
        array_size: str,
        progress_interval: str,
        timeout: int,
        request: pytest.FixtureRequest,
        hip_graph_bubbles_env: dict[str, str],
        hip_graph_bubbles_rules: list[Path],
    ):
        request.node.add_marker(pytest.mark.timeout(timeout))
        result = self.run_test(
            mode,
            "hip-graph-bubbles",
            env=hip_graph_bubbles_env,
            run_args=[num_kernels, num_iterations, array_size, progress_interval],
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=[r"Test completed successfully"],
            fail_regex=[r"HIP error"],
        )
        if mode == "sys_run":
            self.assert_rocpd(result, rules_files=hip_graph_bubbles_rules)
