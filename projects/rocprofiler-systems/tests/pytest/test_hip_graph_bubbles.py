# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the HIP graph capture example with rocTX ranges.
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

# Must match RUN_ARGS in tests/rocprof-sys-hip-graph-bubbles-tests.cmake. If changed, update
# tests/rocpd-validation-rules/hip-graph-bubbles/graph-bubbles-rules.json (kernel count = K * I,
# graph_launch region count = I).
_HIP_GRAPH_BUBBLES_NUM_KERNELS = "64"
_HIP_GRAPH_BUBBLES_NUM_ITERATIONS = "6"

# Match tests/rocprof-sys-hip-graph-bubbles-tests.cmake (REWRITE / SAMPLING / SYS_RUN timeout).
_HIP_GRAPH_BUBBLES_RUN_TIMEOUT_SEC = 900


def _hip_graph_bubbles_rocm_events(gpu_info: GPUInfo) -> str:
    # non-instinct (e.g. Navi) targets -> SQ_WAVES; otherwise full counter set.
    if "instinct" in gpu_info.categories:
        return "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU"
    return "SQ_WAVES"


@pytest.fixture
def hip_graph_bubbles_env(gpu_info: GPUInfo) -> dict[str, str]:
    """Environment variables for HIP graph + marker tests (ROCM_EVENTS always set, like CMake)."""
    return {
        "ROCPROFSYS_ROCM_EVENTS": _hip_graph_bubbles_rocm_events(gpu_info),
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,marker_api",
        "ROCPROFSYS_USE_AMD_SMI": "OFF",
    }


@pytest.fixture
def hip_graph_bubbles_rocpd_env(
    hip_graph_bubbles_env: dict[str, str],
) -> dict[str, str]:
    """Copy for sampling + ROCPD (apply_rocpd_marker adds ROCPROFSYS_USE_ROCPD=ON)."""
    return hip_graph_bubbles_env.copy()


@pytest.fixture
def hip_graph_bubbles_rules(validation_rules_dir: Path) -> list[Path]:
    """ROCPD rules aligned with tests/rocpd-validation-rules/hip-graph-bubbles/."""
    return [
        validation_rules_dir / "default-rules.json",
        validation_rules_dir / "hip-graph-bubbles" / "graph-bubbles-rules.json",
    ]


class TestHipGraphBubbles(RocprofsysTest):
    """Exercise hip-graph-bubbles under rocprofiler-systems (no binary rewrite / runtime)."""

    @pytest.mark.rocpd("hip_graph_bubbles_rocpd_env")
    @pytest.mark.parametrize("mode", ["baseline", "sampling", "sys_run"])
    def test(
        self,
        mode: str,
        hip_graph_bubbles_env: dict[str, str],
        hip_graph_bubbles_rocpd_env: dict[str, str],
        hip_graph_bubbles_rules: list[Path],
    ):
        """Baseline / sampling / sys_run; sampling also validates ROCPD (graph-bubbles-rules.json)."""
        result = self.run_test(
            mode,
            "hip-graph-bubbles",
            env=(
                hip_graph_bubbles_rocpd_env
                if mode == "sampling"
                else hip_graph_bubbles_env
            ),
            run_args=[_HIP_GRAPH_BUBBLES_NUM_KERNELS, _HIP_GRAPH_BUBBLES_NUM_ITERATIONS],
            check_target_arch=True,
            baseline_timeout=_HIP_GRAPH_BUBBLES_RUN_TIMEOUT_SEC,
            sampling_timeout=_HIP_GRAPH_BUBBLES_RUN_TIMEOUT_SEC,
            sys_run_timeout=_HIP_GRAPH_BUBBLES_RUN_TIMEOUT_SEC,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=[r"Test completed successfully"],
            fail_regex=[r"HIP error"],
        )
        if mode == "sampling":
            self.assert_rocpd(result, rules_files=hip_graph_bubbles_rules)
