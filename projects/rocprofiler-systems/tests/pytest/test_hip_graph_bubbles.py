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


@pytest.fixture
def hip_graph_bubbles_env() -> dict[str, str]:
    """Environment variables for HIP graph + marker tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,marker_api",
        "ROCPROFSYS_USE_AMD_SMI": "OFF",
    }


@pytest.fixture
def hip_graph_bubbles_rocpd_env(
    hip_graph_bubbles_env: dict[str, str], gpu_info: GPUInfo
) -> dict[str, str]:
    """Same as CMake ROCPD run: enable PMC collection for rocpd_pmc_event checks."""
    env = hip_graph_bubbles_env.copy()
    env["ROCPROFSYS_ROCM_EVENTS"] = gpu_info.rocm_events_for_test
    return env


@pytest.fixture
def hip_graph_bubbles_rules(validation_rules_dir: Path) -> list[Path]:
    """ROCPD rules aligned with tests/rocpd-validation-rules/hip-graph-bubbles/."""
    return [
        validation_rules_dir / "default-rules.json",
        validation_rules_dir / "hip-graph-bubbles" / "graph-bubbles-rules.json",
    ]


class TestHipGraphBubbles(RocprofsysTest):
    """Exercise hip-graph-bubbles under rocprofiler-systems (no binary rewrite / runtime)."""

    @pytest.mark.ci_disable("assert_rocpd")
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
            timeout=300,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=[r"Test completed successfully"],
            fail_regex=[r"HIP error"],
        )
        if mode == "sampling":
            self.assert_rocpd(result, rules_files=hip_graph_bubbles_rules)
