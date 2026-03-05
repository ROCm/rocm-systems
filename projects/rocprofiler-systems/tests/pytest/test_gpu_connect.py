# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for GPU connectivity
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu, pytest.mark.xgmi, pytest.mark.ci_enable]

# =============================================================================
# GPU connectivity fixtures
# =============================================================================


@pytest.fixture
def gpu_connect_env() -> dict[str, str]:
    """Environment variables for GPU connectivity tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "ON",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,xgmi,pcie",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_PROCESS_SAMPLING_FREQ": "50",
        "ROCPROFSYS_CPU_FREQ_ENABLED": "OFF",
    }


@pytest.fixture
def gpu_connect_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for GPU connectivity tests."""
    rules_dir = validation_rules_dir / "gpu-connect"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "amd-smi-rules.json",
    ]


# =============================================================================
# GPU connectivity tests
# =============================================================================


@pytest.mark.run_if_gpu_category("not apu or instinct")
class TestGPUConnect(RocprofsysTest):
    """Tests for GPU connectivity tests."""

    @pytest.mark.parametrize(
        "mode", [pytest.param("sys_run", marks=pytest.mark.rocpd("gpu_connect_env"))]
    )
    def test_transferbench(self, mode, gpu_connect_env, gpu_connect_rules):
        result = self.run_test(
            mode,
            "transferBench",
            env=gpu_connect_env,
            check_target_arch=True,
            timeout=120,
        )
        if "Error: No valid transfers created" in result.test_output:
            pytest.skip("No valid transfers created")

        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_rocpd(result, rules_files=gpu_connect_rules)
            self.assert_perfetto(
                result,
                counter_names=["XGMI Read Data", "XGMI Write Data"],
            )
