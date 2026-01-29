# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for GPU connectivity
"""

from __future__ import annotations
import pytest
from pathlib import Path

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


@pytest.mark.gpu
@pytest.mark.xgmi
@pytest.mark.run_if_gpu_category("not apu or instinct")
class TestGPUConnect:
    """Tests for GPU connectivity tests."""

    @pytest.mark.rocpd("gpu_connect_env")
    def test_sys_run(
        self,
        run_test,
        gpu_connect_env: dict[str, str],
        gpu_connect_rules: list[Path],
        assert_regex,
        assert_perfetto,
        assert_rocpd,
    ):
        result = run_test(
            "sys_run",
            target="transferBench",
            env=gpu_connect_env,
            timeout=120,
        )

        # Determine whether to skip or not
        if "Error: No valid transfers created" in result.test_output:
            pytest.skip("No valid transfers created")
        else:
            assert_regex(result)
            assert_perfetto(
                result,
                counter_names=["XGMI Read Data", "XGMI Write Data"],
            )
            assert_rocpd(result, rules_files=gpu_connect_rules)
