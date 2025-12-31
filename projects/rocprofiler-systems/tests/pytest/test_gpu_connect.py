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
Tests for GPU connectivity
"""

import pytest
from pathlib import Path

# =============================================================================
# GPU connectivity fixtures
# =============================================================================


@pytest.fixture
def gpu_connect_env() -> dict[str, str]:
    """Environment variables for GPU connectivity tests."""
    return {
        "ROCPROFSYS_TRACE_CACHED": "OFF",
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
