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
Tests for the videodecode example.
"""

import pytest

from rocprofsys import (
    GPUInfo,
    RocprofsysConfig,
)

from pathlib import Path

# =============================================================================
# Video decode fixtures
# =============================================================================


@pytest.fixture
def video_decode_env() -> dict[str, str]:
    """Environment variables for video decode tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,rocdecode_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,vcn_activity,mem_usage",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
    }


@pytest.fixture
def video_decode_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for video decode tests."""
    rules_dir = validation_rules_dir / "video-decode"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# =============================================================================
# Video decode tests
# =============================================================================


@pytest.mark.gpu
class TestVideoDecode:
    """Tests for the videodecode example."""

    @pytest.mark.rocpd("video_decode_env")
    def test_sampling(
        self,
        run_test,
        rocprof_config: RocprofsysConfig,
        video_decode_env: dict[str, str],
        gpu_info: GPUInfo,
        video_decode_rules: list[Path],
        assert_rocpd,
        assert_perfetto,
        assert_regex,
    ):
        env = video_decode_env.copy()
        if env.get("ROCPROFSYS_USE_ROCPD") == "ON" and "instinct" in gpu_info.categories:
            rules_dir = rocprof_config.rocprofsys_validation_rules / "video-decode"
            video_decode_rules.append(rules_dir / "amd-smi-rules.json")

        result = run_test(
            "sampling",
            target="videodecode",
            env=env,
            timeout=120,
            run_args=[
                "-i",
                str(rocprof_config.rocprofsys_examples_dir / "videos"),
                "-t",
                "1",
            ],
        )

        assert_regex(result)
        assert_perfetto(
            result,
            categories=["rocm_rocdecode_api"],
            labels=["rocDecCreateVideoParser"],
            counts=[2],
            depths=[1],
            counter_names=["VCN Activity"] if "instinct" in gpu_info.categories else None,
        )
        assert_rocpd(result, rules_files=video_decode_rules)

    def test_sys_run(
        self,
        run_test,
        rocprof_config: RocprofsysConfig,
        video_decode_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target="videodecode",
            env=video_decode_env,
            timeout=120,
            run_args=[
                "-i",
                str(rocprof_config.rocprofsys_examples_dir / "videos"),
                "-t",
                "1",
            ],
        )

        assert_regex(result)
