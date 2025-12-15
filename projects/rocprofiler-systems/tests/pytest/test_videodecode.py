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
    SamplingRunner,
    SysRunRunner,
    validate_perfetto_trace,
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

# =============================================================================
# Video decode tests
# =============================================================================

@pytest.mark.gpu
class TestVideoDecode:
    """Tests for the videodecode example."""

    def test_video_decode_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        video_decode_env: dict[str, str],
        gpu_info: GPUInfo,
    ):
        """Test video decode sampling."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="videodecode",
            output_dir=test_output_dir,
            env=video_decode_env,
            run_args=["-i", str(rocprof_config.rocprofsys_examples_dir / "videos"), "-t", "1"],
        )
        result = runner.run()
        assert result.success, f"Video decode sampling failed: {result.stderr}"

        # Validate perfetto trace
        assert result.perfetto_file is not None, "Perfetto trace not created"
        counter_names = ["VCN Activity"] if gpu_info.is_mi300 else None
        validation = validate_perfetto_trace(
            trace_path=result.perfetto_file,
            tests_dir=rocprof_config.rocprofsys_tests_dir,
            categories=["rocm_rocdecode_api"],
            labels=["rocDecCreateVideoParser"],
            counts=[2],
            depths=[1],
            counter_names=counter_names,
            print_output=True,
        )
        assert validation.is_valid, f"Perfetto validation failed: {validation.message}"

    def test_jpeg_decode_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        video_decode_env: dict[str, str],
    ):
        """Test video decode with rocprof-sys-run."""
        runner = SysRunRunner(
            config=rocprof_config,
            target="videodecode",
            output_dir=test_output_dir,
            env=video_decode_env,
            run_args=["-i", str(rocprof_config.rocprofsys_examples_dir / "videos"), "-t", "1"],
        )
        result = runner.run()
        assert result.success, f"Video decode sys-run failed: {result.stderr}"
