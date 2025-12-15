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
Tests for the jpegdecode example.
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
# JPEG decode fixtures
# =============================================================================

@pytest.fixture
def jpeg_decode_env() -> dict[str, str]:
    """Environment variables for JPEG decode tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,jpeg_activity,mem_usage",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
    }

# =============================================================================
# JPEG decode tests
# =============================================================================

@pytest.mark.gpu
class TestJPEGDecode:
    """Tests for the jpegdecode example."""

    def test_jpeg_decode_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        jpeg_decode_env: dict[str, str],
        gpu_info: GPUInfo,
    ):
        """Test JPEG decode sampling."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="jpegdecode",
            output_dir=test_output_dir,
            env=jpeg_decode_env,
            run_args=["-i", str(rocprof_config.rocprofsys_examples_dir / "images"), "-b", "32"],
        )
        result = runner.run()
        assert result.success, f"JPEG decode sampling failed: {result.stderr}"

        # Validate perfetto trace
        assert result.perfetto_file is not None, "Perfetto trace not created"
        counter_names = ["JPEG Activity"] if gpu_info.is_mi300 else None
        validation = validate_perfetto_trace(
            trace_path=result.perfetto_file,
            tests_dir=rocprof_config.rocprofsys_tests_dir,
            categories=["rocm_rocjpeg_api"],
            labels=["rocJpegCreate"],
            counts=[1],
            depths=[1],
            counter_names=counter_names,
            print_output=True,
        )
        assert validation.is_valid, f"Perfetto validation failed: {validation.message}"

    def test_jpeg_decode_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        jpeg_decode_env: dict[str, str],
    ):
        """Test JPEG decode with rocprof-sys-run."""
        runner = SysRunRunner(
            config=rocprof_config,
            target="jpegdecode",
            output_dir=test_output_dir,
            env=jpeg_decode_env,
            run_args=["-i", str(rocprof_config.rocprofsys_examples_dir / "images"), "-b", "32"],
        )
        result = runner.run()
        assert result.success, f"JPEG decode sys-run failed: {result.stderr}"
