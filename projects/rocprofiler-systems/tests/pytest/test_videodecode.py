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
    validate_rocpd_database,
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

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        video_decode_env: dict[str, str],
        gpu_info: GPUInfo,
        video_decode_rules: list[Path],
        use_rocpd: bool,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        env = video_decode_env.copy()
        if use_rocpd:
            env["ROCPROFSYS_USE_ROCPD"] = "ON"
            if gpu_info.is_mi300:
                rules_dir = (
                    rocprof_config.rocprofsys_tests_dir
                    / "rocpd-validation-rules"
                    / "video-decode"
                )
                video_decode_rules.append(rules_dir / "amd-smi-rules.json")

        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="videodecode",
                output_dir=test_output_dir,
                env=env,
                run_args=[
                    "-i",
                    str(rocprof_config.rocprofsys_examples_dir / "videos"),
                    "-t",
                    "1",
                ],
            )
        except FileNotFoundError:
            pytest.skip("videodecode binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Video decode sampling failed: {result.test_output}")

        # Validate perfetto trace
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            if result.perfetto_file is None:
                pytest.fail(f"Perfetto trace not created")
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
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")

        # ROCpd validation
        with subtests.test("ROCpd validation"):
            if not use_rocpd:
                pytest.skip("ROCpd is not enabled")
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                pytest.fail(f"ROCpd database not created")
            existing_rules = [r for r in video_decode_rules if r.exists()]
            if not existing_rules:
                pytest.skip("No validation rules found")
            validation = validate_rocpd_database(
                rocpd_file,
                rocprof_config.rocprofsys_tests_dir,
                rules_files=existing_rules,
            )
            if not validation.is_valid:
                pytest.fail(f"ROCpd validation failed: {validation.message}")

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        video_decode_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target="videodecode",
                output_dir=test_output_dir,
                env=video_decode_env,
                run_args=[
                    "-i",
                    str(rocprof_config.rocprofsys_examples_dir / "videos"),
                    "-t",
                    "1",
                ],
            )
        except FileNotFoundError:
            pytest.skip("videodecode binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Video decode sys-run failed: {result.test_output}")
