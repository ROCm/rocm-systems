# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the jpegdecode example.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.decode,
    pytest.mark.jpegdecode,
    pytest.mark.rocm,
]


# =============================================================================
# JPEG decode fixtures
# =============================================================================


@pytest.fixture
def jpeg_decode_env() -> dict[str, str]:
    """Environment variables for JPEG decode tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,rocjpeg_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,jpeg_activity,mem_usage,gfx_clock,mem_clock",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
    }


@pytest.fixture
def jpeg_decode_rules(validation_rules_dir, gpu_info) -> list[Path]:
    """Get validation rules for JPEG decode tests."""
    rules_dir = validation_rules_dir / "jpeg-decode"
    # gfx1250 uses VCN hardware-accelerated JPEG decode which
    # dispatches few or no HIP compute kernels and produces fewer API trace
    # entries than the compute-based path on older GPUs.
    if gpu_info._is_gfx1250:
        validation_rules = rules_dir / "gfx1250-validation-rules.json"
        sdk_metrics_rules = rules_dir / "gfx1250-sdk-metrics-rules.json"
    else:
        validation_rules = rules_dir / "validation-rules.json"
        sdk_metrics_rules = rules_dir / "sdk-metrics-rules.json"
    rules = [
        validation_rules_dir / "default-rules.json",
        validation_rules,
        sdk_metrics_rules,
    ]
    if "instinct" in gpu_info.categories:
        rules.append(rules_dir / "amd-smi-rules.json")
    return rules


@pytest.fixture
def get_run_args(rocprof_config) -> list[str]:
    """Get run arguments for JPEG decode tests."""
    return ["-i", str(rocprof_config.rocprofsys_examples_dir / "images"), "-b", "32"]


@pytest.fixture
def require_jpeg_data(rocprof_config) -> None:
    """Skip the test at runtime when the sample image data is not available.

    The jpegdecode example is always built when rocJPEG is present, but the
    sample images are only shipped by test builds of ROCm. When they are missing
    there is nothing to decode, so skip instead of failing.
    """
    images_dir = rocprof_config.rocprofsys_examples_dir / "images"
    if not (images_dir.is_dir() and any(images_dir.iterdir())):
        pytest.skip(
            f"No rocJPEG sample images found in {images_dir}; "
            "possibly built against a non-test build which doesn't have those files."
        )



# =============================================================================
# JPEG decode tests
# =============================================================================


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("sampling", marks=pytest.mark.rocpd("jpeg_decode_env")),
        "sys_run",
    ],
)
@pytest.mark.class_name("jpeg-decode")
class TestJPEGDecode(RocprofsysTest):
    def test(
        self,
        mode,
        jpeg_decode_env,
        jpeg_decode_rules,
        get_run_args,
        gpu_info,
        require_jpeg_data,
    ):
        result = self.run_test(
            mode,
            "jpegdecode",
            env=jpeg_decode_env,
            run_args=get_run_args,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                categories=["rocm_rocjpeg_api"],
                labels=["rocJpegCreate"],
                counts=[1],
                depths=[1],
                counter_names=(
                    ["JPEG Busy"] if "instinct" in gpu_info.categories else None
                ),
            )
            self.assert_rocpd(result, rules_files=jpeg_decode_rules)
