# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
    pytest.mark.ci_enable,
]


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


@pytest.fixture
def jpeg_decode_rules(validation_rules_dir, gpu_info) -> list[Path]:
    """Get validation rules for JPEG decode tests."""
    rules_dir = validation_rules_dir / "jpeg-decode"
    rules = [
        validation_rules_dir / "default-rules.json",
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]
    if "instinct" in gpu_info.categories:
        rules.append(rules_dir / "amd-smi-rules.json")
    return rules


@pytest.fixture
def get_run_args(rocprof_config) -> list[str]:
    """Get run arguments for JPEG decode tests."""
    return ["-i", str(rocprof_config.rocprofsys_examples_dir / "images"), "-b", "32"]


# =============================================================================
# JPEG decode tests
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("sampling", marks=pytest.mark.rocpd("jpeg_decode_env")),
        "sys_run",
    ],
)
class TestJPEGDecode(RocprofsysTest):
    def test(self, mode, jpeg_decode_env, jpeg_decode_rules, get_run_args, gpu_info):
        result = self.run_test(
            mode,
            "jpegdecode",
            env=jpeg_decode_env,
            run_args=get_run_args,
            timeout=120,
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
                    ["JPEG Activity"] if "instinct" in gpu_info.categories else None
                ),
            )
            self.assert_rocpd(result, rules_files=jpeg_decode_rules)
