# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the jpegdecode example.
"""

from __future__ import annotations
import pytest
from pathlib import Path

pytestmark = [pytest.mark.gpu, pytest.mark.decode, pytest.mark.jpegdecode]

from rocprofsys import (
    GPUInfo,
    RocprofsysConfig,
)

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
def jpeg_decode_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for JPEG decode tests."""
    rules_dir = validation_rules_dir / "jpeg-decode"
    return [
        validation_rules_dir / "default-rules.json",
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# =============================================================================
# JPEG decode tests
# =============================================================================


class TestJPEGDecode:
    """Tests for the jpegdecode example."""

    @pytest.mark.rocpd("jpeg_decode_env")
    def test_sampling(
        self,
        run_test,
        rocprof_config: RocprofsysConfig,
        jpeg_decode_env: dict[str, str],
        gpu_info: GPUInfo,
        jpeg_decode_rules: list[Path],
        assert_regex,
        assert_perfetto,
        assert_rocpd,
    ):
        env = jpeg_decode_env.copy()
        if "instinct" in gpu_info.categories:
            rules_dir = rocprof_config.rocpd_validation_rules / "jpeg-decode"
            jpeg_decode_rules.append(rules_dir / "amd-smi-rules.json")

        result = run_test(
            "sampling",
            target="jpegdecode",
            env=env,
            timeout=120,
            run_args=[
                "-i",
                str(rocprof_config.rocprofsys_examples_dir / "images"),
                "-b",
                "32",
            ],
            no_check_target_arch=True,
        )

        assert_regex(result)
        assert_perfetto(
            result,
            categories=["rocm_rocjpeg_api"],
            labels=["rocJpegCreate"],
            counts=[1],
            depths=[1],
            counter_names=(
                ["JPEG Activity"] if "instinct" in gpu_info.categories else None
            ),
        )
        assert_rocpd(result, rules_files=jpeg_decode_rules)

    def test_sys_run(
        self,
        run_test,
        rocprof_config: RocprofsysConfig,
        jpeg_decode_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target="jpegdecode",
            env=jpeg_decode_env,
            timeout=120,
            run_args=[
                "-i",
                str(rocprof_config.rocprofsys_examples_dir / "images"),
                "-b",
                "32",
            ],
            no_check_target_arch=True,
        )

        assert_regex(result)
