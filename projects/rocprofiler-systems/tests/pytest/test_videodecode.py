# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the videodecode example.
"""

from __future__ import annotations
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.decode, pytest.mark.videodecode]

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
        if "instinct" in gpu_info.categories:
            rules_dir = rocprof_config.rocpd_validation_rules / "video-decode"
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
            no_check_target_arch=True,
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
            no_check_target_arch=True,
        )

        assert_regex(result)
