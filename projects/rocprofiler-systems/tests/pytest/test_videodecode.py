# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the videodecode example.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.decode,
    pytest.mark.videodecode,
    pytest.mark.ci_enable,
]

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
def video_decode_rules(validation_rules_dir, gpu_info) -> list[Path]:
    """Get validation rules for video decode tests."""
    rules_dir = validation_rules_dir / "video-decode"
    rules = [
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]
    if "instinct" in gpu_info.categories:
        rules.append(rules_dir / "amd-smi-rules.json")
    return rules


@pytest.fixture
def get_run_args(rocprof_config) -> list[str]:
    return ["-i", str(rocprof_config.rocprofsys_examples_dir / "videos"), "-t", "1"]


# =============================================================================
# Video decode tests
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("sampling", marks=pytest.mark.rocpd("video_decode_env")),
        "sys_run",
    ],
)
class TestVideoDecode(RocprofsysTest):
    def test(self, mode, video_decode_env, gpu_info, video_decode_rules, get_run_args):
        result = self.run_test(
            mode,
            "videodecode",
            env=video_decode_env,
            run_args=get_run_args,
            timeout=120,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                categories=["rocm_rocdecode_api"],
                labels=["rocDecCreateVideoParser"],
                counts=[2],
                depths=[1],
                counter_names=(
                    ["VCN Activity"] if "instinct" in gpu_info.categories else None
                ),
            )
            self.assert_rocpd(result, rules_files=video_decode_rules)
