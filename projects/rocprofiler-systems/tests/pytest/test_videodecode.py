# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    pytest.mark.rocm,
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
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,vcn_activity,mem_usage,gfx_clock,mem_clock",
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


@pytest.fixture
def require_video_data(rocprof_config) -> None:
    """Skip the test at runtime when the sample video data is not available.

    The videodecode example is always built when rocDecode is present, but the
    H26x sample videos are only shipped by test builds of ROCm. When they are
    missing there is nothing to decode, so skip instead of failing.
    """
    videos_dir = rocprof_config.rocprofsys_examples_dir / "videos"
    if not (videos_dir.is_dir() and any(videos_dir.iterdir())):
        pytest.skip(
            f"No rocDecode sample videos found in {videos_dir}; "
            "possibly built against a non-test build which doesn't have those files."
        )


@pytest.fixture
def require_rocdecode_support() -> None:
    """Skip if the rocDecode VA-API driver cannot be initialized on this system.

    rocDecode uses VA-API for hardware video decoding. On some GPUs (e.g.
    gfx1250 with the current software stack) the VA driver is present but
    exports no __vaDriverInit_* symbol, causing every decode attempt to abort
    with a runtime error. Check for the symbol before running the test and
    skip cleanly when absent. The exact symbol name varies by libva version
    (__vaDriverInit_<major>_<minor>), so any match is accepted.
    """
    import subprocess
    from pathlib import Path

    candidate_dirs = sorted(Path("/opt").glob("rocm*/lib/rocm_sysdeps/lib"))
    candidate_dirs = [Path("/opt/rocm/lib/rocm_sysdeps/lib")] + list(candidate_dirs)

    drv_path = None
    for d in candidate_dirs:
        candidate = d / "radeonsi_drv_video.so"
        if candidate.exists():
            drv_path = candidate
            break

    if drv_path is None:
        pytest.skip(
            "rocDecode VA-API driver (radeonsi_drv_video.so) not found; "
            "video decode tests require a ROCm build with VA-API support"
        )

    result = subprocess.run(
        ["nm", "-D", str(drv_path)], capture_output=True, text=True
    )
    # Check for any __vaDriverInit_<major>_<minor> symbol; the exact name
    # is constructed by libva from its compile-time VA_MAJOR/MINOR constants
    # and therefore varies across libva versions.
    has_va_init = any("__vaDriverInit" in line for line in result.stdout.splitlines())
    if not has_va_init:
        pytest.skip(
            f"{drv_path.name} exports no __vaDriverInit_* symbol; "
            "the VA-API driver is incompatible with the installed libva on this system"
        )


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
@pytest.mark.class_name("video-decode")
class TestVideoDecode(RocprofsysTest):
    @pytest.mark.timeout(120)
    def test(
        self,
        mode,
        video_decode_env,
        gpu_info,
        video_decode_rules,
        get_run_args,
        require_video_data,
        require_rocdecode_support,
    ):
        result = self.run_test(
            mode,
            "videodecode",
            env=video_decode_env,
            run_args=get_run_args,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                categories=["rocm_rocdecode_api"],
                labels=["rocDecCreateVideoParser"],
                counter_names=(
                    ["VCN Busy"] if "instinct" in gpu_info.categories else None
                ),
            )
            self.assert_rocpd(result, rules_files=video_decode_rules)
