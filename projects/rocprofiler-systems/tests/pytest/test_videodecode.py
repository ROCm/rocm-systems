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
def require_rocdecode_support(rocprof_config) -> None:
    """Skip if the rocDecode VA-API driver cannot be initialized on this system.

    rocDecode uses VA-API for hardware video decoding. On some GPUs the VA
    driver is present but libva cannot load it — either because the driver
    exports no __vaDriverInit_* symbol, or because it exports a version
    (__vaDriverInit_1_22) that does not match what the installed libva expects
    (__vaDriverInit_1_0). Either way, every decode attempt aborts at runtime.

    A static symbol check (nm -D) is insufficient because the expected symbol
    name is determined by the installed libva version at runtime, not by the
    driver's exported symbols. Instead, probe by running the videodecode binary
    without LD_PRELOAD against a single video file and checking the output for
    known VA-API initialisation failure patterns. The probe completes quickly on
    both supported and unsupported systems.
    """
    import subprocess
    import os
    import shutil
    import tempfile

    videodecode_bin = rocprof_config.rocprofsys_examples_dir / "videodecode"
    if not videodecode_bin.exists():
        return  # Binary missing; let the test fail naturally

    videos_dir = rocprof_config.rocprofsys_examples_dir / "videos"
    if not videos_dir.is_dir():
        return  # No videos dir; require_video_data fixture will skip
    mp4_files = list(videos_dir.glob("*.mp4"))
    if not mp4_files:
        return  # No videos; require_video_data fixture will skip

    # Strip LD_PRELOAD so rocprofiler-systems does not interfere with the probe.
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}

    # Copy ALL available videos into the probe directory so that we catch
    # mixed-codec failure scenarios. Some codecs (H.264) may decode fine on a
    # given GPU while others (H.265, AV1) fail. Running with a single file
    # could miss the failing codec entirely if that file happened to succeed.
    # With all files present, H.265 is processed (printing the __vaDriverInit
    # error and skipping the file) before the AV1 file triggers an uncaught
    # C++ exception; subprocess.run() captures the full output before the crash.
    with tempfile.TemporaryDirectory() as tmpdir:
        for mp4 in mp4_files:
            shutil.copy2(str(mp4), tmpdir)
        try:
            result = subprocess.run(
                [str(videodecode_bin), "-i", tmpdir, "-t", "1"],
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return  # Hanging; let the test surface the problem

    combined = result.stdout + result.stderr
    # Known failure patterns that indicate the driver stack cannot support
    # video decode on this GPU:
    #   "has no function __vaDriverInit" — libva cannot find the VA-API init symbol
    #   "vaInitialize failed"            — VA-API init failed for any reason
    #   "terminate called"               — uncaught C++ exception from rocDecode
    va_errors = (
        "has no function __vaDriverInit",
        "vaInitialize failed",
        "terminate called",
    )
    if any(err in combined for err in va_errors):
        pytest.skip(
            "rocDecode VA-API initialization failed or the GPU does not support "
            "one or more required video codecs; "
            "video decode is not fully supported by the current driver stack for this GPU."
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
