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



@pytest.fixture
def require_rocjpeg_support(rocprof_config) -> None:
    """Skip if the rocJPEG VA-API decoder cannot be initialized on this system.

    rocJPEG uses VA-API for hardware JPEG decoding. On some GPUs (e.g.
    gfx1036 with certain TheRock builds) the VA-API driver is present and
    exports an __vaDriverInit_* symbol, yet rocJpegCreate() still returns
    ROCJPEG_STATUS_NOT_INITIALIZED because the driver does not expose JPEG
    decode support for that GPU architecture. When the binary is run under
    rocprofiler-systems sampling (LD_PRELOAD) in this state, the partial
    VA surface initialisation causes a SIGSEGV in vlVaHandleSurfaceAllocate.

    To detect this early, probe by running the jpegdecode binary without
    LD_PRELOAD against a single-image temporary directory. An empty directory
    triggers a division-by-zero (SIGFPE) in batch setup before rocJpegCreate
    is reached, so at least one image must be present. The probe completes in
    well under a second on both supported and unsupported systems.
    """
    import subprocess
    import os
    import shutil
    import tempfile

    jpegdecode_bin = rocprof_config.rocprofsys_examples_dir / "jpegdecode"
    if not jpegdecode_bin.exists():
        return  # Binary missing; let the test fail naturally

    # Strip LD_PRELOAD so we get a clean initialization check without
    # rocprofiler-systems interference.
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}

    # The probe needs at least one image: with an empty directory the binary
    # crashes with SIGFPE (division by zero in batch setup) before ever
    # reaching the rocJpegCreate() call that would print NOT_INITIALIZED.
    images_dir = rocprof_config.rocprofsys_examples_dir / "images"
    first_jpg = next(images_dir.glob("*.jpg"), None) if images_dir.is_dir() else None
    if first_jpg is None:
        return  # No images available; require_jpeg_data will skip

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy2(str(first_jpg), tmpdir)
        try:
            result = subprocess.run(
                [str(jpegdecode_bin), "-i", tmpdir, "-b", "1"],
                capture_output=True,
                text=True,
                timeout=15,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return  # Initialization is hanging; let the test surface that

    combined = result.stdout + result.stderr
    if "NOT_INITIALIZED" in combined:
        pytest.skip(
            "rocJPEG VA-API decoder returned ROCJPEG_STATUS_NOT_INITIALIZED; "
            "JPEG decode is not supported by the current VA-API driver for this GPU. "
            f"(probe exit code: {result.returncode})"
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
        require_rocjpeg_support,
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
