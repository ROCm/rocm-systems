# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
rocprof-sys-sample against real HIP workloads, e.g.:

    rocprof-sys-sample --preset=trace-gpu -- ./transpose
    rocprof-sys-sample --rocm=hip,kernel  -- ./transpose
    rocprof-sys-sample --gpu=temp,power   -- ./roctx

CLI tokens go in sampling_args (before --). Needs a GPU.
These cases focus on GPU trace content from the rocprof-sys-sample path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.sample_workloads,
    pytest.mark.gpu,
    pytest.mark.rocm,
    pytest.mark.ci_enable,  # TODO: Deprecate once TheRock switches to CTest
]

# Small transpose workload (<iterations> <M> <N>) to keep CLI+GPU runs fast.
_TRANSPOSE_ARGS = ["2", "100", "50"]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def workload_trace_env() -> dict[str, str]:
    # The rocpd marker sets ROCPROFSYS_USE_ROCPD on this dict before it reaches
    # run_test; the workload-trace preset already turns on rocpd output.
    return {}


@pytest.fixture
def default_rules(validation_rules_dir: Path) -> list[Path]:
    # The transpose runs here are tiny, so use the workload-agnostic rules.
    return [validation_rules_dir / "default-rules.json"]


# =============================================================================
# CLI + GPU recipe cases
# =============================================================================

# (sampling_args, target, run_args, categories)
# categories = Perfetto categories to check, or None to skip the trace check.
WORKLOAD_CASES = [
    # --preset=trace-gpu -- ./transpose
    pytest.param(
        ["--preset=trace-gpu"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api", "kernel_dispatch"],
        id="preset-trace-gpu-transpose",
    ),
    # --preset=trace-hpc -- ./transpose
    pytest.param(
        ["--preset=trace-hpc"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api"],
        id="preset-trace-hpc-transpose",
    ),
    # --preset=sys-trace -- ./transpose   (broadest API tracing)
    pytest.param(
        ["--preset=sys-trace"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api", "kernel_dispatch", "memory_copy"],
        id="preset-sys-trace-transpose",
    ),
    # --rocm=hip,kernel -- ./transpose
    pytest.param(
        ["--rocm=hip,kernel"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api", "kernel_dispatch"],
        id="rocm-hip-kernel-transpose",
    ),
    # --gpu=temp,power -- ./roctx   (run-only, no trace check)
    pytest.param(
        ["--gpu=temp,power"],
        "roctx",
        [],
        None,
        id="gpu-temp-power-roctx",
    ),
    # --preset=trace-hpc --gpu=temp --gpus=0 -- ./transpose
    pytest.param(
        ["--preset=trace-hpc", "--gpu=temp", "--gpus=0"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api"],
        id="compose-trace-hpc-gpu-transpose",
    ),
    # --rocm=hip,kernel -- ./scratch-memory
    pytest.param(
        ["--rocm=hip,kernel"],
        "scratch-memory",
        [],
        ["hip_runtime_api", "kernel_dispatch"],
        id="rocm-hip-kernel-scratch-memory",
    ),
    # --rocm=hip,kernel,memory -- ./transpose
    pytest.param(
        ["--rocm=hip,kernel,memory"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api", "kernel_dispatch", "memory_copy"],
        id="rocm-hip-kernel-memory-transpose",
    ),
    # --rocm=hip,marker -- ./roctx
    pytest.param(
        ["--rocm=hip,marker"],
        "roctx",
        [],
        ["rocm_marker_api"],
        id="rocm-marker-roctx",
    ),
    # --cpu=100 -- ./transpose   (CPU domain on its own, run-only)
    pytest.param(
        ["--cpu=100"],
        "transpose",
        _TRANSPOSE_ARGS,
        None,
        id="cpu-sampling-transpose",
    ),
    # --rocm=hip,kernel --cpu=100 -- ./transpose
    pytest.param(
        ["--rocm=hip,kernel", "--cpu=100"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api", "kernel_dispatch"],
        id="compose-rocm-cpu-transpose",
    ),
    # --preset=runtime-trace -- ./transpose
    pytest.param(
        ["--preset=runtime-trace"],
        "transpose",
        _TRANSPOSE_ARGS,
        ["hip_runtime_api"],
        id="preset-runtime-trace-transpose",
    ),
]


@pytest.mark.class_name("sample-workloads")
class TestSampleWorkloads(RocprofsysTest):
    """rocprof-sys-sample CLI recipes against real HIP binaries."""

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "sampling_args, target, run_args, categories", WORKLOAD_CASES
    )
    def test(self, sampling_args, target, run_args, categories):
        result = self.run_test(
            "sampling",
            target=target,
            sampling_args=sampling_args,
            run_args=run_args,
            check_target_arch=True,
        )
        self.assert_regex(result)
        if categories is not None:
            self.assert_perfetto(result, categories=categories)

    @pytest.mark.timeout(120)
    def test_profile_only(self):
        """--preset=profile-only emits a flat wall_clock profile and no trace."""
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["--preset=profile-only"],
            run_args=_TRANSPOSE_ARGS,
            check_target_arch=True,
        )
        # Clean run, and with tracing off no perfetto trace should be written.
        self.assert_regex(result, fail_regex=[r"\[perfetto\]> Outputting"])
        # The flat profile is still produced.
        self.assert_timemory(result, file_name="wall_clock.json", metric="wall_clock")

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("workload_trace_env")
    def test_trace_rocpd(self, workload_trace_env, default_rules):
        """--preset=workload-trace produces a GPU trace and a rocpd database."""
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["--preset=workload-trace"],
            run_args=_TRANSPOSE_ARGS,
            env=workload_trace_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(result, categories=["hip_runtime_api"])
        self.assert_rocpd(result, rules_files=default_rules)
