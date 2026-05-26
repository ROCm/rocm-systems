# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Live (legacy direct-mode) Perfetto canary on the transpose example."""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu, pytest.mark.hip]


@pytest.fixture
def live_perfetto_env() -> dict[str, str]:
    # ROCPROFSYS_TRACE_LEGACY=true switches Perfetto from cached-buffer
    # post-processing to direct (live) SDK emission. Profiling / sampling
    # off so the trace contains only the slices the canary asserts on.
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_TRACE_LEGACY": "true",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
    }


@pytest.mark.class_name("live-perfetto-canary")
class TestLivePerfettoCanary(RocprofsysTest):
    run_args = ["2", "100"]

    @pytest.mark.timeout(120)
    def test(self, live_perfetto_env):
        result = self.run_test(
            "sys_run",
            target="transpose",
            env=live_perfetto_env,
            run_args=self.run_args,
        )

        # Unconditional structural check: the .proto must exist and be
        # non-trivially populated. Catches "live engine produced no output"
        # even when the perfetto python module isn't available (in which
        # case assert_perfetto silently SUBSKIPs).
        assert (
            result.perfetto_file.exists()
        ), f"live perfetto trace not produced at {result.perfetto_file}"
        assert (
            result.perfetto_file.stat().st_size > 1024
        ), f"live perfetto trace suspiciously small: {result.perfetto_file.stat().st_size} bytes"

        # transpose 2 100 always dispatches 200 transpose_a kernels +
        # 4 amd_rocclr_fillBufferAligned setup kernels, each carrying a
        # debug.kernel_id annotation; 204 is a fixed contract of the
        # workload + live emission path.
        self.assert_perfetto(
            result,
            subtest_name="Live perfetto: kernel-dispatch debug annotations",
            key_names=["kernel_id"],
            key_counts=[204],
        )

        # rocm_hip_stream depth-0 slices are the GPU activity records.
        # Substrings must be listed in emission order (the validator
        # does positional prefix matching).
        self.assert_perfetto(
            result,
            subtest_name="Live perfetto: rocm_hip_stream GPU activity",
            categories=["rocm_hip_stream"],
            label_substrings=[
                "fillBufferAligned",
                "MEMORY_COPY_HOST_TO_DEVICE",
                "transpose_a",
                "MEMORY_COPY_DEVICE_TO_HOST",
            ],
            counts=[4, 24, 200, 24],
            depths=[0, 0, 0, 0],
        )
