# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the trace time window example.
"""

from __future__ import annotations
from pathlib import Path
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.time_window,
    pytest.mark.ci_enable,  # TODO: Deprecate once TheRock switches to CTest
]

# ============================================================================
# Time Window Fixtures
# ============================================================================


@pytest.fixture
def time_window_env() -> dict[str, str]:
    """Environment variables for time window tests."""
    return {
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
    }


# ============================================================================
# Test Class: Trace Time Window Tests
# ============================================================================


@pytest.mark.class_name("trace-time-window")
class TestTraceTimeWindow(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param("binary_rewrite", marks=pytest.mark.timeout(120)),
            pytest.param("runtime_instrument", marks=pytest.mark.timeout(300)),
        ],
    )
    def test(self, mode, time_window_env):

        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})
        result = self.run_test(
            mode,
            "trace-time-window",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
        )
        self.assert_regex(result)

        if mode == "binary_rewrite":
            label_name = "trace-time-window.inst"
        else:
            label_name = "trace-time-window"
        self.assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=[label_name, "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )
        self.assert_perfetto(
            result,
            categories=["host"],
            labels=[label_name, "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param("binary_rewrite", marks=pytest.mark.timeout(120)),
            pytest.param("runtime_instrument", marks=pytest.mark.timeout(300)),
        ],
    )
    def test_delay(self, mode, time_window_env):
        env = time_window_env.copy()
        env.update(
            {"ROCPROFSYS_TRACE_DELAY": "0.75", "ROCPROFSYS_TRACE_DURATION": "0.75"}
        )
        result = self.run_test(
            mode,
            "trace-time-window",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
        )
        self.assert_regex(result)
        self.assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )
        self.assert_perfetto(
            result,
            categories=["host"],
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )


# ============================================================================
# Test Class: TRACE_DELAY with HIP workload (regression for tool_init source gate)
# ============================================================================


@pytest.mark.transpose
@pytest.mark.gpu
class TestTraceDelayHip(RocprofsysTest):
    """Regression coverage: ROCPROFSYS_TRACE_DELAY must halt GPU producers at
    source so the cached Perfetto/RocPD output has zero kernel_dispatch
    records during the delay window. Pre-fix, tool_init unconditionally
    started the SDK contexts during init, so the cached path captured every
    kernel_dispatch even though the direct push was suppressed by the
    trace_categories trait flip.

    Workload: transpose (~3s on a typical run). DELAY=4.0 just exceeds the
    workload runtime, so a passing fix produces zero kernel records and a
    regression produces the full set."""

    REWRITE_ARGS = ["-e", "-v", "2", "-E", "uniform_int_distribution"]
    RUNTIME_ARGS = ["-e", "-v", "1", "-E", "uniform_int_distribution"]

    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    def test_delay_with_hip(self, mode, validation_rules_dir):
        env = {
            "ROCPROFSYS_TRACE_DELAY": "4.0",
            "ROCPROFSYS_ROCM_DOMAINS": (
                "hip_runtime_api,kernel_dispatch,memory_copy,"
                "memory_allocation,hsa_api"
            ),
        }
        result = self.run_test(
            mode,
            "transpose",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            rewrite_timeout=120,
            runtime_timeout=300,
        )
        self.assert_regex(result)
        self.assert_rocpd(
            result,
            rules_files=[
                validation_rules_dir
                / "transpose-trace-delay"
                / "validation-rules.json",
            ],
        )
