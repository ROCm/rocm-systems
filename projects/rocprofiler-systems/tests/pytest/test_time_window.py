# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the trace time window example.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.time_window, pytest.mark.ci_enable]

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


class TestTraceTimeWindow(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    def test(self, mode, time_window_env):

        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})
        result = self.run_test(
            mode,
            "trace-time-window",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            rewrite_timeout=120,
            runtime_timeout=300,
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

    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
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
            rewrite_timeout=120,
            runtime_timeout=300,
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
