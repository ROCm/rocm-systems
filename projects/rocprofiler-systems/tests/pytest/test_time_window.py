# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the trace time window example.
Equivalent to rocprof-sys-time-window-tests.cmake
"""

from __future__ import annotations
import pytest

pytestmark = [pytest.mark.time_window]

# ============================================================================
# Time Window Fixtures
# ============================================================================


@pytest.fixture
def time_window_env() -> dict[str, str]:
    """Environment variables for time window tests."""
    return {
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_VERBOSE": "2",
    }


# ============================================================================
# Test Class: Trace Time Window Tests
# ============================================================================


class TestTraceTimeWindow:

    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    def test_binary_rewrite(
        self,
        run_test,
        time_window_env: dict[str, str],
        assert_perfetto,
        assert_timemory,
        assert_regex,
    ):
        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})

        result = run_test(
            "binary_rewrite",
            target="trace-time-window",
            rewrite_args=self.REWRITE_ARGS,
            env=env,
            timeout=120,
        )

        assert_regex(result)
        assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )
        assert_perfetto(
            result,
            labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )

    def test_runtime_instrument(
        self,
        run_test,
        time_window_env: dict[str, str],
        assert_regex,
        assert_perfetto,
        assert_timemory,
    ):
        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})

        result = run_test(
            "runtime_instrument",
            target="trace-time-window",
            instrument_args=self.RUNTIME_ARGS,
            env=env,
            timeout=400,  # In xdist, it can take much longer
        )

        assert_regex(result)
        assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )
        assert_perfetto(
            result,
            categories=["host"],
            labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
            counts=[1, 1, 1, 1],
            depths=[0, 1, 1, 1],
            fail_regex=["outer_d"],  # time window should exclude this
        )


# ============================================================================
# Test Class: Trace Time Window Delay Tests
# ============================================================================


class TestTraceTimeWindowDelay:
    """Tests for trace time window with delay."""

    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    def test_binary_rewrite(
        self,
        run_test,
        time_window_env: dict[str, str],
        assert_perfetto,
        assert_timemory,
        assert_regex,
    ):
        env = time_window_env.copy()
        env.update(
            {
                "ROCPROFSYS_TRACE_DELAY": "0.75",
                "ROCPROFSYS_TRACE_DURATION": "0.75",
            }
        )
        result = run_test(
            "binary_rewrite",
            target="trace-time-window",
            rewrite_args=self.REWRITE_ARGS,
            env=env,
            timeout=120,
        )

        assert_regex(result)
        assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )
        assert_perfetto(
            result,
            categories=["host"],
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )

    def test_runtime_instrument(
        self,
        run_test,
        time_window_env: dict[str, str],
        assert_perfetto,
        assert_timemory,
        assert_regex,
    ):
        """Test trace time window delay with runtime instrumentation."""
        env = time_window_env.copy()
        env.update(
            {
                "ROCPROFSYS_TRACE_DELAY": "0.75",
                "ROCPROFSYS_TRACE_DURATION": "0.75",
            }
        )

        result = run_test(
            "runtime_instrument",
            target="trace-time-window",
            instrument_args=self.RUNTIME_ARGS,
            env=env,
        )

        assert_regex(result)
        assert_timemory(
            result,
            file_name="wall_clock.json",
            metric="wall_clock",
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )
        assert_perfetto(
            result,
            categories=["host"],
            labels=["outer_c", "outer_d"],
            counts=[1, 1],
            depths=[0, 0],
        )
