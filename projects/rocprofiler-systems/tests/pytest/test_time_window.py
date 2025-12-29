# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests for the trace time window example.
Equivalent to rocprof-sys-time-window-tests.cmake
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Type

# Add the pytest directory to Python path for rocprofsys package
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    validate_perfetto_trace,
    validate_timemory_json,
)

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
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        time_window_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Run failed: {result.failure_reason}")

        # Validate timemory wall_clock.json
        with subtests.test("Timemory validation"):
            timemory_file = result.output_dir / "wall_clock.json"
            if not timemory_file.exists():
                pytest.fail(f"Timemory file not found: {timemory_file}")
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Timemory validation failed: {validation.message}")
            if "outer_d" in validation.stdout:
                pytest.fail(f"outer_d should not appear (time window should exclude it)")

        # Validate perfetto trace
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if perfetto_file is None:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")
            if "outer_d" in validation.stdout:
                pytest.fail(f"outer_d should not appear (time window should exclude it)")

    def test_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        time_window_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        env = time_window_env.copy()
        env.update({"ROCPROFSYS_TRACE_DURATION": "1.25"})

        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Run failed: {result.failure_reason}")

        # Validate timemory wall_clock.json
        with subtests.test("Timemory validation"):
            timemory_file = result.output_dir / "wall_clock.json"
            if not timemory_file.exists():
                pytest.fail(f"Timemory file not found: {timemory_file}")
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Timemory validation failed: {validation.message}")
            if "outer_d" in validation.stdout:
                pytest.fail(f"outer_d should not appear (time window should exclude it)")

        # Validate perfetto trace
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if perfetto_file is None:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")
            if "outer_d" in validation.stdout:
                pytest.fail(f"outer_d should not appear (time window should exclude it)")


# ============================================================================
# Test Class: Trace Time Window Delay Tests
# ============================================================================


class TestTraceTimeWindowDelay:
    """Tests for trace time window with delay."""

    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        time_window_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        env = time_window_env.copy()
        env.update(
            {
                "ROCPROFSYS_TRACE_DELAY": "0.75",
                "ROCPROFSYS_TRACE_DURATION": "0.75",
            }
        )

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Run failed: {result.failure_reason}")

        # Validate timemory wall_clock.json
        with subtests.test("Timemory validation"):
            timemory_file = result.output_dir / "wall_clock.json"
            if not timemory_file.exists():
                pytest.fail(f"Timemory file not found: {timemory_file}")
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Timemory validation failed: {validation.message}")

        # Validate perfetto trace
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if perfetto_file is None:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")

    def test_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        time_window_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        """Test trace time window delay with runtime instrumentation."""
        env = time_window_env.copy()
        env.update(
            {
                "ROCPROFSYS_TRACE_DELAY": "0.75",
                "ROCPROFSYS_TRACE_DURATION": "0.75",
            }
        )

        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Run failed: {result.failure_reason}")

        # Validate timemory wall_clock.json
        with subtests.test("Timemory validation"):
            timemory_file = result.output_dir / "wall_clock.json"
            if not timemory_file.exists():
                pytest.fail(f"Timemory file not found: {timemory_file}")
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Timemory validation failed: {validation.message}")

        # Validate perfetto trace
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if perfetto_file is None:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")
