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
Tests for the transpose example.
Equivalent to rocprof-sys-rocm-tests.cmake
    Note: MPI is not yet supported

This module tests the transpose HIP example with various instrumentation modes:
- Baseline execution (no instrumentation)
- Sampling instrumentation
- Binary rewrite instrumentation
- Runtime instrumentation
- sys-run wrapper execution

It also validates outputs including:
- Perfetto traces
- ROCpd databases
- ROCProfiler counter data
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
    GPUInfo,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
    validate_perfetto_trace,
    validate_rocpd_database,
    validate_timemory_json,
)

@pytest.fixture
def time_window_env(base_env: dict[str, str]) -> dict[str, str]:
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

    def test_binary_rewrite(self, rocprof_config: RocprofsysConfig, test_output_dir: Path, time_window_env: dict[str, str]):
        """Test trace time window with binary rewrite instrumentation."""

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
            pytest.skip("trace-time-window target not built")

        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"
        assert runner.instrumented_exe.exists(), "Instrumented binary not created"

        result = runner.run()
        assert result.success, f"Run failed: {result.failure_reason}"

        # Validate timemory wall_clock.json
        timemory_file = result.output_dir / "wall_clock.json"
        if timemory_file.exists():
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            assert validation.is_valid, f"Timemory validation failed: {validation.message}"
            assert "outer_d" not in validation.stdout, "outer_d should not appear (time window should exclude it)"

        # Validate perfetto trace
        perfetto_file = result.perfetto_file
        if perfetto_file:
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["trace-time-window.inst", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            assert validation.is_valid, f"Perfetto validation failed: {validation.message}"
            assert "outer_d" not in validation.stdout, "outer_d should not appear (time window should exclude it)"

    def test_runtime_instrument(self, rocprof_config: RocprofsysConfig, test_output_dir: Path, time_window_env: dict[str, str]):
        """Test trace time window with runtime instrumentation."""

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
            pytest.skip("trace-time-window target not built")

        result = runner.run()
        assert result.success, f"Run failed: {result.failure_reason}"

        # Validate timemory wall_clock.json
        timemory_file = result.output_dir / "wall_clock.json"
        if timemory_file.exists():
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            assert validation.is_valid, f"Timemory validation failed: {validation.message}"
            assert "outer_d" not in validation.stdout, "outer_d should not appear (time window should exclude it)"

        # Validate perfetto trace
        perfetto_file = result.perfetto_file
        if perfetto_file:
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["trace-time-window", "outer_a", "outer_b", "outer_c"],
                counts=[1, 1, 1, 1],
                depths=[0, 1, 1, 1],
                print_output=True,
            )
            assert validation.is_valid, f"Perfetto validation failed: {validation.message}"
            assert "outer_d" not in validation.stdout, "outer_d should not appear (time window should exclude it)"


# ============================================================================
# Test Class: Trace Time Window Delay Tests
# ============================================================================


class TestTraceTimeWindowDelay:
    """Tests for trace time window with delay."""

    REWRITE_ARGS = ["-e", "-v", "2", "--caller-include", "inner", "-i", "4096"]
    RUNTIME_ARGS = ["-e", "-v", "1", "--caller-include", "inner", "-i", "4096"]

    def test_binary_rewrite(self, rocprof_config: RocprofsysConfig, test_output_dir: Path, time_window_env: dict[str, str]):
        """Test trace time window delay with binary rewrite instrumentation."""
        env = time_window_env.copy()
        env.update({
            "ROCPROFSYS_TRACE_DELAY": "0.75",
            "ROCPROFSYS_TRACE_DURATION": "0.75",
        })

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window target not built")

        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"
        assert runner.instrumented_exe.exists(), "Instrumented binary not created"

        result = runner.run()
        assert result.success, f"Run failed: {result.failure_reason}"

        # Validate timemory wall_clock.json
        timemory_file = result.output_dir / "wall_clock.json"
        if timemory_file.exists():
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            assert validation.is_valid, f"Timemory validation failed: {validation.message}"

        # Validate perfetto trace
        perfetto_file = result.perfetto_file
        if perfetto_file:
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            assert validation.is_valid, f"Perfetto validation failed: {validation.message}"

    def test_runtime_instrument(self, rocprof_config: RocprofsysConfig, test_output_dir: Path, time_window_env: dict[str, str]):
        """Test trace time window delay with runtime instrumentation."""
        env = time_window_env.copy()
        env.update({
            "ROCPROFSYS_TRACE_DELAY": "0.75",
            "ROCPROFSYS_TRACE_DURATION": "0.75",
        })

        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="trace-time-window",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=env,
            )
        except FileNotFoundError:
            pytest.skip("trace-time-window target not built")

        result = runner.run()
        assert result.success, f"Run failed: {result.failure_reason}"

        # Validate timemory wall_clock.json
        timemory_file = result.output_dir / "wall_clock.json"
        if timemory_file.exists():
            validation = validate_timemory_json(
                json_path=timemory_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                metric="wall_clock",
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            assert validation.is_valid, f"Timemory validation failed: {validation.message}"

        # Validate perfetto trace
        perfetto_file = result.perfetto_file
        if perfetto_file:
            validation = validate_perfetto_trace(
                trace_path=perfetto_file,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
                categories=["host"],
                labels=["outer_c", "outer_d"],
                counts=[1, 1],
                depths=[0, 0],
                print_output=True,
            )
            assert validation.is_valid, f"Perfetto validation failed: {validation.message}"
