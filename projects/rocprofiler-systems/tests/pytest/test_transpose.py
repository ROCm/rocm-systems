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
)


# ============================================================================
# Test Class: Basic Transpose Tests
# ============================================================================

@pytest.mark.gpu
class TestTranspose:
    """Basic transpose tests with all instrumentation modes."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "--print-instructions",
        "-E", "uniform_int_distribution",
    ]

    RUNTIME_ARGS = [
        "-e",
        "-v", "1",
        "--label", "file", "line", "return", "args",
        "-E", "uniform_int_distribution",
    ]

    def test_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose runs successfully without instrumentation."""
        runner = BaselineRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"Baseline failed: {result.stderr}"

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with sampling instrumentation."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"Sampling failed: {result.stderr}"
        assert result.output_dir.exists(), "Output directory not created"

        # Verify perfetto trace was created
        perfetto = result.perfetto_file
        assert perfetto is not None, "Perfetto trace not created"
        assert perfetto.stat().st_size > 0, "Perfetto trace is empty"

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with binary rewrite instrumentation."""
        runner = BinaryRewriteRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            rewrite_args=self.REWRITE_ARGS,
            env=transpose_env,
            timeout=120,
        )

        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"
        assert runner.instrumented_exe.exists(), "Instrumented binary not created"

        result = runner.run()
        assert result.success, f"Run failed: {result.stderr}"

        perfetto = result.perfetto_file
        assert perfetto is not None, "Perfetto trace not created"

    def test_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with runtime instrumentation."""
        runner = RuntimeInstrumentRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            instrument_args=self.RUNTIME_ARGS,
            env=transpose_env,
            timeout=480,  # Runtime instrumentation is slower
        )

        result = runner.run()

        assert result.success, f"Runtime instrument failed: {result.stderr}"
        assert result.perfetto_file is not None, "Perfetto trace not created"

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with rocprof-sys-run wrapper."""
        runner = SysRunRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=300,
        )

        result = runner.run()

        assert result.success, f"sys-run failed: {result.stderr}"
        assert result.output_dir.exists(), "Output directory not created"


# ============================================================================
# Test Class: Two Kernels Configuration
# ============================================================================


@pytest.mark.gpu
class TestTransposeTwoKernels:
    """Test transpose with two kernels configuration (1 iteration, 2x2 size)."""

    RUN_ARGS = ["1", "2", "2"]

    def test_sampling_two_kernels(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with minimal kernel configuration."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"Two kernels test failed: {result.stderr}"

    def test_sys_run_two_kernels(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose two kernels with sys-run."""
        runner = SysRunRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=300,
        )

        result = runner.run()

        assert result.success, f"sys-run two kernels failed: {result.stderr}"


# ============================================================================
# Test Class: Loop Instrumentation
# ============================================================================


@pytest.mark.gpu
@pytest.mark.loops
class TestTransposeLoops:
    """Test transpose with loop instrumentation."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "--label", "return", "args",
        "-l",
        "-i", "8",
        "-E", "uniform_int_distribution",
    ]

    RUN_ARGS = ["2", "100", "50"]

    def test_binary_rewrite_loops(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose with loop instrumentation via binary rewrite."""
        runner = BinaryRewriteRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            rewrite_args=self.REWRITE_ARGS,
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )

        # Check rewrite phase
        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

        # Verify loops were instrumented (not 0)
        assert "0 instrumented loops in procedure transpose" not in rewrite_result.stdout, \
            "No loops were instrumented in transpose function"

        # Run the instrumented binary
        result = runner.run()
        assert result.success, f"Run failed: {result.stderr}"

    def test_sampling_loops(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Test transpose loops configuration with sampling."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"Sampling loops failed: {result.stderr}"


# ============================================================================
# Test Class: ROCProfiler Counter Collection
# ============================================================================


@pytest.mark.gpu
@pytest.mark.rocprofiler
class TestTransposeROCProfiler:
    """Test transpose with ROCProfiler counter collection."""

    REWRITE_ARGS = [
        "-e",
        "-v", "2",
        "-E", "uniform_int_distribution",
    ]

    @pytest.fixture
    def rocprofiler_env(
        self, transpose_env: dict[str, str], gpu_info: GPUInfo
    ) -> dict[str, str]:
        """Environment with ROCm events configured."""
        env = transpose_env.copy()
        env["ROCPROFSYS_ROCM_EVENTS"] = gpu_info.rocm_events_for_test
        return env

    def test_sampling_rocprofiler(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
    ):
        """Test transpose with ROCProfiler counters via sampling."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=rocprofiler_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"ROCProfiler sampling failed: {result.stderr}"

        for expected_file in gpu_info.expected_counter_files:
            file_path = result.output_dir / expected_file
            assert file_path.exists(), f"Counter file not found: {expected_file}"

    def test_binary_rewrite_rocprofiler(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
    ):
        """Test transpose with ROCProfiler counters via binary rewrite."""
        runner = BinaryRewriteRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            rewrite_args=self.REWRITE_ARGS,
            env=rocprofiler_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"ROCProfiler rewrite failed: {result.stderr}"

        for expected_file in gpu_info.expected_counter_files:
            file_path = result.output_dir / expected_file
            assert file_path.exists(), f"Counter file not found: {expected_file}"

    def test_validate_perfetto_counters(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
    ):
        """Validate ROCProfiler counters in perfetto trace."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=rocprofiler_env,
            timeout=120,
        )

        result = runner.run()
        assert result.success, f"Test execution failed: {result.stderr}"

        if result.perfetto_file:
            validation = validate_perfetto_trace(
                result.perfetto_file,
                counter_names=gpu_info.counter_names,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
            )
            assert validation.is_valid, f"Perfetto validation failed: {validation.message}"


# ============================================================================
# Test Class: ROCpd Database Validation
# ============================================================================


@pytest.mark.gpu
@pytest.mark.rocpd
class TestTransposeROCpd:
    """Test transpose with ROCpd database output and validation."""

    @pytest.fixture
    def transpose_rules(self, validation_rules_dir: Path) -> list[Path]:
        """Get validation rules files for transpose tests."""
        rules_dir = validation_rules_dir / "transpose"
        return [
            validation_rules_dir / "default-rules.json",
            rules_dir / "validation-rules.json",
            rules_dir / "amd-smi-rules.json",
            rules_dir / "cpu-metrics-rules.json",
            rules_dir / "timer-sampling-rules.json",
            rules_dir / "sdk-metrics-rules.json",
        ]

    def test_sampling_rocpd(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocpd_env: dict[str, str],
    ):
        """Test transpose with ROCpd output enabled."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=rocpd_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, f"ROCpd sampling failed: {result.stderr}"

        rocpd_file = result.rocpd_file
        assert rocpd_file is not None, "ROCpd database not created"
        assert rocpd_file.stat().st_size > 0, "ROCpd database is empty"

    def test_validate_rocpd_database(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocpd_env: dict[str, str],
        transpose_rules: list[Path],
    ):
        """Validate ROCpd database contents for transpose."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=rocpd_env,
            timeout=120,
        )

        result = runner.run()
        assert result.success, f"Test execution failed: {result.stderr}"

        rocpd_file = result.rocpd_file
        assert rocpd_file is not None, "ROCpd database not created"

        existing_rules = [r for r in transpose_rules if r.exists()]


        validation = validate_rocpd_database(
            rocpd_file,
            rocprof_config.rocprofsys_tests_dir,
            rules_files=existing_rules,
        )

        assert validation.is_valid, f"ROCpd validation failed: {validation.message}"


# ============================================================================
# Test Class: Perfetto Trace Validation
# ============================================================================


@pytest.mark.gpu
class TestTransposePerfettoValidation:
    """Validate perfetto trace contents for transpose tests."""

    def test_perfetto_has_kernel_dispatch(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Verify kernel dispatch events in perfetto trace."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()
        assert result.success, f"Test failed: {result.stderr}"

        perfetto_file = result.perfetto_file
        assert perfetto_file is not None, "Perfetto trace not created"

        # Validate trace has kernel dispatch events
        validation = validate_perfetto_trace(
            perfetto_file,
            rocprof_config.rocprofsys_tests_dir,
            categories=["kernel_dispatch"],
        )

        assert validation.is_valid, f"Perfetto validation failed: {validation.message}"
        assert validation.details is not None
        assert validation.details.get("slice_count", 0) > 0, \
            "No kernel dispatch events found in trace"

    def test_perfetto_has_hip_api_calls(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
    ):
        """Verify HIP API calls in perfetto trace."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()
        assert result.success, f"Test failed: {result.stderr}"

        perfetto_file = result.perfetto_file
        assert perfetto_file is not None, "Perfetto trace not created"

        # Validate trace has HIP runtime API events
        validation = validate_perfetto_trace(
            perfetto_file,
            categories=["hip_runtime_api"],
            tests_dir=rocprof_config.rocprofsys_tests_dir,
        )

        assert validation.is_valid, f"Perfetto validation failed: {validation.message}"


# ============================================================================
# Parametrized Tests
# ============================================================================


@pytest.mark.gpu
class TestTransposeParametrized:
    """Parametrized tests for various transpose configurations."""

    @pytest.mark.parametrize(
        "iterations,tile_dim,block_rows",
        [
            (1, 16, 16),
            (2, 32, 32),
            (5, 64, 64),
        ],
        ids=["small", "medium", "large"],
    )
    def test_transpose_configurations(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        iterations: int,
        tile_dim: int,
        block_rows: int,
    ):
        """Test transpose with different iteration and tile configurations."""
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            run_args=[str(iterations), str(tile_dim), str(block_rows)],
            env=transpose_env,
            timeout=120,
        )

        result = runner.run()

        assert result.success, (
            f"Config ({iterations}, {tile_dim}, {block_rows}) failed: {result.stderr}"
        )

    @pytest.mark.parametrize(
        "runner_class,runner_kwargs",
        [
            (SamplingRunner, {}),
            (SysRunRunner, {}),
        ],
        ids=["sampling", "sys-run"],
    )
    def test_instrumentation_modes(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        runner_class,
        runner_kwargs: dict,
    ):
        """Test different instrumentation modes produce valid output."""
        runner = runner_class(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=transpose_env,
            timeout=120,
            **runner_kwargs,
        )

        result = runner.run()

        assert result.success, f"{runner_class.__name__} failed: {result.stderr}"
        assert result.output_dir.exists(), "Output directory not created"
