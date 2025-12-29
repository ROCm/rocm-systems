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

import sys
from pathlib import Path

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

# =============================================================================
# Transpose fixtures
# =============================================================================


@pytest.fixture
def transpose_env() -> dict[str, str]:
    """Environment variables for transpose tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,hsa_api"
    }


@pytest.fixture
def rocprofiler_env(transpose_env: dict[str, str], gpu_info: GPUInfo) -> dict[str, str]:
    """Environment with ROCm events configured."""
    env = transpose_env.copy()
    env["ROCPROFSYS_ROCM_EVENTS"] = gpu_info.rocm_events_for_test
    return env


@pytest.fixture
def transpose_rules(validation_rules_dir: Path) -> list[Path]:
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


# ============================================================================
# Test Class: Basic Transpose Tests
# ============================================================================


@pytest.mark.gpu
class TestTranspose:
    """Basic transpose tests with all instrumentation modes."""

    REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "--print-instructions",
        "-E",
        "uniform_int_distribution",
    ]

    RUNTIME_ARGS = [
        "-e",
        "-v",
        "1",
        "--label",
        "file",
        "line",
        "return",
        "args",
        "-E",
        "uniform_int_distribution",
    ]

    def test_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Baseline failed: {result.test_output}")

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        transpose_rules: list[Path],
        use_rocpd: bool,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        env = transpose_env.copy()
        if use_rocpd:
            env["ROCPROFSYS_USE_ROCPD"] = "ON"

        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                env=env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Sampling failed: {result.test_output}")
        if not result.output_dir.exists():
            pytest.fail(f"Output directory not created")

        # Verify perfetto trace was created
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                pytest.fail(f"Perfetto trace not created")
            if perfetto.stat().st_size == 0:
                pytest.fail(f"Perfetto trace is empty")

        # Verify perfetto trace has HIP runtime API events
        with subtests.test("Perfetto HIP API Call Validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                pytest.fail(f"Perfetto trace not created")
            # Validate trace has HIP runtime API events
            validation = validate_perfetto_trace(
                perfetto,
                categories=["hip_runtime_api"],
                tests_dir=rocprof_config.rocprofsys_tests_dir,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")

        # ROCpd validation
        with subtests.test("ROCpd validation"):
            if not use_rocpd:
                pytest.skip("ROCpd is not enabled")
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                pytest.fail(f"ROCpd database not created")
            existing_rules = [r for r in transpose_rules if r.exists()]
            if not existing_rules:
                pytest.skip("No validation rules found")
            validation = validate_rocpd_database(
                rocpd_file,
                rocprof_config.rocprofsys_tests_dir,
                rules_files=existing_rules,
            )
            if not validation.is_valid:
                pytest.fail(f"ROCpd validation failed: {validation.message}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Binary rewrite failed: {result.test_output}")

        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                pytest.fail(f"Perfetto trace not created")

    def test_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                instrument_args=self.RUNTIME_ARGS,
                env=transpose_env,
                timeout=480,  # Runtime instrumentation is slower
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Runtime instrument failed: {result.test_output}")

        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                pytest.fail(f"Perfetto trace not created")

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                env=transpose_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"sys-run failed: {result.test_output}")
        if not result.output_dir.exists():
            pytest.fail(f"Output directory not created")


# ============================================================================
# Test Class: Two Kernels Configuration
# ============================================================================


@pytest.mark.gpu
class TestTransposeTwoKernels:
    """Test transpose with two kernels configuration (1 iteration, 2x2 size)."""

    RUN_ARGS = ["1", "2", "2"]

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                run_args=self.RUN_ARGS,
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Two kernels test failed: {result.test_output}")

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                run_args=self.RUN_ARGS,
                env=transpose_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"sys-run two kernels failed: {result.test_output}")


# ============================================================================
# Test Class: Loop Instrumentation
# ============================================================================


@pytest.mark.gpu
@pytest.mark.loops
class TestTransposeLoops:
    """Test transpose with loop instrumentation."""

    REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "--label",
        "return",
        "args",
        "-l",
        "-i",
        "8",
        "-E",
        "uniform_int_distribution",
    ]

    RUN_ARGS = ["2", "100", "50"]

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                run_args=self.RUN_ARGS,
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Sampling loops failed: {result.test_output}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        transpose_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                run_args=self.RUN_ARGS,
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Binary rewrite failed: {result.test_output}")

        # Verify loops were instrumented (not 0)
        if "0 instrumented loops in procedure transpose" in result.test_output:
            pytest.fail(f"No loops were instrumented in transpose function")


# ============================================================================
# Test Class: ROCProfiler Counter Collection
# ============================================================================


@pytest.mark.gpu
@pytest.mark.rocprofiler
class TestTransposeROCProfiler:
    """Test transpose with ROCProfiler counter collection."""

    REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "-E",
        "uniform_int_distribution",
    ]

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                env=rocprofiler_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"ROCProfiler sampling failed: {result.test_output}")

        for expected_file in gpu_info.expected_counter_files:
            file_path = result.output_dir / expected_file
            if not file_path.exists():
                pytest.fail(f"Counter file not found: {expected_file}")

        with subtests.test("Validate Perfetto Counters"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto = result.perfetto_file
            if perfetto is None:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                perfetto,
                counter_names=gpu_info.counter_names,
                tests_dir=rocprof_config.rocprofsys_tests_dir,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
        collect_result,
    ):
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=rocprofiler_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Binary rewrite test failed: {result.test_output}")

        for expected_file in gpu_info.expected_counter_files:
            file_path = result.output_dir / expected_file
            if not file_path.exists():
                pytest.fail(f"Counter file not found: {expected_file}")


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
        collect_result,
    ):
        """Test transpose with different iteration and tile configurations."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                run_args=[str(iterations), str(tile_dim), str(block_rows)],
                env=transpose_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(
                f"Config ({iterations}, {tile_dim}, {block_rows}) failed: {result.test_output}"
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
        collect_result,
    ):
        """Test different instrumentation modes produce valid output."""
        try:
            runner = runner_class(
                config=rocprof_config,
                target="transpose",
                output_dir=test_output_dir,
                env=transpose_env,
                timeout=120,
                **runner_kwargs,
            )
        except FileNotFoundError:
            pytest.skip("transpose binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"{runner_class.__name__} failed: {result.test_output}")
        if not result.output_dir.exists():
            pytest.fail(f"Output directory not created")
