# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
import pytest
from pathlib import Path

pytestmark = [pytest.mark.transpose, pytest.mark.gpu]

from rocprofsys import (
    GPUInfo,
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
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test("baseline", target="transpose", env=transpose_env, timeout=120)
        assert_regex(result)

    @pytest.mark.rocpd("transpose_env")
    def test_sampling(
        self,
        run_test,
        transpose_env: dict[str, str],
        transpose_rules: list[Path],
        assert_rocpd,
        assert_perfetto,
        assert_regex,
    ):
        result = run_test("sampling", target="transpose", env=transpose_env, timeout=120)
        if not result.output_dir.exists():
            pytest.fail(f"Output directory not created")

        assert_regex(result)
        assert_perfetto(
            result,
            subtest_name="Perfetto HIP API Call Validation",
            categories=["hip_runtime_api"],
        )
        assert_rocpd(result, rules_files=transpose_rules)

    def test_binary_rewrite(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_perfetto,
        assert_regex,
    ):
        result = run_test(
            "binary_rewrite",
            target="transpose",
            rewrite_args=self.REWRITE_ARGS,
            env=transpose_env,
            timeout=120,
        )

        assert_regex(result)
        assert_perfetto(result)

    def test_runtime_instrument(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_perfetto,
        assert_regex,
    ):
        result = run_test(
            "runtime_instrument",
            target="transpose",
            instrument_args=self.RUNTIME_ARGS,
            env=transpose_env,
            timeout=480,
        )
        assert_regex(result)
        assert_perfetto(result)

    def test_sys_run(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target="transpose",
            env=transpose_env,
            timeout=300,
        )
        assert_regex(result)


# ============================================================================
# Test Class: Two Kernels Configuration
# ============================================================================


class TestTransposeTwoKernels:
    """Test transpose with two kernels configuration (1 iteration, 2x2 size)."""

    RUN_ARGS = ["1", "2", "2"]

    def test_sampling(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sampling",
            target="transpose",
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )
        assert_regex(result)

    def test_sys_run(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sys_run",
            target="transpose",
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=300,
        )
        assert_regex(result)


# ============================================================================
# Test Class: Loop Instrumentation
# ============================================================================


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
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "sampling",
            target="transpose",
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )
        assert_regex(result)

    def test_binary_rewrite(
        self,
        run_test,
        transpose_env: dict[str, str],
        assert_regex,
    ):
        result = run_test(
            "binary_rewrite",
            target="transpose",
            rewrite_args=self.REWRITE_ARGS,
            run_args=self.RUN_ARGS,
            env=transpose_env,
            timeout=120,
        )
        assert_regex(result, fail_regex=["0 instrumented loops in procedure transpose"])


# ============================================================================
# Test Class: ROCProfiler Counter Collection
# ============================================================================


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
        run_test,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
        assert_perfetto,
        assert_regex,
        assert_file_exists,
    ):
        result = run_test(
            "sampling",
            target="transpose",
            env=rocprofiler_env,
            timeout=120,
        )

        assert_regex(result)
        counter_files = [result.output_dir / f for f in gpu_info.expected_counter_files]
        assert_file_exists(
            counter_files, subtest_name="ROCProfiler counter files existence validation"
        )
        assert_perfetto(
            result,
            subtest_name="Perfetto counter validation",
            counter_names=gpu_info.counter_names,
        )

    def test_binary_rewrite(
        self,
        run_test,
        rocprofiler_env: dict[str, str],
        gpu_info: GPUInfo,
        assert_file_exists,
        assert_regex,
    ):
        result = run_test(
            "binary_rewrite",
            target="transpose",
            rewrite_args=self.REWRITE_ARGS,
            env=rocprofiler_env,
            timeout=120,
        )

        assert_regex(result)
        counter_files = [result.output_dir / f for f in gpu_info.expected_counter_files]
        assert_file_exists(
            counter_files, subtest_name="ROCProfiler counter files existence validation"
        )


# ============================================================================
# Parametrized Tests
# ============================================================================


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
        run_test,
        transpose_env: dict[str, str],
        iterations: int,
        tile_dim: int,
        block_rows: int,
        assert_regex,
    ):
        """Test transpose with different iteration and tile configurations."""
        result = run_test(
            "sampling",
            target="transpose",
            run_args=[str(iterations), str(tile_dim), str(block_rows)],
            env=transpose_env,
            timeout=120,
            fail_message=f"Config ({iterations}, {tile_dim}, {block_rows}) failed",
        )
        assert_regex(result)

    @pytest.mark.parametrize(
        "runner_type,runner_kwargs",
        [
            ("sampling", {}),
            ("sys_run", {}),
        ],
        ids=["sampling", "sys-run"],
    )
    def test_instrumentation_modes(
        self,
        run_test,
        transpose_env: dict[str, str],
        runner_type: str,
        runner_kwargs: dict,
        assert_regex,
    ):
        """Test different instrumentation modes produce valid output."""
        result = run_test(
            runner_type,
            target="transpose",
            env=transpose_env,
            timeout=120,
            **runner_kwargs,
        )
        if not result.output_dir.exists():
            pytest.fail(f"Output directory not created")

        assert_regex(result)
