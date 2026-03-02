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
from conftest import RocprofsysTest

pytestmark = [pytest.mark.transpose, pytest.mark.gpu, pytest.mark.ci_enable]

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


@pytest.mark.mpi_optional("transpose")
class TestTranspose(RocprofsysTest):
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
    TWO_KERNELS_RUN_ARGS = ["1", "2", "2"]
    LOOPS_REWRITE_ARGS = [
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
    LOOPS_RUN_ARGS = ["2", "100", "50"]

    @pytest.mark.parametrize(
        "mode", ["baseline", "binary_rewrite", "runtime_instrument", "sys_run"]
    )
    def test(self, mode, transpose_env, num_processes):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            check_target_arch=True,
            mpi_ranks=num_processes,
        )
        self.assert_regex(result)
        if mode != "baseline":
            self.assert_perfetto(result)

    @pytest.mark.rocpd("transpose_env")
    def test_sampling(self, transpose_env, transpose_rules, num_processes):
        result = self.run_test(
            "sampling",
            target="transpose",
            env=transpose_env,
            check_target_arch=True,
            timeout=120,
            mpi_ranks=num_processes,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API Call Validation",
            categories=["hip_runtime_api"],
        )
        self.assert_rocpd(result, rules_files=transpose_rules)

    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    def test_two_kernels(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            run_args=self.TWO_KERNELS_RUN_ARGS,
            check_target_arch=True,
            sampling_timeout=120,
            sys_run_timeout=300,
        )
        self.assert_regex(result)

    @pytest.mark.loops
    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test_loops(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            rewrite_args=self.LOOPS_REWRITE_ARGS,
            run_args=self.LOOPS_RUN_ARGS,
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_fail_regex=["0 instrumented loops in procedure transpose"],
        )

    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    @pytest.mark.parametrize(
        "iterations,tile_dim,block_rows",
        [
            (1, 16, 16),
            (2, 32, 32),
            (5, 64, 64),
        ],
    )
    def test_parametrized(self, mode, iterations, tile_dim, block_rows, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            run_args=[str(iterations), str(tile_dim), str(block_rows)],
            fail_message=f"Config ({iterations}, {tile_dim}, {block_rows}) failed",
            check_target_arch=True,
            timeout=120,
        )
        self.assert_regex(result)


# ============================================================================
# Test Class: ROCProfiler Counter Collection
# ============================================================================


@pytest.mark.mpi_optional("transpose")
@pytest.mark.rocprofiler
@pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
class TestTransposeROCProfiler(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "-E", "uniform_int_distribution"]

    def test(self, mode, rocprofiler_env, gpu_info, num_processes):
        result = self.run_test(
            mode,
            "transpose",
            env=rocprofiler_env,
            check_target_arch=True,
            timeout=120,
            mpi_ranks=num_processes,
        )
        self.assert_regex(result)
        counter_files = [result.output_dir / f for f in gpu_info.expected_counter_files]
        self.assert_file_exists(
            counter_files, subtest_name="ROCProfiler counter files existence validation"
        )
        if mode == "sampling":
            self.assert_perfetto(
                result,
                subtest_name="Perfetto counter validation",
                counter_names=gpu_info.counter_names,
            )
