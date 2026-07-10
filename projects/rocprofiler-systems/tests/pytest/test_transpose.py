# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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

pytestmark = [
    pytest.mark.transpose,
    pytest.mark.gpu,
]

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
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,hsa_api",
        "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,mem_usage,gfx_clock,mem_clock",
    }


@pytest.fixture
def rocprofiler_env(transpose_env: dict[str, str], gpu_info: GPUInfo) -> dict[str, str]:
    """Environment with ROCm events configured."""
    env = transpose_env.copy()
    env["ROCPROFSYS_ROCM_EVENTS"] = gpu_info.rocm_events_for_test
    return env


@pytest.fixture
def gpu_perf_counter_env(
    transpose_env: dict[str, str], gpu_info: GPUInfo
) -> dict[str, str]:
    """Environment with GPU perf counters configured."""
    env = transpose_env.copy()
    env["ROCPROFSYS_GPU_PERF_COUNTERS"] = gpu_info.gpu_perf_counters_for_test
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


@pytest.fixture
def rocprofiler_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for GPU hardware counter RocPD output."""
    rules_dir = validation_rules_dir / "transpose"
    return [
        validation_rules_dir / "default-rules.json",
        rules_dir / "hw-counter-rules.json",
    ]


@pytest.fixture
def transpose_default_rules(validation_rules_dir: Path) -> list[Path]:
    return [validation_rules_dir / "default-rules.json"]


# CLI case presets/domain tests do not cover: real GPU trace + AMD-SMI counter track.
TRANSPOSE_SAMPLE_CASES = [
    pytest.param(
        ["--preset=trace-hpc", "--gpu=temp", "--gpus=0"],
        {
            "hip_labels": ["hipMalloc"],
            "counter_names": ["Temperature"],
        },
        id="compose-trace-hpc-gpu",
    ),
]


# ============================================================================
# Test Class: Basic Transpose Tests
# ============================================================================


@pytest.mark.timeout(120)
@pytest.mark.mpi_optional("transpose")
class TestTranspose(RocprofsysTest):
    BINARY_REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "--print-instructions",
        "-E",
        "uniform_int_distribution",
    ]
    RUNTIME_INSTRUMENT_ARGS = [
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
    LOOPS_BINARY_REWRITE_ARGS = [
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
    SAMPLING_RUN_ARGS = ["4", "500", "100"]
    SAMPLING_ENV = {
        "ROCPROFSYS_SAMPLING_REALTIME": "ON",
        "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "300",
        "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
    }
    AMD_SMI_COUNTER_NAMES = [
        "Temperature",
        "Avg. Power",
        "GFX Busy",
        "Memory Usage",
        "GFX Clock",
        "Memory Clock",
    ]
    HIP_API_CATEGORY = "rocm_hip_api"
    HIP_STREAM_CATEGORY = "rocm_hip_stream"
    MEMORY_ALLOCATE_CATEGORY = "rocm_memory_allocate"
    HIP_API_LABELS = [
        "hipMalloc",
        "hipMemsetAsync",
        "hipMemcpyAsync",
        "hipStreamSynchronize",
        "hipFree",
    ]
    KERNEL_LABEL_SUBSTRINGS = ["transpose_a"]
    MEMORY_COPY_LABELS = [
        "MEMORY_COPY_HOST_TO_DEVICE",
        "MEMORY_COPY_DEVICE_TO_HOST",
    ]
    MEMORY_ALLOCATE_LABELS = ["ALLOCATE"]
    PROFILE_ONLY_RUN_ARGS = ["2", "100", "50"]

    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            "binary_rewrite",
            "runtime_instrument",
            "sys_run",
        ],
    )
    def test(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)
        if mode != "baseline":
            self.assert_perfetto(
                result,
                subtest_name="Perfetto HIP API",
                categories=[self.HIP_API_CATEGORY],
                labels=self.HIP_API_LABELS[:2],
            )
            self.assert_perfetto(
                result,
                subtest_name="Perfetto kernel dispatch",
                categories=[self.HIP_STREAM_CATEGORY],
                label_substrings=self.KERNEL_LABEL_SUBSTRINGS,
            )

    @pytest.mark.sampling
    @pytest.mark.rocpd("transpose_env")
    def test_sampling(self, transpose_env, transpose_rules):
        env = transpose_env.copy()
        env.update(self.SAMPLING_ENV)
        result = self.run_test(
            "sampling",
            target="transpose",
            env=env,
            run_args=self.SAMPLING_RUN_ARGS,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API",
            categories=[self.HIP_API_CATEGORY],
            labels=self.HIP_API_LABELS,
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto kernel dispatch",
            categories=[self.HIP_STREAM_CATEGORY],
            label_substrings=self.KERNEL_LABEL_SUBSTRINGS,
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto memory copy",
            categories=[self.HIP_STREAM_CATEGORY],
            labels=self.MEMORY_COPY_LABELS,
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto memory allocate",
            categories=[self.MEMORY_ALLOCATE_CATEGORY],
            labels=self.MEMORY_ALLOCATE_LABELS,
        )
        self.assert_perfetto(
            result,
            subtest_name="Perfetto AMD-SMI counters",
            counter_names=self.AMD_SMI_COUNTER_NAMES,
        )
        self.assert_rocpd(result, rules_files=transpose_rules)

    @pytest.mark.sampling
    def test_profile_only(self):
        """--preset=profile-only emits a flat wall_clock profile and no trace."""
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["--preset=profile-only"],
            run_args=self.PROFILE_ONLY_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result, fail_regex=[r"\[perfetto\]> Outputting"])
        assert (
            result.perfetto_file is None
        ), f"profile-only wrote an unexpected trace: {result.perfetto_file}"
        self.assert_timemory(result, file_name="wall_clock.json", metric="wall_clock")

    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    def test_two_kernels(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            run_args=self.TWO_KERNELS_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result)

    @pytest.mark.loops
    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test_loops(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            binary_rewrite_args=self.LOOPS_BINARY_REWRITE_ARGS,
            run_args=self.LOOPS_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_fail_regex=["0 instrumented loops in procedure transpose"],
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
        )
        self.assert_regex(result)

    @pytest.mark.rocm_min_version("7.0")
    @pytest.mark.hip_stream
    @pytest.mark.parametrize("mode", ["sampling", "sys_run"])
    @pytest.mark.parametrize(
        "type",
        [
            pytest.param("group-by-queue", marks=pytest.mark.group_by_queue),
            pytest.param("group-by-stream", marks=pytest.mark.group_by_stream),
        ],
    )
    def test_hip_stream(self, mode, type):
        if type == "group-by-queue":
            env = {"ROCPROFSYS_ROCM_GROUP_BY_QUEUE": "YES"}
        else:
            env = {"ROCPROFSYS_ROCM_GROUP_BY_QUEUE": "NO"}

        result = self.run_test(
            mode,
            "transpose",
            env=env,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)


# ============================================================================
# Test Class: rocprof-sys-sample CLI on transpose
# ============================================================================


@pytest.mark.timeout(120)
@pytest.mark.sampling
@pytest.mark.class_name("transpose-sample-cli")
class TestTransposeSampleCLI(RocprofsysTest):
    """rocprof-sys-sample flags against the transpose HIP binary."""

    RUN_ARGS = ["2", "100", "50"]

    def _assert_sample_expectations(self, result, expect: dict) -> None:
        if "hip_labels" in expect:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto HIP API",
                categories=[TestTranspose.HIP_API_CATEGORY],
                labels=expect["hip_labels"],
            )
        if "kernel_substrings" in expect:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto kernel dispatch",
                categories=[TestTranspose.HIP_STREAM_CATEGORY],
                label_substrings=expect["kernel_substrings"],
            )
        if "memory_copy_labels" in expect:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto memory copy",
                categories=[TestTranspose.HIP_STREAM_CATEGORY],
                labels=expect["memory_copy_labels"],
            )
        if "counter_names" in expect:
            self.assert_perfetto(
                result,
                subtest_name="Perfetto AMD-SMI counters",
                counter_names=expect["counter_names"],
            )

    @pytest.mark.parametrize("sampling_args,expect", TRANSPOSE_SAMPLE_CASES)
    def test(self, sampling_args, expect):
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=sampling_args,
            run_args=self.RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self._assert_sample_expectations(result, expect)

    @pytest.mark.rocpd("transpose_env")
    def test_workload_trace(self, transpose_env, transpose_default_rules):
        """--preset=workload-trace produces a GPU trace and a rocpd database."""
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["--preset=workload-trace"],
            run_args=self.RUN_ARGS,
            env=transpose_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API",
            categories=[TestTranspose.HIP_API_CATEGORY],
            labels=TestTranspose.HIP_API_LABELS[:1],
        )
        self.assert_rocpd(result, rules_files=transpose_default_rules)


# ============================================================================
# Test Class: ROCProfiler Counter Collection
# ============================================================================


@pytest.mark.mpi_optional("transpose")
@pytest.mark.timeout(120)
@pytest.mark.rocprofiler
@pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
@pytest.mark.class_name("transpose-rocprofiler")
class TestTransposeROCProfiler(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "-E", "uniform_int_distribution"]

    @pytest.mark.rocpd("rocprofiler_env")
    def test(self, mode, rocprofiler_env, gpu_info, rocprofiler_rules):
        result = self.run_test(
            mode,
            "transpose",
            env=rocprofiler_env,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
        )
        self.assert_regex(result)
        # Counter file device ID depends on GPU topology, search across IDs 0-9
        counter_files = []
        for pattern in gpu_info.expected_counter_files:
            matches = list(result.output_dir.glob(pattern))
            counter_files.extend(matches if matches else [result.output_dir / pattern])
        self.assert_file_exists(
            counter_files,
            description="Counter file",
            subtest_name="Counter file check",
        )
        if mode == "sampling":
            self.assert_perfetto(
                result,
                subtest_name="Perfetto counter validation",
                counter_names=gpu_info.counter_names,
                check_counter_pairing=True,
            )
            self.assert_rocpd(
                result,
                subtest_name="RocPD HW counter validation",
                rules_files=rocprofiler_rules,
            )


# ============================================================================
# Test Class: GPU Performance Counter Collection (Device Counting Service)
# ============================================================================


@pytest.mark.mpi_optional("transpose")
@pytest.mark.rocprofiler
@pytest.mark.timeout(120)
@pytest.mark.class_name("transpose-gpu-perf-counters")
class TestTransposeGPUPerfCounters(RocprofsysTest):
    @pytest.mark.rocpd("gpu_perf_counter_env")
    def test(
        self,
        gpu_perf_counter_env,
        gpu_info,
        validation_rules_dir,
    ):
        if "gfx1151" in gpu_info.architectures:
            pytest.skip("transpose GPU perf counter test skipped on gfx1151")

        result = self.run_test(
            "sampling",
            "transpose",
            env=gpu_perf_counter_env,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto GPU perf counter validation",
            counter_names=gpu_info.counter_names,
        )
        rules_dir = validation_rules_dir / "transpose"
        self.assert_rocpd(
            result,
            subtest_name="ROCpd GPU perf counter validation",
            rules_files=[rules_dir / "gpu-perf-counter-rules.json"],
        )
