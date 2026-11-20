# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the transpose example.
Equivalent to rocprof-sys-rocm-tests.cmake
    Note: MPI multi-process execution is exercised if built with MPI support.

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

from pathlib import Path
from typing import ClassVar

import pytest
from conftest import RocprofsysTest, assert_resolved_settings

pytestmark = [
    pytest.mark.transpose,
    pytest.mark.gpu,
    pytest.mark.rocm,
]

from rocprofsys import (
    GPUInfo,
)
from rocprofsys.gpu import UNSUPPORTED_PERF_COUNTER_GFX

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
def trace_delay_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules asserting zero kernels were captured.

    Deliberately does NOT include default-rules.json/transpose_rules: those
    require at least one kernel row, which is the opposite of what this
    rule set checks.
    """
    return [validation_rules_dir / "transpose" / "trace-delay-rules.json"]


# ============================================================================
# Test Class: Basic Transpose Tests
# ============================================================================


@pytest.mark.mpi_optional("transpose")
class TestTranspose(RocprofsysTest):
    BINARY_REWRITE_ARGS: ClassVar[list[str]] = [
        "-e",
        "-v",
        "2",
        "--print-instructions",
        "-E",
        "uniform_int_distribution",
    ]
    RUNTIME_INSTRUMENT_ARGS: ClassVar[list[str]] = [
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
    TWO_KERNELS_RUN_ARGS: ClassVar[list[str]] = ["1", "2", "2"]

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
            self.assert_perfetto(result)

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("transpose_env")
    def test_sampling(self, transpose_env, transpose_rules):
        sampling_run_args = ["4", "500", "100"]
        env = {
            **transpose_env,
            "ROCPROFSYS_SAMPLING_REALTIME": "ON",
            "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "300",
            "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
        }
        result = self.run_test(
            "sampling",
            target="transpose",
            env=env,
            run_args=sampling_run_args,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto HIP API Call Validation",
            categories=["hip_runtime_api"],
        )
        self.assert_rocpd(result, rules_files=transpose_rules)

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param("sampling", marks=pytest.mark.timeout(120)),
            pytest.param("sys_run"),
        ],
    )
    def test_two_kernels(self, mode, transpose_env):
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            run_args=self.TWO_KERNELS_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result)

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("transpose_env")
    @pytest.mark.parametrize("mode", ["binary_rewrite"])
    def test_trace_delay_gates_gpu_contexts(self, mode, transpose_env, trace_delay_rules):
        """
        Regression test: ROCPROFSYS_TRACE_DELAY must gate the actual startup of
        the "main" rocprofiler-sdk contexts (primary_ctx, counter_ctx), not just
        suppress downstream category emission. Before the fix, tool_init()
        unconditionally started every context whenever no roctx marker/trace
        region client existed - the common case for plain GPU tracing with just
        TRACE_DELAY set - so a configured delay never produced a real gap in
        cached GPU/RocPD data.

        transpose with TWO_KERNELS_RUN_ARGS (1 thread, 2 iterations, sync every
        2) completes in well under a second end-to-end. A 60s TRACE_DELAY is
        two orders of magnitude larger than that, so the delay window is
        guaranteed to never close before the process exits: if the fix works,
        the GPU contexts never start and zero kernels are ever captured.
        """
        env = transpose_env.copy()
        env["ROCPROFSYS_TRACE_DELAY"] = "60"
        result = self.run_test(
            mode,
            "transpose",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
            run_args=self.TWO_KERNELS_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_rocpd(
            result,
            subtest_name="ROCpd TRACE_DELAY GPU context gating validation",
            rules_files=trace_delay_rules,
        )

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("transpose_env")
    @pytest.mark.parametrize("mode", ["sampling"])
    def test_trace_periods_open_window(self, mode, transpose_env, validation_rules_dir):
        """Related to the above `test_trace_delay_gates_gpu_contexts`, but tests the
        `--trace-periods` flag instead of `TRACE_DELAY`.
        Same short run that should complete before the 60 second window closes.
        This test uses the command line argument instead of the environment variable.
        Assert that the command line argument is resolved to the expected value.
        """
        sampling_args = ["--trace-periods", "0:60"]
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            sampling_args=sampling_args,
            run_args=self.TWO_KERNELS_RUN_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        assert_resolved_settings(
            result,
            {"ROCPROFSYS_TRACE_PERIODS": "0:60"},
        )
        self.assert_perfetto(
            result,
            label_substrings=["transpose_a"],
            subtest_name="open window recorded transpose_a",
        )
        self.assert_rocpd(
            result,
            subtest_name="open window recorded kernels in rocpd",
            rules_files=[validation_rules_dir / "default-rules.json"],
        )

    @pytest.mark.locks
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "lock_mode, regression",
        [
            pytest.param(
                "mutex-locks",
                "hung rocprof-sys-run indefinitely "
                "(self-deadlock on buffer_storage's m_mutex)",
                id="mutex-locks",
            ),
            pytest.param(
                "rw-locks",
                "aborted rocprof-sys-run with SIGABRT "
                "(self-deadlock on synchronized<>'s rwlock)",
                id="rw-locks",
            ),
        ],
    )
    def test_locks(self, lock_mode, regression, transpose_env):
        """
        Regression test: pthread_mutex_gotcha used to intercept rocprof-sys's
        own internal locks, recursively re-entering them on the same thread
        while recording the trace event for the lock acquisition itself.
        ``regression`` is the per-mode failure signature reported on failure.
        """
        result = self.run_test(
            "sys_run",
            "transpose",
            env=transpose_env,
            sys_run_args=["-I", lock_mode],
            run_args=["2", "50", "10"],
            check_target_arch=True,
            fail_message=f"Regression: {regression}",
        )
        self.assert_regex(result)

    @pytest.mark.timeout(120)
    @pytest.mark.loops
    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test_loops(self, mode, transpose_env):
        loops_binary_rewrite_args = [
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
        loops_run_args = ["2", "100", "50"]
        result = self.run_test(
            mode,
            "transpose",
            env=transpose_env,
            binary_rewrite_args=loops_binary_rewrite_args,
            run_args=loops_run_args,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_fail_regex=["0 instrumented loops in procedure transpose"],
        )

    @pytest.mark.timeout(120)
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
    @pytest.mark.timeout(120)
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
# Test Class: ROCProfiler Counter Collection
# ============================================================================


@pytest.mark.mpi_optional("transpose")
@pytest.mark.rocprofiler
@pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
@pytest.mark.class_name("transpose-rocprofiler")
class TestTransposeROCProfiler(RocprofsysTest):

    @pytest.mark.timeout(120)
    @pytest.mark.rocpd("rocprofiler_env")
    def test(self, mode, rocprofiler_env, gpu_info, rocprofiler_rules):
        binary_rewrite_args = ["-e", "-v", "2", "-E", "uniform_int_distribution"]
        result = self.run_test(
            mode,
            "transpose",
            env=rocprofiler_env,
            check_target_arch=True,
            launcher="mpi",
            num_procs=2,
            binary_rewrite_args=binary_rewrite_args,
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
@pytest.mark.class_name("transpose-gpu-perf-counters")
@pytest.mark.timeout(120)
@pytest.mark.cap_perfmon
@pytest.mark.disable_archs(UNSUPPORTED_PERF_COUNTER_GFX)
class TestTransposeGPUPerfCounters(RocprofsysTest):
    @pytest.mark.rocpd("gpu_perf_counter_env")
    def test(
        self,
        gpu_perf_counter_env,
        gpu_info,
        validation_rules_dir,
    ):
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


# ============================================================================
# rocprof-sys-sample flags on the transpose HIP workload
# ============================================================================

# transpose: threads, kernel iterations, sync interval
TRANSPOSE_SAMPLE_ARGS = ["2", "100", "50"]

# Printed perfetto row ends with the slice name padded, then " |".
# "hipFree " does not match hipFreeAsync.
HIP_FREE_SLICE = r"hipFree\s+\|"

HIP_MALLOC_TRACE = {"categories": ["rocm_hip_api"], "labels": ["hipMalloc"]}


@pytest.mark.sampling
@pytest.mark.timeout(180)
@pytest.mark.class_name("transpose-sample-flags")
class TestTransposeSampleFlags(RocprofsysTest):
    def run_transpose(self, sampling_args, env=None):
        return self.run_test(
            "sampling",
            "transpose",
            env=env,
            sampling_args=sampling_args,
            run_args=TRANSPOSE_SAMPLE_ARGS,
            check_target_arch=True,
        )

    @pytest.mark.parametrize(
        "flag_args, pass_regex",
        [
            pytest.param(
                ["--perfetto-flush-period-ms=250"],
                [r"ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS=250"],
                id="perfetto-flush-period",
            ),
            pytest.param(
                ["--causal-backend=timer"],
                [r"ROCPROFSYS_CAUSAL_BACKEND=timer"],
                id="causal-backend",
            ),
            pytest.param(
                ["--kill-delay=1"],
                [r"ROCPROFSYS_KILL_DELAY=1"],
                id="kill-delay",
            ),
        ],
    )
    def test_setting(self, flag_args, pass_regex):
        result = self.run_transpose(flag_args)
        self.assert_regex(result, pass_regex=pass_regex)
        self.assert_perfetto(result, **HIP_MALLOC_TRACE)

    @pytest.mark.parametrize(
        "flag_args, pass_regex",
        [
            pytest.param(
                ["--rocm-hip-runtime-api-operations", "hipMalloc"],
                [r"ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS=hipMalloc"],
                id="include",
            ),
            pytest.param(
                ["--rocm-hip-runtime-api-operations-exclude", "hipFree"],
                [r"ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS_EXCLUDE=hipFree"],
                id="exclude",
            ),
        ],
    )
    def test_runtime_api(self, flag_args, pass_regex):
        result = self.run_transpose(flag_args)
        self.assert_regex(result, pass_regex=pass_regex)
        self.assert_perfetto(result, **HIP_MALLOC_TRACE, fail_regex=[HIP_FREE_SLICE])

    def test_annotate_backtrace(self):
        # The trace cache records the backtrace when the rocpd backend is on.
        # Two threads each call hipMalloc twice, so four slices carry frame#0.
        result = self.run_transpose(
            ["--rocm-hip-runtime-api-operations-annotate-backtrace", "hipMalloc"],
            env={"ROCPROFSYS_USE_ROCPD": "ON"},
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_ROCM_HIP_RUNTIME_API_OPERATIONS_ANNOTATE_BACKTRACE=hipMalloc"
            ],
        )
        self.assert_perfetto(
            result,
            **HIP_MALLOC_TRACE,
            key_names=["frame#0"],
            key_counts=[4],
            pass_regex=[r"string_value\s*::\s*hipMalloc"],
        )

    def test_output(self):
        # Harness already sets OUTPUT_PATH, so the flag has to land somewhere else.
        out_dir = self.test_output_dir / "custom"
        result = self.run_transpose(["--output", str(out_dir), "sample-out"])
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_OUTPUT_PATH=.*/custom",
                r"ROCPROFSYS_OUTPUT_PREFIX=sample-out",
            ],
        )
        self.assert_perfetto(
            result,
            perfetto_file=Path("custom") / "sample-out-perfetto-trace.proto",
            **HIP_MALLOC_TRACE,
        )

    def test_trace(self):
        result = self.run_transpose(["--trace"], env={"ROCPROFSYS_TRACE": "OFF"})
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE=true"])
        self.assert_perfetto(result, **HIP_MALLOC_TRACE)

    def test_trace_legacy(self):
        # Both writers produce the same slices, so only the log tells them apart:
        # legacy writes the proto in-process, the default path from the trace cache.
        result = self.run_transpose(["--trace-legacy"])
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE_LEGACY=true",
                r"\[perfetto\]> Outputting",
            ],
            fail_regex=[
                r"Generating \[[^\]]*perfetto[^\]]*\] format\(s\) with collected data from trace cache"
            ],
        )
        self.assert_perfetto(result, **HIP_MALLOC_TRACE)

    def test_profile(self):
        # Call-stack profile nests HIP calls under the parent (`|_hipMalloc`).
        result = self.run_transpose(["--profile"], env={"ROCPROFSYS_PROFILE": "OFF"})
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_PROFILE=true"])
        self.assert_file_regex(
            result.output_dir / "wall_clock.txt",
            pass_regex=[r"\|_hipMalloc"],
        )

    def test_flat_profile(self):
        # Flat profile lists each function at depth 0, with no `|_` nesting.
        result = self.run_transpose(
            ["--flat-profile"],
            env={"ROCPROFSYS_PROFILE": "OFF", "ROCPROFSYS_FLAT_PROFILE": "OFF"},
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_FLAT_PROFILE=true",
                r"ROCPROFSYS_PROFILE=true",
            ],
        )
        self.assert_file_regex(
            result.output_dir / "wall_clock.txt",
            pass_regex=[r">>> hipMalloc"],
            fail_regex=[r"\|_hipMalloc"],
        )
