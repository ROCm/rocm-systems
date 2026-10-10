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
import json
import shutil
import signal
import pytest
from pathlib import Path
from conftest import RocprofsysTest

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

    _LOCK_MODE_REGRESSIONS = {
        "mutex-locks": "hung rocprof-sys-run indefinitely "
        "(self-deadlock on buffer_storage's m_mutex)",
        "rw-locks": "aborted rocprof-sys-run with SIGABRT "
        "(self-deadlock on synchronized<>'s rwlock)",
    }

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

    @pytest.mark.locks
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "lock_mode",
        [
            pytest.param("mutex-locks", id="mutex-locks"),
            pytest.param("rw-locks", id="rw-locks"),
        ],
    )
    def test_locks(self, lock_mode, transpose_env):
        """
        Regression test: pthread_mutex_gotcha used to intercept rocprof-sys's
        own internal locks, recursively re-entering them on the same thread
        while recording the trace event for the lock acquisition itself.
        See _LOCK_MODE_REGRESSIONS for the per-mode failure signature.
        """
        result = self.run_test(
            "sys_run",
            "transpose",
            env=transpose_env,
            sys_run_args=["-I", lock_mode],
            run_args=["2", "50", "10"],
            check_target_arch=True,
            fail_message=f"Regression: {self._LOCK_MODE_REGRESSIONS[lock_mode]}",
        )
        self.assert_regex(result)

    @pytest.mark.timeout(120)
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
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "-E", "uniform_int_distribution"]

    @pytest.mark.timeout(120)
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


TRANSPOSE_ARGS = ["2", "100", "50"]

# effect is what the flag changes in the run output:
#   kernels     - an open trace window records the transpose kernel
#   no_kernels  - a window that never opens records none
#   diff        - --profile-diff writes a diff against a previous profile
SAMPLE_TRANSPOSE_FLAG_CASES = [
    pytest.param(
        ["--trace-periods", "0:10"],
        {"ROCPROFSYS_TRACE_PERIODS": "0:10"},
        "kernels",
        id="trace_periods",
    ),
    pytest.param(
        ["--trace-periods", "10:15"],
        {"ROCPROFSYS_TRACE_PERIODS": "10:15"},
        "no_kernels",
        id="trace_periods_closed",
    ),
    pytest.param(
        ["--profile-diff"],
        {
            "ROCPROFSYS_DIFF_OUTPUT": True,
            "ROCPROFSYS_INPUT_PATH": "{baseline}",
        },
        "diff",
        id="profile_diff",
    ),
]


def _resolved_settings(result) -> dict:
    """Settings block written by the profiled process, not the launcher echo."""
    metadata_file = result.get_output_file("metadata*.json")
    if metadata_file is None:
        pytest.fail(f"No metadata*.json under {result.output_dir}")
    settings = json.loads(metadata_file.read_text())["rocprofiler-systems"]["metadata"][
        "settings"
    ]
    return {
        key: entry["value"]
        for key, entry in settings.items()
        if isinstance(entry, dict) and "value" in entry
    }


def _check_resolved_settings(result, flag_args, expected: dict) -> None:
    settings = _resolved_settings(result)
    for key, value in expected.items():
        if key not in settings:
            pytest.fail(f"{flag_args}: {key} missing from metadata.json")
        if settings[key] != value:
            pytest.fail(
                f"{flag_args}: metadata.json has {key}={settings[key]!r}, "
                f"expected {value!r}"
            )


def _profile_inputs(output_dir: Path) -> list[Path]:
    return [
        path
        for path in output_dir.iterdir()
        if path.is_file() and "wall_clock" in path.name and ".diff." not in path.name
    ]


@pytest.mark.sampling
@pytest.mark.timeout(180)
@pytest.mark.class_name("sample-transpose-cli")
@pytest.mark.rocpd("transpose_env")
class TestSampleTransposeCli(RocprofsysTest):
    # transpose_env must be threaded through run_test: the rocpd marker turns
    # ROCPROFSYS_USE_ROCPD on by mutating that fixture's dict, so a run that
    # does not pass it leaves the assert_rocpd calls below depending on the
    # build's default output format instead of on the test.
    def _run_transpose(self, sampling_args, env):
        return self.run_test(
            "sampling",
            target="transpose",
            env=env,
            run_args=TRANSPOSE_ARGS,
            sampling_args=sampling_args,
            check_target_arch=True,
        )

    def _assert_transpose_trace(self, result, present: bool):
        traces = _perfetto_traces(result.output_dir)
        if not traces and result.perfetto_file is not None:
            traces = [result.perfetto_file]
        if not traces:
            pytest.fail(f"No Perfetto trace under {result.output_dir}")
        for trace in traces:
            if present:
                self.assert_perfetto(
                    result,
                    perfetto_file=trace,
                    label_substrings=["transpose_a"],
                    subtest_name=f"{trace.name} recorded transpose_a",
                )
            else:
                self.assert_perfetto(
                    result,
                    perfetto_file=trace,
                    label_substrings=["transpose_a"],
                    counts=[0],
                    subtest_name=f"{trace.name} has no transpose_a",
                )

    @pytest.mark.parametrize(
        "flag_args, expected_settings, effect", SAMPLE_TRANSPOSE_FLAG_CASES
    )
    def test_flag_changes_output(
        self,
        flag_args,
        expected_settings,
        effect,
        transpose_env,
        test_output_dir,
        trace_delay_rules,
        validation_rules_dir,
    ):
        resolved_flags = list(flag_args)
        expected = dict(expected_settings)
        if effect == "diff":
            baseline = test_output_dir / "baseline"
            first = self._run_transpose([], transpose_env)
            profiles = _profile_inputs(first.output_dir)
            if not profiles:
                pytest.fail(
                    f"no wall_clock profile under {first.output_dir} to diff against"
                )
            baseline.mkdir()
            for profile in profiles:
                shutil.copy2(profile, baseline / profile.name)
            resolved_flags = ["--profile-diff", str(baseline)]
            expected["ROCPROFSYS_INPUT_PATH"] = str(baseline)

        result = self._run_transpose(resolved_flags, transpose_env)
        # metadata.json is written by the profiled process, so this proves the
        # runtime kept the value. The launcher's stdout echo only shows what it
        # put in the environment before exec, so there is nothing to add here
        # beyond the default abort check.
        _check_resolved_settings(result, resolved_flags, expected)
        self.assert_regex(result)

        if effect == "kernels":
            self._assert_transpose_trace(result, present=True)
            # default-rules requires a kernel row. validation-rules.json
            # also requires 1000 calls, which 2 100 50 does not produce.
            self.assert_rocpd(
                result,
                subtest_name="open window recorded kernels in rocpd",
                rules_files=[validation_rules_dir / "default-rules.json"],
            )
        elif effect == "no_kernels":
            self._assert_transpose_trace(result, present=False)
            self.assert_rocpd(
                result,
                subtest_name="closed window recorded no kernels in rocpd",
                rules_files=trace_delay_rules,
            )
        elif effect == "diff":
            if any(baseline.glob("*.diff.*")):
                pytest.fail(
                    f"--profile-diff wrote a diff file into the baseline {baseline}"
                )
            diff_txt = result.output_dir / "wall_clock.diff.txt"
            diff_json = result.output_dir / "wall_clock.diff.json"
            if not diff_txt.is_file() or not diff_json.is_file():
                pytest.fail(
                    f"--profile-diff did not write wall_clock.diff.txt and "
                    f"wall_clock.diff.json under {result.output_dir}"
                )
            # Seconds are deltas, so the exact number is not fixed.
            self.assert_file_regex(
                diff_txt,
                pass_regex=[
                    r"vs\.",
                    r"baseline/wall_clock\.json",
                    r"sec\s+\|\s+-?\d+\.\d+",
                ],
                subtest_name="wall_clock.diff.txt compares the baseline in seconds",
            )
            self.assert_file_regex(
                diff_json,
                pass_regex=[r'"unit_repr"\s*:\s*"sec"', r'"repr_data"\s*:\s*-?\d+\.\d+'],
                subtest_name="wall_clock.diff.json has second values",
            )
            self._assert_transpose_trace(result, present=True)


# <nfib> <nthreads> <nitr>. Long enough for a CPU-metric sample.
CPU_METRICS_ARGS = ["20", "2", "100000"]


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("sample-cpu-metrics")
class TestSampleCpuMetrics(RocprofsysTest):
    """--cpu-metrics selects host counters. parallel-overhead has no GPU work."""

    def test_frequency_and_load(self, rocprof_config):
        _example_executable(rocprof_config, "parallel-overhead")
        result = self.run_test(
            "sampling",
            target="parallel-overhead",
            run_args=CPU_METRICS_ARGS,
            sampling_args=["--cpu-metrics", "frequency,load"],
            fail_on_not_found=True,
        )
        _check_resolved_settings(
            result,
            ["--cpu-metrics", "frequency,load"],
            {"ROCPROFSYS_CPU_METRICS": "frequency,load"},
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="frequency and load counters",
            counter_names_present=["Frequency (S)", "Load (S)"],
        )
        trace = result.perfetto_file.read_bytes()
        if b"Page RSS" in trace:
            pytest.fail(
                "cpu-metrics frequency,load still recorded Page RSS, "
                f"which is outside that selection: {result.perfetto_file}"
            )


# Realtime sampling with a trace period. Cputime sampling is off so
# timer_sampling rows come from the wall-clock sampler.
PERIOD_SAMPLING = {
    "ROCPROFSYS_USE_SAMPLING": "ON",
    "ROCPROFSYS_SAMPLING_REALTIME": "ON",
    "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "500",
    "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
    "ROCPROFSYS_SAMPLING_DELAY": "0.1",
}

# Long enough that --periods 0:5 stays open while
# sampling records fib frames.
OPEN_PERIOD_ARGS = ["20", "2", "100000"]

# Finishes well under a second,
# so a delay of 8s or 10s never opens.
CLOSED_TRANSPOSE_ARGS = ["2", "50", "10"]
CLOSED_PERIODS = [
    pytest.param("10:20", id="10s-20s"),
    pytest.param("8:2", id="8s-2s"),
]


def _example_executable(rocprof_config, name: str) -> str:
    try:
        return str(rocprof_config.get_target_executable(name))
    except FileNotFoundError:
        pytest.skip(f"{name} example not built")


def _perfetto_traces(output_dir) -> list:
    """Perfetto traces written as .pftrace and, when present, .proto."""
    traces = []
    for ext in ("pftrace", "proto"):
        traces.extend(sorted(output_dir.glob(f"perfetto-trace*.{ext}")))
    return traces


@pytest.mark.skip(reason="issue is reported in AIPROFSYST-773")
@pytest.mark.sys_run
@pytest.mark.timeout(120)
@pytest.mark.class_name("transpose-periods")
class TestTransposePeriods(RocprofsysTest):
    """--periods 0:5 with sampling on records timer samples for the whole run."""

    def test_open_window(self, rocprof_config, validation_rules_dir):
        workload = _example_executable(rocprof_config, "parallel-overhead")
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            env=PERIOD_SAMPLING,
            run_args=["--periods", "0:5", "--", workload, *OPEN_PERIOD_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(result)
        _check_resolved_settings(
            result, ["--periods", "0:5"], {"ROCPROFSYS_TRACE_PERIODS": "0:5"}
        )

        self.assert_perfetto(
            result,
            label_substrings=["fib"],
            subtest_name="open window trace has sampled fib frames",
        )
        self.assert_rocpd(
            result,
            subtest_name="open window has timer_sampling rows",
            rules_files=[
                validation_rules_dir / "transpose" / "timer-sampling-rules.json"
            ],
        )


@pytest.mark.skip(reason="issue is reported in AIPROFSYST-773")
@pytest.mark.sys_run
@pytest.mark.timeout(120)
@pytest.mark.class_name("transpose-periods-closed")
class TestTransposePeriodsClosed(RocprofsysTest):
    """A --periods delay longer than transpose records nothing in the
    Perfetto trace (.pftrace or .proto) or in rocpd. Sampling is still on.
    """

    @pytest.mark.parametrize("period", CLOSED_PERIODS)
    def test_empty(self, rocprof_config, period, validation_rules_dir):
        workload = _example_executable(rocprof_config, "transpose")
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            env=PERIOD_SAMPLING,
            run_args=["--periods", period, "--", workload, *CLOSED_TRANSPOSE_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(result)
        _check_resolved_settings(
            result, ["--periods", period], {"ROCPROFSYS_TRACE_PERIODS": period}
        )

        traces = _perfetto_traces(result.output_dir)
        if not traces:
            pytest.fail(f"No Perfetto .pftrace or .proto under {result.output_dir}")
        for trace in traces:
            self.assert_perfetto(
                result,
                perfetto_file=trace,
                label_substrings=["transpose_a"],
                counts=[0],
                subtest_name=f"{period} {trace.name} has no transpose kernel",
            )
        self.assert_rocpd(
            result,
            subtest_name=f"{period} recorded no samples or kernels",
            rules_files=[validation_rules_dir / "transpose" / "trace-delay-rules.json"],
        )


# ============================================================================
# rocprof-sys-sample flag coverage on the transpose HIP workload
#
# Each class below pins one flag that previously had no test outside
# --help text, and asserts the effect the flag has on the run rather than
# only that the launcher echoed the setting. An env echo on its own passes
# whether or not the flag does anything.
# ============================================================================


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("sample-transpose-mode")
@pytest.mark.rocpd("transpose_env")
class TestSampleTransposeMode(RocprofsysTest):
    """--mode on a HIP workload.

    Tests:  --mode coverage switches the run from tracing to code coverage.
    Input:  --mode coverage on transpose 2 100 50.
    Expect: ROCPROFSYS_MODE=coverage with TRACE and both samplers off, and
            neither a Perfetto trace nor a ROCpd database on disk.

    "coverage" is the only choice with an effect worth asserting here:
    "trace" is the default (a test of it would pass against no flag at
    all) and "sampling"/"causal" only move ROCPROFSYS_MODE in the
    settings dump when passed to rocprof-sys-sample. "coverage" turns
    tracing and both samplers off while leaving kernel dispatch
    tracing on, which is visible in the output artifacts.
    """

    MODE_FLAGS = ["--mode", "coverage"]

    def test_coverage_disables_trace_and_sampling(self, transpose_env):
        result = self.run_test(
            "sampling",
            target="transpose",
            env=transpose_env,
            run_args=TRANSPOSE_ARGS,
            sampling_args=self.MODE_FLAGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        _check_resolved_settings(
            result,
            self.MODE_FLAGS,
            {
                "ROCPROFSYS_MODE": "coverage",
                "ROCPROFSYS_TRACE": False,
                "ROCPROFSYS_USE_SAMPLING": False,
                "ROCPROFSYS_USE_PROCESS_SAMPLING": False,
                "ROCPROFSYS_USE_AMD_SMI": False,
                "ROCPROFSYS_USE_CODE_COVERAGE": True,
            },
        )

        # Tracing off means neither backend writes anything, so the effect to
        # assert is the absence of both artifacts. There is no trace or
        # database left to hand to assert_perfetto/assert_rocpd.
        traces = _perfetto_traces(result.output_dir)
        if result.perfetto_file is not None:
            traces.append(result.perfetto_file)
        if traces:
            pytest.fail(
                "--mode coverage turns ROCPROFSYS_TRACE off, but a Perfetto "
                f"trace was still written: {sorted(str(t) for t in traces)}"
            )
        if result.rocpd_files:
            pytest.fail(
                "--mode coverage turns ROCPROFSYS_TRACE off, but a ROCpd "
                f"database was still written: {[str(f) for f in result.rocpd_files]}"
            )


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("sample-transpose-cputime-signal")
@pytest.mark.rocpd("transpose_env")
class TestSampleTransposeCputimeSignal(RocprofsysTest):
    """--sampling-cputime-signal moves the cputime sampler off SIGPROF.

    Tests:  the cputime sampler is armed on the requested signal, not the
            SIGPROF default, and keeps sampling once moved.
    Input:  --sampling-cputime-signal 26 (SIGVTALRM) on transpose 2 100 50.
    Expect: a "[SIG26] Sampler for thread 0" line and no "[SIG27] Sampler"
            line, with GPU kernels and timer samples still recorded.

    SIGVTALRM, not an arbitrary free signal. It is the other timer signal
    the kernel delivers on a CPU-time expiry, so the HIP runtime treats
    an interrupted call the same way it treats the SIGPROF default.
    Sampling transpose on SIGUSR2 instead fails roughly 6 runs in 10 with
    "HIP error : out of memory" out of hipStreamCreate, against 0 in 10
    on the default signal - the override is real enough to break the
    workload, so the signal the test picks is not interchangeable.
    """

    CPUTIME_SIGNAL = str(signal.SIGVTALRM.value)
    SIGNAL_FLAGS = ["--sampling-cputime-signal", CPUTIME_SIGNAL]

    def test_override_keeps_sampling(self, transpose_env, validation_rules_dir):
        result = self.run_test(
            "sampling",
            target="transpose",
            env=transpose_env,
            run_args=TRANSPOSE_ARGS,
            sampling_args=self.SIGNAL_FLAGS,
            check_target_arch=True,
        )
        # The sampler logs the signal it armed, which is the runtime acting on
        # the value rather than the launcher echoing it back. Without this the
        # test would pass on a build that accepted the flag and armed SIGPROF.
        self.assert_regex(
            result,
            pass_regex=[rf"\[SIG{self.CPUTIME_SIGNAL}\] Sampler for thread 0"],
            fail_regex=[rf"\[SIG{signal.SIGPROF.value}\] Sampler for thread"],
        )
        _check_resolved_settings(
            result,
            self.SIGNAL_FLAGS,
            {"ROCPROFSYS_SAMPLING_CPUTIME_SIGNAL": int(self.CPUTIME_SIGNAL)},
        )
        self.assert_perfetto(
            result,
            label_substrings=["transpose_a"],
            subtest_name="GPU kernels still recorded on the overridden signal",
        )
        # No rocpd row records which signal carried the sampler, so this is a
        # no-regression guard rather than a check on the flag itself: the
        # flag-specific assertion is the [SIG<n>] pair above.
        self.assert_rocpd(
            result,
            subtest_name="overriding the signal did not stop sampling",
            rules_files=[
                validation_rules_dir / "transpose" / "cputime-signal-rules.json"
            ],
        )


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("sample-transpose-trace-clock-id")
@pytest.mark.rocpd("transpose_env")
class TestSampleTransposeTraceClockId(RocprofsysTest):
    """--trace-clock-id selects the clock a --trace-periods window runs on.

    Tests:  the --trace-periods delay is measured against process CPU time
            instead of the realtime default.
    Input:  --trace-periods 2:10 --trace-clock-id cputime on transpose 2 100 50.
    Expect: the window opens about 2s of CPU time in, so transpose_a kernels
            land in both the trace and ROCpd; realtime would record none.

    The delay is what makes this clock-specific rather than a rerun of
    the --trace-periods cases. transpose 2 100 50 runs for about 1.5
    seconds of realtime while burning roughly 5 seconds of process CPU
    time across its threads, so a 2 second delay opens early on cputime
    and never opens at all on the realtime default. Drop the
    --trace-clock-id argument and the same periods record zero kernels.
    """

    CLOCK_FLAGS = ["--trace-periods", "2:10", "--trace-clock-id", "cputime"]

    def test_cputime_window(self, transpose_env, validation_rules_dir):
        result = self.run_test(
            "sampling",
            target="transpose",
            env=transpose_env,
            run_args=TRANSPOSE_ARGS,
            sampling_args=self.CLOCK_FLAGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        _check_resolved_settings(
            result,
            self.CLOCK_FLAGS,
            {
                "ROCPROFSYS_TRACE_PERIODS": "2:10",
                "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID": "cputime",
            },
        )
        self.assert_perfetto(
            result,
            label_substrings=["transpose_a"],
            subtest_name="cputime window recorded transpose_a",
        )
        self.assert_rocpd(
            result,
            subtest_name="cputime window opened before the run ended",
            rules_files=[
                validation_rules_dir / "transpose" / "cputime-window-rules.json"
            ],
        )


@pytest.mark.sampling
@pytest.mark.timeout(120)
@pytest.mark.class_name("sample-transpose-cpu-metrics")
@pytest.mark.rocpd("transpose_env")
class TestSampleTransposeCpuMetrics(RocprofsysTest):
    """--cpu-metrics on a HIP workload.

    Tests:  selecting host counters does not cost the GPU kernel rows that
            would otherwise be recorded alongside them.
    Input:  -H --cpus 0 --cpu-metrics frequency,load on transpose 2 100 50.
    Expect: "Frequency (S)" and "Load (S)" counter tracks beside transpose_a,
            and cpu_frequency/cpu_load PMC samples in ROCpd.

    TestSampleCpuMetrics covers the flag on parallel-overhead, which has
    no GPU work; this pins that selecting host counters does not cost the
    GPU kernel rows alongside them.

    -H and --cpus are not incidental: ROCPROFSYS_CPU_FREQ_ENABLED defaults
    to OFF and ROCPROFSYS_SAMPLING_CPUS to "none", so --cpu-metrics on its
    own selects metrics for a collector that never runs and emits no
    counter tracks at all.
    """

    CPU_METRICS_FLAGS = ["-H", "--cpus", "0", "--cpu-metrics", "frequency,load"]

    def test_frequency_and_load_with_gpu_kernels(
        self, transpose_env, validation_rules_dir
    ):
        result = self.run_test(
            "sampling",
            target="transpose",
            env=transpose_env,
            run_args=TRANSPOSE_ARGS,
            sampling_args=self.CPU_METRICS_FLAGS,
            check_target_arch=True,
        )
        self.assert_regex(result)
        _check_resolved_settings(
            result,
            self.CPU_METRICS_FLAGS,
            {
                "ROCPROFSYS_CPU_METRICS": "frequency,load",
                "ROCPROFSYS_CPU_FREQ_ENABLED": True,
                "ROCPROFSYS_SAMPLING_CPUS": "0",
            },
        )
        self.assert_perfetto(
            result,
            subtest_name="host frequency and load counters alongside GPU kernels",
            counter_names_present=["Frequency (S)", "Load (S)"],
            label_substrings=["transpose_a"],
        )
        # cpu-metrics-rules.json is a data-integrity sweep over whatever PMCs a
        # run produced; it passes with --cpu-metrics removed because the
        # thread_* metrics fill the same tables. These rules name the two PMCs
        # the flag selected and require samples for them.
        self.assert_rocpd(
            result,
            subtest_name="selected CPU PMCs reached ROCpd",
            rules_files=[
                validation_rules_dir / "transpose" / "cpu-metrics-selection-rules.json"
            ],
        )
