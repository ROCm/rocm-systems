# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for OpenMP integration with rocprofiler-systems.
"""

from __future__ import annotations
import os
import pytest
from pathlib import Path
from conftest import RocprofsysTest, check_use_perfetto

pytestmark = [
    pytest.mark.openmp,
    pytest.mark.rocm_min_version(
        "6.4"
    ),  # Requires SDK version >= 600, 6.3 ships with 500
]

# ============================================================================
# Sampling Duration Validation
# ============================================================================

SAMPLING_DURATION_SEC = 0.25

# The window is closed by a timer, so the last in-flight sample and the
# finalization behind it can land after the deadline.
SAMPLING_DURATION_UPPER_SLACK_SEC = 0.35

# A window that collapses to nothing would satisfy the upper bound trivially.
SAMPLING_DURATION_MIN_SEC = 0.02

# The run has to outlive the window, otherwise a workload that simply finished
# first is indistinguishable from a duration that actually cut sampling off.
SAMPLING_TRUNCATION_MARGIN_SEC = 0.25

_SAMPLING_WINDOW_QUERY = """
    SELECT COUNT(*) AS track_count,
           MAX(track_end_ns - track_start_ns) AS max_track_span_ns,
           MAX(track_end_ns) - MIN(track_start_ns) AS window_span_ns
    FROM (
        SELECT track_id,
               MIN(ts) AS track_start_ns,
               MAX(ts + COALESCE(dur, 0)) AS track_end_ns
        FROM slice
        WHERE category = 'timer_sampling'
        GROUP BY track_id
    )
"""


def _query_sampling_window(perfetto_file: Path):
    """Return the aggregated timer_sampling extents from a Perfetto trace."""
    from perfetto.common.exceptions import PerfettoException
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig

    trace_processor_path = os.environ.get("ROCPROFSYS_TRACE_PROC_SHELL")
    config = None
    if trace_processor_path and Path(trace_processor_path).is_file():
        config = TraceProcessorConfig(bin_path=trace_processor_path)

    try:
        trace_processor = (
            TraceProcessor(trace=str(perfetto_file), config=config)
            if config is not None
            else TraceProcessor(trace=str(perfetto_file))
        )
    except PerfettoException as exc:
        pytest.skip(f"Perfetto trace processor unavailable on this system: {exc}")

    try:
        rows = list(trace_processor.query(_SAMPLING_WINDOW_QUERY))
    finally:
        close = getattr(trace_processor, "close", None)
        if close is not None:
            close()

    return rows[0] if rows else None


def _assert_sampling_duration_window(result) -> None:
    """Assert ROCPROFSYS_SAMPLING_DURATION actually bounds timer sampling.

    The window is enforced by control::triggers::time_window, which emits no log
    output, so the trace is the only place the contract is observable.
    """
    if not check_use_perfetto():
        pytest.skip("Perfetto is disabled")

    perfetto_file = result.perfetto_file
    if perfetto_file is None or not perfetto_file.exists():
        pytest.fail("No Perfetto trace file was produced by the run")

    row = _query_sampling_window(perfetto_file)
    if row is None or not row.track_count or row.max_track_span_ns is None:
        pytest.fail("No timer_sampling rows found in Perfetto output")

    max_track_span_sec = row.max_track_span_ns / 1.0e9
    window_span_sec = row.window_span_ns / 1.0e9
    max_allowed_sec = SAMPLING_DURATION_SEC + SAMPLING_DURATION_UPPER_SLACK_SEC

    if max_track_span_sec > max_allowed_sec:
        pytest.fail(
            "Sampling outlived the configured duration: max per-track span "
            f"{max_track_span_sec:.3f}s > allowed {max_allowed_sec:.3f}s "
            f"(ROCPROFSYS_SAMPLING_DURATION={SAMPLING_DURATION_SEC}s)"
        )

    if window_span_sec < SAMPLING_DURATION_MIN_SEC:
        pytest.fail(
            f"Sampling window collapsed to {window_span_sec:.3f}s; expected at "
            f"least {SAMPLING_DURATION_MIN_SEC:.3f}s of samples before the "
            "duration elapsed"
        )

    run_duration_sec = result.duration
    if run_duration_sec is None:
        return

    if window_span_sec + SAMPLING_TRUNCATION_MARGIN_SEC > run_duration_sec:
        pytest.fail(
            f"Run lasted {run_duration_sec:.3f}s but sampling covered "
            f"{window_span_sec:.3f}s of it, so the duration cutoff is unproven "
            "- the workload may simply have ended before the window closed"
        )


# ============================================================================
# OpenMP Fixtures
# ============================================================================


@pytest.fixture
def ompt_base_env() -> dict[str, str]:
    """Environment variables for OMPT tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_OMPT": "ON",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
    }


@pytest.fixture
def ompt_sampling_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for sampling duration tests."""
    env = ompt_base_env.copy()
    env.update(
        {
            "ROCPROFSYS_USE_OMPT": "OFF",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
            "ROCPROFSYS_SAMPLING_FREQ": "100",
            "ROCPROFSYS_SAMPLING_DELAY": "0.1",
            "ROCPROFSYS_SAMPLING_DURATION": "0.25",
            "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
            "ROCPROFSYS_SAMPLING_REALTIME": "ON",
            "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "1000",
            "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "500",
            "ROCPROFSYS_MONOCHROME": "ON",
        }
    )
    return env


@pytest.fixture
def ompt_target_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for OpenMP target (GPU) tests."""
    env = ompt_base_env.copy()
    env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,hsa_api,kernel_dispatch"
    return env


@pytest.fixture
def ompt_no_tmp_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for no-tmp-files tests."""
    env = ompt_base_env.copy()
    env.update(
        {
            "ROCPROFSYS_USE_OMPT": "OFF",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
            "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "700",
            "ROCPROFSYS_USE_TEMPORARY_FILES": "OFF",
            "ROCPROFSYS_MONOCHROME": "ON",
        }
    )
    return env


@pytest.fixture
def openmp_target_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for OpenMP target tests."""
    rules_dir = validation_rules_dir / "openmp-target"
    return [
        rules_dir / "kernel-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# ============================================================================
# Test Class: OpenMP CG Tests
# ============================================================================


class TestOpenMPCG(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-cg",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
        )
        self.assert_regex(result)

    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_sampling_env,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)
        _assert_sampling_duration_window(result)

    @pytest.mark.no_tmp_files
    def test_no_tmp_files(self, ompt_no_tmp_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_no_tmp_env,
        )
        self.assert_regex(result, pass_regex=self.NOTMP_SAMPLING_FILE_REGEX)
        self.assert_perfetto(result)


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


class TestOpenMPLU(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    BINARY_REWRITE_PASS_REGEX = ["\\|_omp_"]
    BINARY_REWRITE_FAIL_REGEX = ["0 instrumented loops in procedure"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("mode", ["baseline", "sampling", "binary_rewrite"])
    def test(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "ON"
        env["ROCPROFSYS_SAMPLING_FREQ"] = "50"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-lu",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=self.BINARY_REWRITE_PASS_REGEX,
            binary_rewrite_fail_regex=self.BINARY_REWRITE_FAIL_REGEX,
        )

    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-lu",
            env=ompt_sampling_env,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)
        _assert_sampling_duration_window(result)


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.build_only
@pytest.mark.rocm
@pytest.mark.gpu
@pytest.mark.class_name("openmp-target")
class TestOpenMPTarget(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            pytest.param("sampling", marks=pytest.mark.rocpd("ompt_target_env")),
            "sys_run",
        ],
    )
    def test(self, mode, ompt_target_env, openmp_target_rules):
        result = self.run_test(mode, "openmp-target", env=ompt_target_env)
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_rocpd(result, rules_files=openmp_target_rules)
            self.assert_perfetto(
                result,
                subtest_name="Perfetto Kernel Dispatch Validation",
                categories=["rocm_kernel_dispatch"],
                label_substrings=[
                    "Z4vmulIiEvPT_S1_S1_i_l51.kd",
                    "Z4vmulIfEvPT_S1_S1_i_l51.kd",
                    "Z4vmulIdEvPT_S1_S1_i_l51.kd",
                ],
                counts=[4, 4, 4],
            )


# ============================================================================
# Test Class: OpenMP Fortran Tests
# ============================================================================


@pytest.mark.fortran
@pytest.mark.class_name("openmp-fortran")
class TestOpenMPFortran(RocprofsysTest):

    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    RUNTIME_INSTRUMENT_ARGS = ["-e", "-v", "2", "--label", "return", "args"]

    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    def test_host(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-fortran-host",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=["omp_parallel"],
            runtime_instrument_pass_regex=["omp_parallel"],
            sys_run_pass_regex=["omp_parallel"],
        )

    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            "sampling",
            "binary_rewrite",
            "sys_run",
            pytest.param(
                "runtime_instrument",
                marks=[
                    pytest.mark.slow,
                    pytest.mark.serialize,
                ],
            ),
        ],
    )
    @pytest.mark.gpu
    def test_offload(self, mode, ompt_target_env):
        env = ompt_target_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        # libomptarget exceeds the default --max-library-functions threshold and
        # would normally be skipped. Force it to be kept via module-include
        runtime_instrument_args = self.RUNTIME_INSTRUMENT_ARGS + [
            "-MI",
            "libomptarget",
        ]

        result = self.run_test(
            mode,
            "openmp-fortran-offload",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=runtime_instrument_args,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=["omp_offloading"],
            runtime_instrument_pass_regex=["omp_offloading"],
            sys_run_pass_regex=["omp_offloading"],
        )
