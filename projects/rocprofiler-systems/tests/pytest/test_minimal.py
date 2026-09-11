# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
These tests exercise minimal examples to verify targeted behavior of the profiler.
"""

from __future__ import annotations
import os
import pytest
from pathlib import Path
from conftest import RocprofsysTest, check_use_perfetto

pytestmark = [pytest.mark.minimal]

# =============================================================================
# CPU-time resume validation
# =============================================================================

# Sampling is paused for the whole delay, and parse_timer_data() drops any
# sample overlapping a pause interval, so every timer_sampling row in the trace
# was necessarily produced after the resume.
CPUTIME_RESUME_DELAY_SEC = 0.5

# One straggler would satisfy a bare "> 0"; require enough to show the timer is
# genuinely re-armed and firing, not that a single in-flight signal landed.
CPUTIME_RESUME_MIN_SAMPLES = 50
CPUTIME_RESUME_MIN_SPAN_SEC = 0.1

_CPUTIME_SAMPLES_QUERY = """
    SELECT COUNT(*) AS sample_count,
           MAX(ts) - MIN(ts) AS span_ns
    FROM slice
    WHERE category = 'timer_sampling'
"""


def _query_cputime_samples(perfetto_file: Path):
    """Return aggregated timer_sampling extents from a Perfetto trace."""
    # check_use_perfetto() only proves that `import perfetto` succeeds, which a
    # partial install satisfies without providing the trace_processor bindings.
    try:
        from perfetto.common.exceptions import PerfettoException
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    except ImportError as exc:
        pytest.skip(f"Perfetto trace processor bindings unavailable: {exc}")

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
        rows = list(trace_processor.query(_CPUTIME_SAMPLES_QUERY))
    finally:
        close = getattr(trace_processor, "close", None)
        if close is not None:
            close()

    return rows[0] if rows else None


_TIMER_SAMPLING_THREAD_COUNT_QUERY = """
    SELECT COUNT(DISTINCT track_id) AS thread_count
    FROM slice
    WHERE category = 'timer_sampling'
"""


def _query_timer_sampling_thread_count(perfetto_file: Path):
    """Return the number of distinct tracks with timer_sampling slices."""
    try:
        from perfetto.common.exceptions import PerfettoException
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    except ImportError as exc:
        pytest.skip(f"Perfetto trace processor bindings unavailable: {exc}")

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
        rows = list(trace_processor.query(_TIMER_SAMPLING_THREAD_COUNT_QUERY))
    finally:
        close = getattr(trace_processor, "close", None)
        if close is not None:
            close()

    return rows[0].thread_count if rows else 0


def _assert_sampling_thread_count(result, expected_thread_count: int) -> None:
    """Assert every workload thread - not just thread 0 - got a sampler.

    A thread spawned via pthread_create while a control-session trigger
    (e.g. TRACE_DELAY) holds an initial pause never reached configure() at
    all if the pthread_create gotcha itself was disabled by that pause -
    distinct from (and stricter than) merely having an unarmed timer.
    """
    if not check_use_perfetto():
        pytest.skip("Perfetto is disabled")

    perfetto_file = result.perfetto_file
    if perfetto_file is None or not perfetto_file.exists():
        pytest.fail("No Perfetto trace file was produced by the run")

    thread_count = _query_timer_sampling_thread_count(perfetto_file)
    if thread_count != expected_thread_count:
        pytest.fail(
            f"Expected {expected_thread_count} threads with timer_sampling "
            f"slices, found {thread_count}; a thread created while the "
            f"control session held an initial pause never got a sampler"
        )


def _assert_cputime_samples_after_resume(result) -> None:
    """Assert CPU-time sampling actually produced samples once the pause ended.

    The pause/resume path emits no log output, so asserting on the configure
    message only proves the sampler was set up - not that its timer was ever
    re-armed. The trace is the only place the resume is observable.
    """
    if not check_use_perfetto():
        pytest.skip("Perfetto is disabled")

    perfetto_file = result.perfetto_file
    if perfetto_file is None or not perfetto_file.exists():
        pytest.fail("No Perfetto trace file was produced by the run")

    row = _query_cputime_samples(perfetto_file)
    if row is None or not row.sample_count:
        pytest.fail(
            "No CPU-time samples were recorded after the pause elapsed; the "
            "timer was never re-armed on resume"
        )

    span_sec = (row.span_ns or 0) / 1.0e9
    if row.sample_count < CPUTIME_RESUME_MIN_SAMPLES:
        pytest.fail(
            f"Only {row.sample_count} CPU-time samples after resume; expected at "
            f"least {CPUTIME_RESUME_MIN_SAMPLES}"
        )
    if span_sec < CPUTIME_RESUME_MIN_SPAN_SEC:
        pytest.fail(
            f"CPU-time sampling spanned only {span_sec:.3f}s after resume; "
            f"expected at least {CPUTIME_RESUME_MIN_SPAN_SEC}s"
        )


OVERFLOW_RESUME_DELAY_SEC = 0.3

# Modest floor, not a bare "> 0": a single stray sample would pass vacuously.
OVERFLOW_RESUME_MIN_SAMPLES = 10

_OVERFLOW_SAMPLES_QUERY = """
    SELECT COUNT(*) AS sample_count
    FROM slice
    WHERE category = 'overflow_sampling'
"""


def _query_overflow_sample_count(perfetto_file: Path):
    """Return the number of overflow_sampling slices in a Perfetto trace."""
    try:
        from perfetto.common.exceptions import PerfettoException
        from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    except ImportError as exc:
        pytest.skip(f"Perfetto trace processor bindings unavailable: {exc}")

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
        rows = list(trace_processor.query(_OVERFLOW_SAMPLES_QUERY))
    finally:
        close = getattr(trace_processor, "close", None)
        if close is not None:
            close()

    return rows[0].sample_count if rows else 0


def _assert_overflow_samples_after_resume(result) -> None:
    """Assert overflow (perf-event) sampling produced samples after resume.

    Resuming re-initializes the perf event's signal ownership from whichever
    thread calls resume() - the control session's time_window worker, not
    the sampled thread - which misdirects delivery for the rest of the run
    if that rebinding is not a no-op.
    """
    if not check_use_perfetto():
        pytest.skip("Perfetto is disabled")

    perfetto_file = result.perfetto_file
    if perfetto_file is None or not perfetto_file.exists():
        pytest.fail("No Perfetto trace file was produced by the run")

    sample_count = _query_overflow_sample_count(perfetto_file)
    if sample_count < OVERFLOW_RESUME_MIN_SAMPLES:
        pytest.fail(
            f"Only {sample_count} overflow samples after resume; expected "
            f"at least {OVERFLOW_RESUME_MIN_SAMPLES}"
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rocpd_env() -> dict[str, str]:
    return {}


@pytest.fixture
def recursion_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "minimal"
    return [rules_dir / "recursion-rules.json"]


@pytest.fixture
def pthreads_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "minimal"
    return [rules_dir / "pthreads-rules.json"]


# =============================================================================
# Tests
# =============================================================================


class TestMinimal(RocprofsysTest):
    """Test minimal examples."""

    RECURSION_DEPTH = 100

    @pytest.mark.rocpd("rocpd_env")
    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument"])
    def test_recursion(self, mode, rocpd_env, recursion_rules):
        """
        Ensure that recursion traces are properly present in both
        perfetto and rocpd.
        """
        env = rocpd_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "minimal-recursion",
            env=env,
            binary_rewrite_args=["--min-instructions", "0"],
            runtime_instrument_args=["--min-instructions", "0"],
            run_args=[str(self.RECURSION_DEPTH)],
        )
        self.assert_regex(result)

        # Look for the last line | recurse | 1 | <depth + 1> |
        deepest_depth = self.RECURSION_DEPTH + 1
        self.assert_perfetto(
            result,
            subtest_name="Perfetto Recursion Validation",
            categories=["host"],
            pass_regex=[rf"\|_recurse\s+\|\s+1\s+\|\s+{deepest_depth}\s+\|"],
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd Recursion Validation",
            rules_files=recursion_rules,
        )

        # Every 'recurse' frame carries a source_object trace-arg
        # with the value 'minimal-recursion'
        self.assert_perfetto(
            result,
            subtest_name="Perfetto Debug Validation",
            key_names=["source_object"],
            key_counts=[deepest_depth],
            pass_regex=[
                r"key\s*::\s*debug\.source_object",
                r"string_value\s*::\s*minimal-recursion",
            ],
        )

    @pytest.mark.rocpd("rocpd_env")
    @pytest.mark.parametrize("mode", ["binary_rewrite", "runtime_instrument", "sys_run"])
    def test_pthreads(self, mode, rocpd_env, pthreads_rules):
        """
        Ensure that pthread_create gotcha arguments (the incoming arg0
        annotation and the return value) are stored in the trace cache and
        surfaced in both perfetto and rocpd.
        """
        env = rocpd_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "minimal-pthreads",
            env=env,
            binary_rewrite_args=["--min-instructions", "0"],
            runtime_instrument_args=["--min-instructions", "0"],
        )
        self.assert_regex(result)

        # Both the pthread_create and pthread_join gotcha regions carry a
        # "return" trace-arg of 0 (two annotations total)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto pthread Return Validation",
            key_names=["return"],
            key_counts=[2],
            pass_regex=[
                r"key\s*::\s*debug\.return",
                r"string_value\s*::\s*0",
            ],
        )

        self.assert_rocpd(
            result,
            subtest_name="ROCpd pthread_create Args Validation",
            rules_files=pthreads_rules,
        )

    SLEEP_ITERATIONS = 60
    SAMPLING_ENV = {
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_SAMPLING_REALTIME": "ON",
        "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "1000",
        "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
    }

    def _run_sleep_interrupts(self, env_overrides: dict[str, str]):
        env = dict(self.SAMPLING_ENV)
        env.update(env_overrides)
        return self.run_test(
            "sys_run",
            "minimal-sleep-interrupts",
            env=env,
            run_args=[str(self.SLEEP_ITERATIONS)],
        )

    def test_paused_sampling_leaves_sleeps_undisturbed(self):
        """
        A paused sampler must stop its timers, not merely discard the samples
        they produce. Pausing used to leave the timers armed, so the signals
        kept cutting the application's sleeps short for the whole window.

        Run as a matched pair: the active run establishes that this kernel
        does exhibit the interruptions at all, so the paused run's zero is
        meaningful rather than vacuous.
        """
        active = self._run_sleep_interrupts({})
        self.assert_regex(
            active,
            subtest_name="Active sampling interrupts sleeps",
            pass_regex=[r"sleep_interrupts:.*interrupted=[1-9]\d*"],
        )

        paused = self._run_sleep_interrupts({"ROCPROFSYS_TRACE_DELAY": "60.0"})
        self.assert_regex(
            paused,
            subtest_name="Paused sampling leaves sleeps undisturbed",
            pass_regex=[r"sleep_interrupts:.*interrupted=0 short_sleeps=0"],
        )

    def test_sampling_resumes_after_delay(self):
        """
        The delay above outlasts the whole run, so it never proves the timers
        come back. Here the delay elapses partway through, so any
        interruption in the aggregate count can only have come from sampling
        that resumed.
        """
        resumed = self._run_sleep_interrupts({"ROCPROFSYS_TRACE_DELAY": "0.5"})
        self.assert_regex(
            resumed,
            subtest_name="Sampling resumes once the delay elapses",
            pass_regex=[r"sleep_interrupts:.*interrupted=[1-9]\d*"],
        )

    # <nfib> <nthreads> <nitr>. Long enough that a thread accrues far more CPU
    # time than the delay below, so the CPU-time timer is guaranteed to fire.
    CPU_WORKLOAD_ARGS = ["20", "2", "100000"]

    def test_cputime_sampling_survives_pause_resume(self):
        """
        The sleep-interrupts tests above run realtime-only, so nothing here
        covered CPU-time sampling across a pause. Routing the CPU-time timer
        outside the sampler's trigger list once left SIGPROF with no handler,
        and the first expiry killed the target outright.

        The workload is CPU-bound so the timer genuinely expires, and the delay
        elapses partway through so the timer survives a full pause/resume
        cycle. run_test fails on a non-zero return code, which is what catches
        the SIGPROF kill; the regex keeps the test from passing vacuously if
        CPU-time sampling were silently skipped instead of exercised.

        Configuration alone does not prove the timer came back, so the trace is
        checked for samples that can only have been produced after the resume.
        """
        result = self.run_test(
            "sys_run",
            "parallel-overhead",
            env={
                "ROCPROFSYS_USE_SAMPLING": "ON",
                "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
                "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
                "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "300",
                "ROCPROFSYS_TRACE_DELAY": str(CPUTIME_RESUME_DELAY_SEC),
            },
            run_args=self.CPU_WORKLOAD_ARGS,
        )
        self.assert_regex(
            result,
            subtest_name="CPU-time sampling survives a pause/resume cycle",
            pass_regex=[r"Sampler for thread \d+ will be triggered .*of CPU-time"],
        )
        _assert_cputime_samples_after_resume(result)

    # 2 worker threads + thread 0 (main), all spawned before TRACE_DELAY elapses.
    WORKLOAD_THREAD_COUNT = 3

    def test_threads_created_during_delay_get_sampled(self):
        """
        Worker threads pthread_create'd while TRACE_DELAY holds its initial
        pause must still get a sampler configured - not merely an unarmed
        timer, but configure() running at all. The pause is a control-session
        vote that must never disable the pthread_create gotcha itself, since
        that gotcha is the only place a new thread's instrumentation is set
        up; only threads created after this delay elapses were ever visible
        to rocprofsys before the fix.
        """
        result = self.run_test(
            "sys_run",
            "parallel-overhead",
            env={
                "ROCPROFSYS_USE_SAMPLING": "ON",
                "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
                "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
                "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "300",
                "ROCPROFSYS_TRACE_DELAY": str(CPUTIME_RESUME_DELAY_SEC),
            },
            run_args=self.CPU_WORKLOAD_ARGS,
        )
        _assert_sampling_thread_count(result, self.WORKLOAD_THREAD_COUNT)

    @pytest.mark.overflow
    def test_overflow_sampling_survives_pause_resume(self):
        """
        Resuming overflow (perf-event) sampling after a pause re-initializes
        the perf event's signal ownership - see set_ready_signal() - from
        whichever thread calls resume(), not the sampled thread. Before the
        fix this permanently misdirected sample delivery for the rest of the
        run, unlike CPU-time/realtime sampling which bind their target
        thread explicitly at timer creation and are unaffected by which
        thread later calls resume().

        A software counter is used, not a hardware one, so this doesn't
        depend on PMU access being available on the CI host - matching
        test_overflow.py's own portability choice.
        """
        result = self.run_test(
            "sys_run",
            "parallel-overhead",
            env={
                "ROCPROFSYS_USE_SAMPLING": "ON",
                "ROCPROFSYS_SAMPLING_OVERFLOW": "ON",
                "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "PERF_COUNT_SW_TASK_CLOCK",
                "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
                "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
                "ROCPROFSYS_TRACE_DELAY": str(OVERFLOW_RESUME_DELAY_SEC),
            },
            run_args=self.CPU_WORKLOAD_ARGS,
        )
        _assert_overflow_samples_after_resume(result)
