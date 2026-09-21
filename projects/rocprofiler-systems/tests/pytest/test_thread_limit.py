# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Thread limit tests.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pytest
from conftest import RocprofsysTest, get_rocprof_config

pytestmark = [pytest.mark.thread_limit]

# rocprof-sys' own threads (sampling, ROCm) consume slots without producing
# workload rows, so back off by this much when at or over the limit.
INTERNAL_THREAD_OFFSET = 20

# Workload args shared by every case. Threads are created in batches of
# WORKLOAD_CONCURRENCY, so slots are consumed cumulatively without many being live.
WORKLOAD_FIB = 35
WORKLOAD_CONCURRENCY = 2

# ============================================================================
# Thread Limit Fixtures
# ============================================================================


@pytest.fixture
def thread_limit_env() -> dict[str, str]:
    """Environment variables for thread limit tests."""
    return {
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_SAMPLING_FREQ": "250",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,peak_rss,page_rss",
    }


# ============================================================================
# Thread Limit Case Matrix
# ============================================================================


@dataclass(frozen=True)
class ThreadLimitCase:
    """A thread-limit scenario expressed as functions of the runtime thread limit."""

    thread_count: Callable[[int], int]
    pass_value: Callable[[int, int], int]
    fail_value: Callable[[int, int], int]
    expect_exhausted_warning: bool


# Below the limit ("half") every launched thread is profiled, so the highest index
# is the exact top launched index and the absent index is one past it. At or over
# the limit, internal/offset threads consume slots, so back off by
# INTERNAL_THREAD_OFFSET and expect one past the limit to be absent. Only the
# over-limit row is required to emit the thread-limit warning.
THREAD_LIMIT_CASES: dict[str, ThreadLimitCase] = {
    "half": ThreadLimitCase(
        thread_count=lambda limit: limit // 2,
        pass_value=lambda thread_count, limit: thread_count - 1,
        fail_value=lambda thread_count, limit: thread_count + 1,
        expect_exhausted_warning=False,
    ),
    "at_limit": ThreadLimitCase(
        thread_count=lambda limit: limit,
        pass_value=lambda thread_count, limit: (limit - 1) - INTERNAL_THREAD_OFFSET,
        fail_value=lambda thread_count, limit: limit + 1,
        expect_exhausted_warning=False,
    ),
    "exhausted": ThreadLimitCase(
        thread_count=lambda limit: limit * 4,
        pass_value=lambda thread_count, limit: (limit - 1) - INTERNAL_THREAD_OFFSET,
        fail_value=lambda thread_count, limit: limit + 1,
        expect_exhausted_warning=True,
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================


def get_thread_limit() -> int:
    """Thread limit for the test, or skip when rocprof-sys-avail could not report it."""
    thread_limit = get_rocprof_config().capabilities.max_threads
    if not thread_limit:
        pytest.skip(
            "Could not determine ROCPROFSYS_MAX_THREADS from "
            "'rocprof-sys-avail --max-threads'"
        )
    return thread_limit


def get_thread_limit_warning_regex(thread_limit: int) -> str:
    """Regex for pthread_create_gotcha thread-limit warning in runner logs."""
    return (
        rf"\[warning\] Maximum allowed thread limit \({thread_limit}\) reached\. "
        r"Further profiling will be disabled to prevent resource exhaustion\. "
        r"Consider increasing the limit at compile time using the "
        r"ROCPROFSYS_MAX_THREADS CMake option\."
    )


# ============================================================================
# Thread Limit Tests
# ============================================================================


@pytest.mark.parametrize(
    "mode", ["sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
)
@pytest.mark.parametrize("thread_ratio", list(THREAD_LIMIT_CASES))
@pytest.mark.class_name("thread-limit")
class TestThreadLimit(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "-i", "1024", "--label", "return", "args"]
    RUNTIME_INSTRUMENT_ARGS = ["-e", "-v", "2", "-i", "1024", "--label", "return", "args"]

    def test(self, mode, thread_ratio, thread_limit_env):
        thread_limit = get_thread_limit()
        case = THREAD_LIMIT_CASES[thread_ratio]
        thread_count = case.thread_count(thread_limit)
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=[
                str(WORKLOAD_FIB),
                str(WORKLOAD_CONCURRENCY),
                str(thread_count),
            ],
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
        )
        pass_regex = [f"\\|{case.pass_value(thread_count, thread_limit)}>>>"]
        if case.expect_exhausted_warning:
            pass_regex.append(get_thread_limit_warning_regex(thread_limit))

        self.assert_regex(
            result,
            mode,
            pass_regex=pass_regex,
            fail_regex=[f"\\|{case.fail_value(thread_count, thread_limit)}>>>"],
        )
