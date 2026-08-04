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

# rocprof-sys may initialize internal/offset threads (sampling, ROCm, etc.)
# that consume thread slots without appearing as workload thread rows. Expect the
# highest profiled thread index to be at most thread_limit - INTERNAL_THREAD_OFFSET.
# The half-thread case is far below the limit, so use the exact highest launched index.
INTERNAL_THREAD_OFFSET = 20
OVERFLOW_THREAD_LOAD_MULTIPLIER = 8

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
# Helper Function
# ============================================================================


def get_thread_limit() -> int:
    """Get the thread limit for the test"""
    return get_rocprof_config().capabilities.max_threads


@dataclass(frozen=True)
class ThreadLimitCase:
    """A thread-limit scenario expressed as functions of the runtime thread limit.

    ``count`` takes the compile-time thread limit and returns the concrete
    thread count to launch. ``pass_value``/``fail_value`` take that already
    computed ``(count, limit)`` and return the highest profiled thread index
    expected in the output and a thread index that must NOT appear,
    respectively. Grouping them per-row keeps the expected output readable next
    to the input that produces it.
    """

    count: Callable[[int], int]
    pass_value: Callable[[int, int], int]
    fail_value: Callable[[int, int], int]


# Highest profiled index when the count is below the limit ("half"): the exact
# top launched index (count - 1). When at/over the limit, internal/offset
# threads evict some slots, so subtract INTERNAL_THREAD_OFFSET. The fail index
# is one past whichever cap bites first.
THREAD_LIMIT_CASES: dict[str, ThreadLimitCase] = {
    # ratio    count(limit)                          pass_value(count, limit)                    fail_value(count, limit)
    "half": ThreadLimitCase(
        count=lambda limit: limit // 2,
        pass_value=lambda count, limit: count - 1,
        fail_value=lambda count, limit: count + 1,
    ),
    "at": ThreadLimitCase(
        count=lambda limit: limit,
        pass_value=lambda count, limit: (limit - 1) - INTERNAL_THREAD_OFFSET,
        fail_value=lambda count, limit: limit + 1,
    ),
    "double": ThreadLimitCase(
        count=lambda limit: limit * 2,
        pass_value=lambda count, limit: (limit - 1) - INTERNAL_THREAD_OFFSET,
        fail_value=lambda count, limit: limit + 1,
    ),
    "load": ThreadLimitCase(
        count=lambda limit: limit * OVERFLOW_THREAD_LOAD_MULTIPLIER,
        pass_value=lambda count, limit: (limit - 1) - INTERNAL_THREAD_OFFSET,
        fail_value=lambda count, limit: limit + 1,
    ),
}


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
@pytest.mark.parametrize("thread_ratio", ["half", "at", "double"])
@pytest.mark.class_name("thread-limit")
class TestThreadLimit(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "-i", "1024", "--label", "return", "args"]
    RUNTIME_INSTRUMENT_ARGS = ["-e", "-v", "1", "-i", "1024", "--label", "return", "args"]

    def test(self, mode, thread_ratio, thread_limit_env):
        thread_limit = get_thread_limit()
        case = THREAD_LIMIT_CASES[thread_ratio]
        thread_count = case.count(thread_limit)
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=["35", "2", str(thread_count)],
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            pass_regex=[f"\\|{case.pass_value(thread_count, thread_limit)}>>>"],
            fail_regex=[f"\\|{case.fail_value(thread_count, thread_limit)}>>>"],
        )


@pytest.mark.parametrize("mode", ["sampling", "sys_run"])
@pytest.mark.timeout(600)
@pytest.mark.class_name("thread-limit-load-test")
class TestThreadLimitLoadTest(RocprofsysTest):
    def test(self, mode, thread_limit_env, rocprof_config):
        concurrency = min(rocprof_config.capabilities.num_procs, 16)
        thread_limit = get_thread_limit()
        case = THREAD_LIMIT_CASES["load"]
        thread_count = case.count(thread_limit)
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=["30", str(concurrency), str(thread_count)],
        )
        warning_re = get_thread_limit_warning_regex(thread_limit)
        self.assert_regex(
            result,
            mode,
            pass_regex=[
                f"\\|{case.pass_value(thread_count, thread_limit)}>>>",
                warning_re,
            ],
            fail_regex=[f"\\|{case.fail_value(thread_count, thread_limit)}>>>"],
        )
