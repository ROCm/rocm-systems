# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Thread limit tests.
"""

from __future__ import annotations
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
    """Get the thread limit values for the test."""
    # ROCPROFSYS_MAX_THREADS may have been explicitly set, so use that if it exists
    import os

    max_threads = os.getenv("ROCPROFSYS_MAX_THREADS")
    if max_threads:
        return int(max_threads)

    rocprof_config = get_rocprof_config()
    num_procs = rocprof_config.capabilities.num_procs
    if num_procs < 128:
        thread_count = 2048
    else:
        # Round up to nearest power of 2
        n = 16 * num_procs - 1
        n |= n >> 1
        n |= n >> 2
        n |= n >> 4
        n |= n >> 8
        n |= n >> 16
        n |= n >> 32
        thread_count = n + 1
    return thread_count


def get_overflow_thread_load_count() -> int:
    """Thread count for load test — scales with the configured thread limit."""
    return get_thread_limit() * OVERFLOW_THREAD_LOAD_MULTIPLIER


def get_expected_pass_value(thread_count: int, thread_limit: int) -> int:
    """Highest profiled thread index expected after internal/offset overhead."""
    max_profiled = min(thread_count - 1, thread_limit - 1)
    if thread_count == thread_limit // 2:
        return max_profiled
    return max_profiled - INTERNAL_THREAD_OFFSET


def get_expected_fail_value(thread_count: int, thread_limit: int) -> int:
    """Thread index that must not appear in profile output."""
    if thread_count >= thread_limit:
        return thread_limit + 1
    return thread_count + 1


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
@pytest.mark.parametrize(
    "thread_count",
    [
        get_thread_limit() // 2,
        get_thread_limit(),
        get_thread_limit() * 2,
    ],
    ids=["half", "at", "double"],
)
@pytest.mark.timeout(480)
@pytest.mark.class_name("thread-limit")
class TestThreadLimit(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "-i", "1024", "--label", "return", "args"]
    RUNTIME_INSTRUMENT_ARGS = ["-e", "-v", "1", "-i", "1024", "--label", "return", "args"]

    def test(self, mode, thread_count, thread_limit_env):
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=["35", "2", str(thread_count)],
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
        )
        thread_limit = get_thread_limit()
        pass_value = get_expected_pass_value(thread_count, thread_limit)
        fail_value = get_expected_fail_value(thread_count, thread_limit)

        self.assert_regex(
            result,
            mode,
            pass_regex=[f"\\|{pass_value}>>>"],
            fail_regex=[f"\\|{fail_value}>>>"],
        )


@pytest.mark.parametrize("mode", ["sampling", "sys_run"])
@pytest.mark.timeout(600)
@pytest.mark.class_name("thread-limit-load-test")
class TestThreadLimitLoadTest(RocprofsysTest):
    def test(self, mode, thread_limit_env, rocprof_config):
        concurrency = min(rocprof_config.capabilities.num_procs, 16)
        thread_count = get_overflow_thread_load_count()
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=["30", str(concurrency), str(thread_count)],
        )
        thread_limit = get_thread_limit()
        pass_value = get_expected_pass_value(thread_count, thread_limit)
        fail_value = get_expected_fail_value(thread_count, thread_limit)
        warning_re = get_thread_limit_warning_regex(thread_limit)
        self.assert_regex(
            result,
            mode,
            pass_regex=[f"\\|{pass_value}>>>", warning_re],
            fail_regex=[f"\\|{fail_value}>>>"],
        )
