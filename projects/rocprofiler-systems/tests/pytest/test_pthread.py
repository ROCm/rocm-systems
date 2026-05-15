# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
pthread tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.pthreads]


# =============================================================================
# pthread fixtures
# =============================================================================


@pytest.fixture
def pthread_env(lock_env) -> dict[str, str]:
    """Environment variables for pthread tests."""
    env = lock_env
    env["ROCPROFSYS_PROFILE"] = "ON"
    env["ROCPROFSYS_TRACE"] = "ON"
    env["ROCPROFSYS_COLLAPSE_THREADS"] = "OFF"
    env["ROCPROFSYS_SAMPLING_REALTIME"] = "ON"
    env["ROCPROFSYS_SAMPLING_REALTIME_FREQ"] = "10"
    env["ROCPROFSYS_SAMPLING_REALTIME_TIDS"] = "0"
    env["ROCPROFSYS_SAMPLING_KEEP_INTERNAL"] = "OFF"
    return env


@pytest.fixture
def pthread_timemory_env(lock_env) -> dict[str, str]:
    """Environment variables for pthread timemory tests."""
    env = lock_env
    env["ROCPROFSYS_FLAT_PROFILE"] = "ON"
    env["ROCPROFSYS_PROFILE"] = "ON"
    env["ROCPROFSYS_TRACE_LEGACY"] = "OFF"
    env["ROCPROFSYS_SAMPLING_KEEP_INTERNAL"] = "OFF"
    return env


# =============================================================================
# pthread tests
# =============================================================================


class TestPthreads(RocprofsysTest):
    OVERHEAD_LOCKS_PASS_REGEX = [
        r"wall_clock .*"
        r"\|_pthread_create .* 4 .*"
        r"\|_pthread_mutex_lock .* 1000 .*"
        r"\|_pthread_mutex_unlock .* 1000 .*"
        r"\|_pthread_mutex_lock .* 1000 .*"
        r"\|_pthread_mutex_unlock .* 1000 .*"
        r"\|_pthread_mutex_lock .* 1000 .*"
        r"\|_pthread_mutex_unlock .* 1000 .*"
        r"\|_pthread_mutex_lock .* 1000 .*"
        r"\|_pthread_mutex_unlock .* 1000 .*"
        r"\|_pthread_mutex_lock .* 1000 .*"
        r"\|_pthread_mutex_unlock .* 1000"
    ]
    OVERHEAD_LOCKS_TIMEMORY_PASS_REGEX = [
        r"start_thread (.*) 4 (.*) "
        r"pthread_mutex_lock (.*) 4000 (.*) "
        r"pthread_mutex_unlock (.*) 4000"
    ]

    TIMEMORY_REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "--min-instructions=32",
        "--dyninst-options",
        "InstrStackFrames",
        "SaveFPR",
        "TrampRecursive",
    ]

    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    def test_parallel_overhead_locks(self, mode, pthread_env):
        result = self.run_test(
            mode,
            "parallel-overhead-locks",
            env=pthread_env,
            rewrite_args=["-e", "-i", "256"],
            runtime_args=["-e", "-i", "256"],
            run_args=["30", "4", "1000"],
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=self.OVERHEAD_LOCKS_PASS_REGEX,
            runtime_pass_regex=self.OVERHEAD_LOCKS_PASS_REGEX,
        )

    @pytest.mark.parametrize(
        "mode", ["baseline", "sampling", "binary_rewrite", "sys_run"]
    )
    def test_parallel_overhead_locks_timemory(self, mode, pthread_timemory_env):
        result = self.run_test(
            mode,
            "parallel-overhead-locks",
            env=pthread_timemory_env,
            rewrite_args=self.TIMEMORY_REWRITE_ARGS,
            run_args=["10", "4", "1000"],
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=self.OVERHEAD_LOCKS_TIMEMORY_PASS_REGEX,
        )
