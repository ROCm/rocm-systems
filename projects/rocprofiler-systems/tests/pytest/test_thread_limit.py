# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Thread limit tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest, get_rocprof_config

pytestmark = [pytest.mark.thread_limit]

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


# Can't be a fixture as thread_count parametrized value is
# used at collection time
def get_thread_limit() -> int:
    """Get the thread limit values for the test."""
    rocprof_config = get_rocprof_config()
    num_procs = rocprof_config.capabilities.num_procs
    if num_procs < 8:
        thread_count = 128
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


# ============================================================================
# Thread Limit Tests
# ============================================================================


@pytest.mark.parametrize(
    "mode", ["sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
)
@pytest.mark.parametrize(
    "thread_count",
    [
        get_thread_limit() - 1,
        get_thread_limit() + 24,
        get_thread_limit(),
    ],
)
class TestThreadLimit(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "-i", "1024", "--label", "return", "args"]
    RUNTIME_ARGS = ["-e", "-v", "1", "-i", "1024", "--label", "return", "args"]

    def test(self, mode, thread_count, thread_limit_env):
        result = self.run_test(
            mode,
            "thread-limit",
            env=thread_limit_env,
            run_args=["35", "2", str(thread_count)],
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
            timeout=480,
        )
        thread_limit = get_thread_limit()
        pass_value = thread_count
        fail_value = thread_count + 1
        if thread_count >= thread_limit:
            pass_value = thread_limit - 1
            fail_value = thread_limit + 1

        self.assert_regex(
            result,
            mode,
            pass_regex=[f"\\|{pass_value}>>>"],
            fail_regex=[f"\\|{fail_value}>>>"],
        )
