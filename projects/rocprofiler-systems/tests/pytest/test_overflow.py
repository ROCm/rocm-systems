# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
overflow tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.overflow]


# =============================================================================
# overflow fixtures
# =============================================================================


@pytest.fixture
def overflow_env() -> dict[str, str]:
    """Environment variables for overflow tests."""
    return {
        "ROCPROFSYS_VERBOSE": "2",
        "ROCPROFSYS_LOG_LEVEL": "trace",
        "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
        "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
        "ROCPROFSYS_SAMPLING_OVERFLOW": "ON",
        "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "PERF_COUNT_SW_CPU_CLOCK",
        "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ": "10000",
        "ROCPROFSYS_DEBUG_THREADING_GET_ID": "ON",
    }


# =============================================================================
# overflow tests
# =============================================================================


@pytest.mark.parametrize(
    "mode", ["sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
)
class TestOverflow(RocprofsysTest):
    PASS_REGEX = ["sampling_wall_clock.txt"]

    def test_parallel_overhead(self, mode, overflow_env):
        result = self.run_test(
            mode,
            "parallel-overhead",
            env=overflow_env,
            rewrite_args=["-e", "-v", "2"],
            runtime_args=["-e", "-v", "1"],
            run_args=["30", "2", "200"],
        )
        self.assert_regex(result, mode, pass_regex=self.PASS_REGEX)
