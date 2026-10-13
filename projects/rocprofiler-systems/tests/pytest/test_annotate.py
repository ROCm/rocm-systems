# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Annotate tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.annotate]

# =============================================================================
# Annotate fixtures
# =============================================================================


@pytest.fixture
def annotate_papi_condition(rocprof_config) -> bool:
    """Check if PAPI is available and usable."""
    caps = rocprof_config.capabilities
    return caps.papi_availability and caps.perf_events_usable


@pytest.fixture
def annotate_env(annotate_papi_condition) -> dict[str, str]:
    """Environment variables for Annotate tests."""
    env = {
        "ROCPROFSYS_USE_SAMPLING": "OFF",
    }
    if annotate_papi_condition:
        env["ROCPROFSYS_TIMEMORY_COMPONENTS"] = "thread_cpu_clock papi_array"
        env["ROCPROFSYS_PAPI_EVENTS"] = "perf::PERF_COUNT_SW_CPU_CLOCK"
    else:
        env["ROCPROFSYS_TIMEMORY_COMPONENTS"] = "thread_cpu_clock"
    return env


# =============================================================================
# Annotate tests
# =============================================================================


@pytest.mark.annotate
@pytest.mark.parametrize("mode", ["sampling", "binary_rewrite", "sys_run"])
class TestAnnotate(RocprofsysTest):
    BINARY_REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "-R",
        "run",
        "--allow-overlapping",
        "--print-available",
        "functions",
        "--print-overlapping",
        "functions",
        "--print-excluded",
        "functions",
        "--print-instrumented",
        "functions",
        "--print-instructions",
    ]
    RUN_ARGS = ["30", "2", "200"]

    def test_parallel_overhead(self, mode, annotate_env):

        result = self.run_test(
            mode,
            "parallel-overhead",
            env=annotate_env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            run_args=self.RUN_ARGS,
        )
        self.assert_regex(result)
        # Perfetto annotation key checks (thread_cpu_clock, PAPI) return with the
        # annotations follow-up.
