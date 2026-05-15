# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
    return rocprof_config.capabilities.papi_availability and (
        rocprof_config.capabilities.perf_event_paranoid <= 3
        or rocprof_config.capabilities.cap_sys_admin
        or rocprof_config.capabilities.cap_perfmon
    )


@pytest.fixture
def annotate_env(annotate_papi_condition) -> dict[str, str]:
    """Environment variables for Annotate tests."""
    env = {
        "ROCPROFSYS_TRACE_LEGACY": "ON",
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
    REWRITE_ARGS = [
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

    def test_parallel_overhead(self, mode, annotate_env, annotate_papi_condition):

        result = self.run_test(
            mode,
            "parallel-overhead",
            env=annotate_env,
            rewrite_args=self.REWRITE_ARGS,
            run_args=self.RUN_ARGS,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                key_names=["thread_cpu_clock"],
                key_counts=[6],
            )

        if mode == "binary_rewrite":
            if annotate_papi_condition:
                key_names = ["perf::PERF_COUNT_SW_CPU_CLOCK", "thread_cpu_clock"]
                key_counts = [8, 8]
            else:
                key_names = ["thread_cpu_clock"]
                key_counts = [8]

            self.assert_perfetto(
                result,
                key_names=key_names,
                key_counts=key_counts,
            )
