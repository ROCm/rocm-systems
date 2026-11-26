# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
overflow tests
"""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest, assert_resolved_settings

# =============================================================================
# overflow fixtures
# =============================================================================

OVERFLOW_MODES = [
    pytest.param("sampling", marks=pytest.mark.sampling),
    pytest.param("binary_rewrite", marks=pytest.mark.binary_rewrite),
    pytest.param("runtime_instrument", marks=pytest.mark.runtime_instrument),
    pytest.param("sys_run", marks=pytest.mark.sys_run),
]

_OVERFLOW_BASE_ENV = {
    "ROCPROFSYS_SAMPLING_CPUTIME": "OFF",
    "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
    "ROCPROFSYS_SAMPLING_OVERFLOW": "ON",
}


@pytest.fixture
def overflow_env() -> dict[str, str]:
    """Environment variables for overflow tests."""
    return {
        **_OVERFLOW_BASE_ENV,
        "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "PERF_COUNT_SW_CPU_CLOCK",
        "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ": "10000",
    }


# =============================================================================
# overflow tests
# =============================================================================


@pytest.mark.overflow
@pytest.mark.parametrize("mode", OVERFLOW_MODES)
class TestOverflow(RocprofsysTest):
    @pytest.mark.rocpd("overflow_env")
    def test_parallel_overhead(self, mode, overflow_env, validation_rules_dir):
        result = self.run_test(
            mode,
            "parallel-overhead",
            env=overflow_env,
            run_args=["30", "2", "200"],
        )
        assert_resolved_settings(
            result,
            {
                "ROCPROFSYS_SAMPLING_OVERFLOW": True,
                "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "PERF_COUNT_SW_CPU_CLOCK",
                "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ": 10000,
                "ROCPROFSYS_SAMPLING_CPUTIME": False,
                "ROCPROFSYS_SAMPLING_REALTIME": False,
            },
        )
        self.assert_timemory(
            result,
            file_name="sampling_wall_clock.json",
            metric="sampling_wall_clock",
            pass_regex=[r"fib[^|]*\|\s+[1-9]"],
            subtest_name="wall-clock profile contains fib",
        )
        self.assert_perfetto(
            result,
            categories=["overflow_sampling"],
            label_substrings=["fib"],
            subtest_name="perfetto overflow samples contain fib",
        )
        self.assert_rocpd(
            result,
            rules_files=[
                validation_rules_dir / "overflow" / "overflow-sampling-rules.json"
            ],
            subtest_name="rocpd overflow samples contain fib",
        )


@pytest.mark.timeout(60)
@pytest.mark.class_name("overflow-open-failure")
@pytest.mark.parametrize("mode", OVERFLOW_MODES)
class TestOverflowOpenFailure(RocprofsysTest):
    def test_exits_cleanly(self, mode, rocprof_config):
        if rocprof_config.capabilities.perf_events_usable:
            pytest.skip("perf events are usable; open failure is not reachable")

        result = self.run_test(
            mode,
            "parallel-overhead",
            env={
                **_OVERFLOW_BASE_ENV,
                "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "PERF_COUNT_HW_INSTRUCTIONS",
                "ROCPROFSYS_SAMPLING_OVERFLOW_FREQ": "10000000",
            },
            run_args=["10", "1", "100"],
            fail_on_pass=True,
            fail_on_not_found=True,
        )

        paranoid_path = "/proc/sys/kernel/perf_event_paranoid"
        try:
            with open(paranoid_path, encoding="ascii") as paranoid_file:
                paranoid_value = int(paranoid_file.read().strip())
            paranoid_phrase = f"current value is {paranoid_value}"
        except (OSError, ValueError):
            paranoid_phrase = f"could not read {paranoid_path}"

        self.assert_regex(
            result,
            pass_regex=[
                r"perf backend for overflow failed to activate",
                r"Failed to open perf event: (Permission denied|Operation not permitted)",
                r"Consider tweaking /proc/sys/kernel/perf_event_paranoid to 2 or less",
                paranoid_phrase,
                r"CAP_PERFMON",
            ],
            use_abort_fail_regex=False,
        )
        if result.returncode != 1:
            pytest.fail(f"overflow open failure should exit 1, got {result.returncode}")
