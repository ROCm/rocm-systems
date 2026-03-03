# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
attach tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.attach]

# ====================================================================================== #
# Attach fixtures
# ====================================================================================== #


@pytest.fixture
def attach_env(rocprof_config) -> dict[str, str]:
    return {
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_OMPT": "ON",
        "ROCPROFSYS_USE_KOKKOSP": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
        "OMP_NUM_THREADS": str(rocprof_config.capabilities.num_procs),
        "ROCPROFSYS_OUTPUT_PATH": "rocprof-sys-tests-output",
        "ROCPROFSYS_OUTPUT_PREFIX": "attach/",
    }


# ====================================================================================== #
# Attach tests
# ====================================================================================== #


class TestAttach(RocprofsysTest):
    PASS_REGEX = [
        r"Outputting.*(perfetto-trace.proto).*Outputting.*(wall_clock.txt)",
    ]
    FAIL_REGEX = [
        r"Dyninst was unable to attach to the specified process",
    ]

    def test_parallel_overhead(self, attach_env, rocprof_config):
        script_path = rocprof_config.rocprofsys_tests_dir / "run-rocprof-sys-pid.sh"
        if not script_path.exists():
            pytest.skip("run-rocprof-sys-pid.sh not found")
        script_path = str(script_path)

        try:
            target = rocprof_config.get_target_executable("parallel-overhead")
        except FileNotFoundError:
            pytest.skip("parallel-overhead not found")

        command = [
            script_path,
            str(rocprof_config.rocprofsys_instrument),
            "-ME",
            r"\.c$",
            "-E",
            "fib",
            "-e",
            "-v",
            "1",
            "--label",
            "return",
            "args",
            "file",
            "-l",
            "--",
            str(target),
            "30",
            "8",
            "1000",
        ]
        result = self.run_test(
            "baseline",
            target,
            env=attach_env,
            command=command,
            timeout=300,
        )
        self.assert_regex(result, pass_regex=self.PASS_REGEX, fail_regex=self.FAIL_REGEX)
