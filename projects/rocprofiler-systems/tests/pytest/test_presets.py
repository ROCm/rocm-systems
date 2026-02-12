# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Preset tests (verify presets work correctly with simple commands)
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

PRESETS = [
    "balanced",
    "profile-only",
    "detailed",
    "trace-hpc",
    "workload-trace",
    "sys-trace",
    "runtime-trace",
    "trace-gpu",
    "trace-openmp",
    "profile-mpi",
    "trace-hw-counters",
]

# ============================================================================
# Preset Tests
# ============================================================================


class TestPresets(RocprofsysTest):
    @pytest.mark.sampling
    @pytest.mark.parametrize("preset", PRESETS)
    def test_sample(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--{preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        --{preset}"],
        )

    @pytest.mark.sampling
    def test_sample_mutual_exclusion(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--balanced", "--profile-only", "-v", "2", "--", "ls"],
            timeout=30,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["Multiple preset modes specified", "Only ONE preset"],
        )

    @pytest.mark.sys_run
    @pytest.mark.parametrize("preset", PRESETS)
    def test_run(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--{preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        --{preset}"],
        )

    @pytest.mark.sys_run
    def test_run_mutual_exclusion(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--trace-hpc", "--workload-trace", "-v", "2", "--", "ls"],
            timeout=30,
            fail_on_pass=True,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["Multiple preset modes specified", "Only ONE preset"],
        )
