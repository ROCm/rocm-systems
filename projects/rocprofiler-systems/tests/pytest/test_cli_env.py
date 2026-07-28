# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the --env flag of rocprof-sys-run and rocprof-sys-sample.

--env sets environment variables for the target process in VARIABLE=VALUE form
and is repeatable. The launcher echoes every variable it updated as
"ROCPROFSYS: KEY=value" on stderr before exec, so a non-instrumentable target
like ``ls`` is sufficient to observe the effect.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.env_flag]

TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


# ============================================================================
# --env sets variables for the target process
# ----------------------------------------------------------------------------
# Each --env VARIABLE=VALUE flows through argparse update_env into the launched
# environment and is echoed as an updated variable. Repeating --env sets each
# variable independently.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("cli-env-sets-vars")
class TestEnvFlagSetsVars(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_repeated_env_are_set(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "--env",
                "ROCPROFSYS_CPU_FREQ_ENABLED=YES",
                "--env",
                "ROCPROFSYS_SAMPLING_CPUS=0",
                "--",
                "ls",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_CPU_FREQ_ENABLED=YES",
                r"ROCPROFSYS_SAMPLING_CPUS=0",
            ],
        )


# ============================================================================
# --env rejects values not in VARIABLE=VALUE form
# ----------------------------------------------------------------------------
# A missing '=' or an empty variable name (leading '=') is an argument error;
# the launcher exits non-zero before exec with a message naming the offending
# value.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("cli-env-malformed")
class TestEnvFlagRejectsMalformed(RocprofsysTest):
    @pytest.mark.parametrize(
        "bad_value",
        [
            pytest.param("NOEQUALS", id="missing-equals"),
            pytest.param("=VALUE", id="empty-name"),
        ],
    )
    @pytest.mark.parametrize("target", TARGETS)
    def test_malformed_env_rejected(self, target, bad_value):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--env", bad_value, "--", "ls"],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"not in form VARIABLE=VALUE"],
            use_abort_fail_regex=False,  # negative test intentionally exits non-zero
        )
