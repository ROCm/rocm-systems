# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Regression tests for the shared tool_runner used by rocprof-sys-run and
rocprof-sys-sample. Each test pins behavior that previously broke or that the
unification refactor consolidated; failures here indicate a regression in
tool_runner, argparse env-update semantics, or the conflict-detection path.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


# ============================================================================
# update_env(REPLACE) must not leave duplicate KEY= entries
# ----------------------------------------------------------------------------
# Regression for the bug fixed in this refactor: a shell-exported value plus a
# preset that REPLACEs the same key used to produce "KEY=preset_value:shell_value"
# after consolidate_env_entries joined the two surviving entries. The original
# reproducer was ROCPROFSYS_TRACE=true + --preset=profile-only, which silently
# turned tracing back on under "profile only".
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-replace-env")
class TestReplaceEnvNoDuplicates(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_preset_replaces_shell_env(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            env={"ROCPROFSYS_TRACE": "true"},
            run_args=["--preset=profile-only", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"ROCPROFSYS_TRACE=false(?![^\n]*:)"],
            fail_regex=[r"ROCPROFSYS_TRACE=\S*:\S*"],
        )


# ============================================================================
# --profile and --flat-profile must not be accepted together
# ----------------------------------------------------------------------------
# argparse declares the conflict via .conflicts({"flat-profile"}); tool_runner's
# apply_post_parse keeps a defensive throw as a second line of defense. Either
# layer must reject the combination with a non-zero exit code and a message
# mentioning the conflict.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-profile-conflict")
class TestProfileFlatProfileConflict(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_profile_and_flat_profile_rejected(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--profile", "--flat-profile", "--", "ls"],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--profile.*conflicts.*--flat-profile"],
        )
