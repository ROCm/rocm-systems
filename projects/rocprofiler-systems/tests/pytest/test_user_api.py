# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for the user API.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.user_api]

# =============================================================================
# User API Tests
# =============================================================================


@pytest.mark.parametrize(
    "mode", ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"]
)
class TestUserAPI(RocprofsysTest):
    REWRITE_ARGS = [
        "-e",
        "-v",
        "2",
        "-l",
        "--min-instructions=8",
        "-E",
        "custom_push_region",
    ]
    RUNTIME_ARGS = [
        "-e",
        "-v",
        "1",
        "-l",
        "--min-instructions=8",
        "-E",
        "custom_push_region",
        "--label",
        "file",
        "line",
        "return",
        "args",
    ]
    BASELINE_FAIL_REGEX = ["Pushing custom region"]
    REWRITE_FAIL_REGEX = ["0 instrumented loops in procedure"]
    CUSTOM_MARKER_PASS_REGEX = ["Pushing custom region :: run.10. x 1000"]

    def test(self, mode):
        result = self.run_test(
            mode,
            "user-api",
            run_args=["10", str(self.num_threads), "1000"],
            rewrite_args=self.REWRITE_ARGS,
            runtime_args=self.RUNTIME_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            baseline_fail_regex=self.BASELINE_FAIL_REGEX,
            rewrite_fail_regex=self.REWRITE_FAIL_REGEX,
            rewrite_run_pass_regex=self.CUSTOM_MARKER_PASS_REGEX,
            runtime_pass_regex=self.CUSTOM_MARKER_PASS_REGEX,
            sampling_pass_regex=self.CUSTOM_MARKER_PASS_REGEX,
        )
