# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
rewrite caller tests
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.rewrite_caller]


# =============================================================================
# rewrite caller tests
# =============================================================================


@pytest.mark.caller_include
class TestRewriteCaller(RocprofsysTest):
    REWRITE_ARGS = [
        "-e",
        "-i",
        "256",
        "--caller-include",
        "^inner",
        "-v",
        "2",
        "--print-instrumented",
        "functions",
    ]
    BASELINE_PASS_REGEX = ["number of calls made = 17"]
    REWRITE_PASS_REGEX = [
        r"\[function\]\[Forcing\] caller-include-regex :: 'outer'",
        r">>> ._outer ([ \|]+) 17",
    ]

    @pytest.mark.parametrize("mode", ["baseline", "binary_rewrite", "sys_run"])
    def test(self, mode):
        result = self.run_test(
            mode,
            "rewrite-caller",
            env={"ROCPROFSYS_COUT_OUTPUT": "ON"},
            rewrite_args=self.REWRITE_ARGS,
            run_args=["17"],
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=self.REWRITE_PASS_REGEX,
            rewrite_run_pass_regex=self.REWRITE_PASS_REGEX,
        )
