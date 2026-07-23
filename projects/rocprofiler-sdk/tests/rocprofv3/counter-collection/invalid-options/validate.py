#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import subprocess
import sys

import pytest

APPLICATION_SUCCESS = "ROCPROFV3_INVALID_OPTIONS_APPLICATION_RAN"

INVALID_OPTION_CASES = [
    pytest.param(
        ["--pmc", "SQ_WAVES", "--input", "{pmc_groups_input}"],
        "Cannot specify both --pmc and (input file) pmc_groups",
        id="cli-pmc-with-input-pmc-groups",
    ),
    pytest.param(
        ["--spm-beta-enabled", "--spm", "SQ_WAVES", "--pmc", "SQ_WAVES"],
        "SPM feature cannot be enabled along with pc sampling or pmc counter collection",
        id="spm-with-pmc",
    ),
    pytest.param(
        [
            "--pmc",
            "SQ_WAVES",
            "--pmc",
            "GRBM_COUNT",
            "--collection-period",
            "0:1:1",
        ],
        "Multi-pass counter collection (multiple --pmc flags) is not compatible "
        "with --collection-period",
        id="multipass-with-collection-period",
    ),
    pytest.param(
        ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--pid", "1"],
        "Multi-pass counter collection (multiple --pmc flags) is not compatible "
        "with attach mode (--pid)",
        id="multipass-with-pid",
    ),
    pytest.param(
        ["--pmc", "SQ_WAVES", "--pmc"],
        "Each --pmc must specify at least one counter.",
        id="empty-pmc-group",
    ),
]


@pytest.mark.parametrize("tool_args, expected_error", INVALID_OPTION_CASES)
def test_invalid_counter_collection_options_fail_before_application(
    rocprofv3,
    pmc_groups_input,
    tool_args,
    expected_error,
):
    tool_args = [
        argument.format(pmc_groups_input=pmc_groups_input) for argument in tool_args
    ]
    application = [
        sys.executable,
        "-c",
        f"print({APPLICATION_SUCCESS!r})",
    ]

    result = subprocess.run(
        [rocprofv3, *tool_args, "--", *application],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert expected_error in output
    assert APPLICATION_SUCCESS not in output
    assert "Traceback" not in output


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
