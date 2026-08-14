#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""GPU-free unit tests for the rocprofv3 multi-pass helpers."""

import os
import sys

import pytest


def _write(tmp_path, name, text):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w") as ofs:
        ofs.write(text)
    return path


# --pmc with action="append": one flag -> [["A","B"]] (single-pass, later
# flattened by patch_args); two flags -> [["A"],["B"]] (multi-pass).
def test_single_pmc_flag_is_single_pass(rocprofv3):
    args, _ = rocprofv3.parse_arguments(
        ["--pmc", "SQ_WAVES", "GRBM_COUNT", "--", "./app"]
    )
    assert args.pmc == [["SQ_WAVES", "GRBM_COUNT"]]
    assert len(args.pmc) == 1, "one --pmc flag must be a single group (single-pass)"

    rocprofv3.patch_args(args)
    assert args.pmc == ["SQ_WAVES", "GRBM_COUNT"], "single-pass pmc must be flattened"


def test_multiple_pmc_flags_is_multipass(rocprofv3):
    args, _ = rocprofv3.parse_arguments(
        ["--pmc", "SQ_WAVES", "--pmc", "GRBM_COUNT", "--", "./app"]
    )
    assert args.pmc == [["SQ_WAVES"], ["GRBM_COUNT"]]
    assert len(args.pmc) > 1, "two --pmc flags must be two groups (multi-pass)"

    rocprofv3.patch_args(args)
    assert args.pmc == [
        ["SQ_WAVES"],
        ["GRBM_COUNT"],
    ], "multi-pass groups must be preserved"


def test_parse_input_json_three_jobs(rocprofv3, tmp_path):
    path = _write(
        tmp_path,
        "in.json",
        '{"jobs":[{"pmc":["SQ_WAVES"]},{"pmc":["GRBM_COUNT"]},{"pmc":["GRBM_GUI_ACTIVE"]}]}',
    )
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 3
    assert [j["sub_directory"] for j in jobs] == ["pass_", "pass_", "pass_"]
    assert [j["pmc"] for j in jobs] == [["SQ_WAVES"], ["GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]


def test_parse_input_yaml_three_jobs(rocprofv3, tmp_path):
    path = _write(
        tmp_path,
        "in.yml",
        "jobs:\n"
        "  - pmc:\n    - SQ_WAVES\n"
        "  - pmc:\n    - GRBM_COUNT\n"
        "  - pmc:\n    - GRBM_GUI_ACTIVE\n",
    )
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 3
    assert [j["sub_directory"] for j in jobs] == ["pass_", "pass_", "pass_"]
    assert [j["pmc"] for j in jobs] == [["SQ_WAVES"], ["GRBM_COUNT"], ["GRBM_GUI_ACTIVE"]]


def test_parse_input_text_uses_pmc_subdir(rocprofv3, tmp_path):
    path = _write(tmp_path, "in.txt", "pmc: SQ_WAVES\npmc: GRBM_COUNT\n")
    jobs = rocprofv3.parse_input(path)
    assert len(jobs) == 2
    assert [j["sub_directory"] for j in jobs] == ["pmc_", "pmc_"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
