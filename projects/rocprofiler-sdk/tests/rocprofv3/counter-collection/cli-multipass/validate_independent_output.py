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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import csv
import os


def _read_counter_names(path):
    assert os.path.isfile(path), f"expected counter output is missing: {path}"
    with open(path, "r") as inp:
        rows = list(csv.DictReader(inp))
    assert rows, f"counter output is empty: {path}"
    return {row["Counter_Name"] for row in rows}


def test_combined_cli_and_input_outputs_are_independent(cli_output_dir, input_output_dir):
    cli_pass = os.path.join(cli_output_dir, "pass_1")
    input_pass = os.path.join(input_output_dir, "pass_2")

    assert _read_counter_names(os.path.join(cli_pass, "cli_counter_collection.csv")) == {
        "GRBM_COUNT"
    }
    assert _read_counter_names(
        os.path.join(input_pass, "input_counter_collection.csv")
    ) == {"SQ_WAVES"}

    assert os.path.isfile(os.path.join(cli_pass, "cli_agent_info.csv"))
    assert os.path.isfile(os.path.join(input_pass, "input_agent_info.csv"))
    assert not os.path.exists(os.path.join(cli_output_dir, "pass_2"))
    assert not os.path.exists(os.path.join(input_output_dir, "pass_1"))
