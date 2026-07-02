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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys

import pytest

HOST_TRAP_COLUMNS = [
    "Sample_Timestamp",
    "Exec_Mask",
    "Dispatch_Id",
    "Instruction",
    "Instruction_Comment",
    "Correlation_Id",
]

MIN_SAMPLES = 100
MIN_V_MOV_B32_SAMPLES = 100
MIN_V_MOV_B32_RATIO = 0.30


def _tool(json_data):
    tool = json_data["rocprofiler-sdk-tool"]
    return tool[0] if isinstance(tool, list) else tool


def test_validate_pc_sampling_selected_regions_csv(pc_csv):
    # CSV schema, non-trivial volume, and sane per-row values.
    assert (
        list(pc_csv.columns) == HOST_TRAP_COLUMNS
    ), f"unexpected columns: {list(pc_csv.columns)}"
    assert len(pc_csv) >= MIN_SAMPLES, f"too few samples: {len(pc_csv)}"
    assert (pc_csv["Exec_Mask"] > 0).all(), "Exec_Mask must be > 0"
    assert (pc_csv["Dispatch_Id"] > 0).all(), "Dispatch_Id must be > 0"
    assert (pc_csv["Correlation_Id"] >= 0).all(), "Correlation_Id must be >= 0"


def test_validate_pc_sampling_selected_regions_json(pc_csv, json_data):
    tool = _tool(json_data)
    records = tool["buffer_records"]["pc_sample_host_trap"]

    # JSON buffer is present and agrees with the CSV record count.
    assert len(records) > 0, "no host_trap PC sampling records in JSON"
    assert len(records) == len(
        pc_csv
    ), f"CSV rows ({len(pc_csv)}) != JSON records ({len(records)})"

    # pc_sampling_kernel is a v_mov_b32 loop -> samples must be dominated by it,
    # confirming the right kernel was sampled and instructions decoded correctly.
    instructions = tool["strings"]["pc_sample_instructions"]
    v_mov_b32_count = 0
    for sample in records:
        inst_index = sample["inst_index"]
        if inst_index >= 0 and instructions[inst_index].startswith("v_mov_b32"):
            v_mov_b32_count += 1

    assert (
        v_mov_b32_count >= MIN_V_MOV_B32_SAMPLES
    ), f"expected >= {MIN_V_MOV_B32_SAMPLES} v_mov_b32 samples, got {v_mov_b32_count}"
    ratio = v_mov_b32_count / len(records)
    assert (
        ratio >= MIN_V_MOV_B32_RATIO
    ), f"expected v_mov_b32 samples >= {MIN_V_MOV_B32_RATIO:.0%}, got {ratio:.2%}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
