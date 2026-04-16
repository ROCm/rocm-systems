#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import pytest


def test_validate_pc_sampling_roctx_pause_resume(json_data):
    """
    Minimal validation: verify that PC sampling still collects a non-trivial number of
    v_mov_b32 samples (at least 100, comprising at least 30% of all samples) when
    roctx pause/resume is used. This ensures that pause/resume does not silently
    suppress or corrupt PC sampling data.
    """
    data = json_data["rocprofiler-sdk-tool"]

    pc_sampling_key = "pc_sample_host_trap"
    assert (
        pc_sampling_key in data["buffer_records"]
    ), f"No '{pc_sampling_key}' key found in buffer_records"

    samples = data["buffer_records"][pc_sampling_key]
    assert len(samples) > 0, "Expected at least one PC sampling record"

    instructions = data["strings"]["pc_sample_instructions"]

    v_mov_b32_count = 0
    for sample in samples:
        inst_index = sample["inst_index"]
        if inst_index >= 0 and instructions[inst_index].startswith("v_mov_b32"):
            v_mov_b32_count += 1

    assert (
        v_mov_b32_count >= 100
    ), f"Expected at least 100 samples with v_mov_b32 instruction, got {v_mov_b32_count}"

    v_mov_b32_ratio = v_mov_b32_count / len(samples)
    assert (
        v_mov_b32_ratio >= 0.30
    ), f"Expected v_mov_b32 samples to be at least 30% of total, got {v_mov_b32_ratio:.2%}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
