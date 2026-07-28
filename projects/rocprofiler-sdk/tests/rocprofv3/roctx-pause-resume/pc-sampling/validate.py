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
    Verify that every resumed profiling window samples its pc_sampling_kernel
    dispatch, and that those windows collectively contain a non-trivial number
    of v_mov_b32 samples.
    """
    data = json_data["rocprofiler-sdk-tool"]

    pc_sampling_key = "pc_sample_host_trap"
    assert (
        pc_sampling_key in data["buffer_records"]
    ), f"No '{pc_sampling_key}' key found in buffer_records"

    samples = data["buffer_records"][pc_sampling_key]
    assert len(samples) > 0, "Expected at least one PC sampling record"

    pc_sampling_kernel_ids = {
        symbol["kernel_id"]
        for symbol in data["kernel_symbols"]
        if "pc_sampling_kernel" in symbol["kernel_name"]
    }
    assert pc_sampling_kernel_ids, "pc_sampling_kernel is missing from kernel metadata"

    target_dispatch_ids = {
        record["dispatch_info"]["dispatch_id"]
        for record in data["buffer_records"]["kernel_dispatch"]
        if record["dispatch_info"]["kernel_id"] in pc_sampling_kernel_ids
    }
    assert len(target_dispatch_ids) == 4, (
        "Expected one pc_sampling_kernel dispatch in each of four resumed "
        f"profiling windows, got {sorted(target_dispatch_ids)}"
    )

    instructions = data["strings"]["pc_sample_instructions"]

    v_mov_b32_dispatch_counts = {}
    for sample in samples:
        inst_index = sample["inst_index"]
        if inst_index >= 0 and instructions[inst_index].startswith("v_mov_b32"):
            dispatch_id = sample["record"]["dispatch_id"]
            v_mov_b32_dispatch_counts[dispatch_id] = (
                v_mov_b32_dispatch_counts.get(dispatch_id, 0) + 1
            )

    sampled_target_dispatches = target_dispatch_ids & set(v_mov_b32_dispatch_counts)
    assert sampled_target_dispatches == target_dispatch_ids, (
        "Expected v_mov_b32 samples from every pc_sampling_kernel dispatch; "
        f"sampled {sorted(sampled_target_dispatches)}, expected "
        f"{sorted(target_dispatch_ids)}"
    )

    target_samples = [
        sample
        for sample in samples
        if sample["record"]["dispatch_id"] in target_dispatch_ids
    ]
    v_mov_b32_count = sum(
        count
        for dispatch_id, count in v_mov_b32_dispatch_counts.items()
        if dispatch_id in target_dispatch_ids
    )

    assert (
        v_mov_b32_count >= 100
    ), f"Expected at least 100 target-kernel v_mov_b32 samples, got {v_mov_b32_count}"

    v_mov_b32_ratio = v_mov_b32_count / len(target_samples)
    assert v_mov_b32_ratio >= 0.30, (
        "Expected v_mov_b32 to be at least 30% of target-kernel samples, "
        f"got {v_mov_b32_ratio:.2%}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
