#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""
validate.py -- the owner's worked example (overlapping subsets incl. RCCL).

  Tool A (early): HIP + RCCL
  Tool B (late):  HIP + RCCL + memory copy + kernel dispatch

  overlap (HIP, RCCL): present in BOTH tools
  B-only (memory copy, kernel dispatch): present in B, absent from A
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _buf(data, key):
    return _sdk(data)["buffer_records"][key]


def test_structure(tool_a_data, tool_b_data):
    for data in (tool_a_data, tool_b_data):
        sdk = _sdk(data)
        assert "metadata" in sdk
        for key in ("pid", "init_time", "fini_time"):
            assert key in sdk["metadata"], f"metadata missing {key}"
        assert "buffer_records" in sdk


# ---- overlap services present in BOTH ---------------------------------------


@pytest.mark.parametrize("key", ["hip_api_traces", "rccl_api_traces"])
def test_overlap_service_in_both(tool_a_data, tool_b_data, key):
    assert len(_buf(tool_a_data, key)) > 0, f"Tool A missing overlapping service {key}"
    assert len(_buf(tool_b_data, key)) > 0, f"Tool B missing overlapping service {key}"


# ---- B-exclusive services --------------------------------------------------


def test_dispatch_and_copy_are_tool_b_exclusive(tool_a_data, tool_b_data):
    assert len(_buf(tool_b_data, "kernel_dispatch")) > 0, "Tool B missing dispatches"
    assert len(_buf(tool_b_data, "memory_copies")) > 0, "Tool B missing copies"
    assert len(_buf(tool_a_data, "kernel_dispatch")) == 0, "Tool A leaked dispatches"
    assert len(_buf(tool_a_data, "memory_copies")) == 0, "Tool A leaked copies"


# ---- Tool B count proofs ---------------------------------------------------


def test_tool_b_counts_are_phase2(tool_b_data, phase2_dispatches, phase2_copies):
    """Memory copies are deterministic (exact). Kernel dispatches are a lower bound:
    RCCL all-reduce launches its own GPU kernels, so the count is the explicit kernels
    plus RCCL's internal kernels."""
    disp = len(_buf(tool_b_data, "kernel_dispatch"))
    copies = len(_buf(tool_b_data, "memory_copies"))
    assert disp >= phase2_dispatches, (
        f"expected at least {phase2_dispatches} dispatches (phase 2 explicit kernels), "
        f"got {disp}"
    )
    assert (
        copies == phase2_copies
    ), f"expected {phase2_copies} memory copies (phase 2), got {copies}"


def test_tool_b_records_after_registration(tool_b_data):
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    for key in ("kernel_dispatch", "memory_copies"):
        for itr in sdk["buffer_records"][key]:
            assert (
                init_time < itr["start_timestamp"]
            ), f"{key} record before Tool B registered (init_time={init_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
