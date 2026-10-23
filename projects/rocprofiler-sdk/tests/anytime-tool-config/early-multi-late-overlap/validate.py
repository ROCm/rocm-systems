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
validate.py -- Two tools with overlapping service subsets.

  Tool A (early): HIP, HSA, dispatch, memcpy, marker
  Tool B (late):  HSA, dispatch, memcpy, marker, allocation

Overlap services (HSA, dispatch, memcpy, marker) must appear in BOTH tools. The
A-exclusive service (HIP) must be absent from B; the B-exclusive service (allocation)
must be absent from A. For the shared services, Tool B must capture exactly its Phase 2
share (count proof on the deterministic ones: dispatch, memcpy, allocate).
"""

import sys
import pytest

OP_ALLOCATE = 1


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _buf(data, key):
    return _sdk(data)["buffer_records"][key]


# ---- structure -------------------------------------------------------------


def test_structure(tool_a_data, tool_b_data):
    for data in (tool_a_data, tool_b_data):
        sdk = _sdk(data)
        assert "metadata" in sdk
        for key in ("pid", "main_tid", "init_time", "fini_time"):
            assert key in sdk["metadata"], f"metadata missing {key}"
        assert "buffer_records" in sdk


# ---- exclusive services ----------------------------------------------------


def test_hip_is_tool_a_exclusive(tool_a_data, tool_b_data):
    """HIP was enabled only by Tool A."""
    assert len(_buf(tool_a_data, "hip_api_traces")) > 0, "Tool A missing HIP records"
    assert (
        len(_buf(tool_b_data, "hip_api_traces")) == 0
    ), "Tool B captured HIP records but did not enable HIP_API_BUFFERED"


def test_alloc_is_tool_b_exclusive(tool_a_data, tool_b_data):
    """Memory allocation was enabled only by Tool B."""
    assert (
        len(_buf(tool_b_data, "memory_allocations")) > 0
    ), "Tool B missing memory allocation records"
    assert (
        len(_buf(tool_a_data, "memory_allocations")) == 0
    ), "Tool A captured allocations but did not enable MEMORY_ALLOCATION_BUFFERED"


# ---- overlapping services: present in BOTH ---------------------------------


@pytest.mark.parametrize(
    "key", ["hsa_api_traces", "kernel_dispatch", "memory_copies", "marker_api_traces"]
)
def test_overlap_service_in_both_tools(tool_a_data, tool_b_data, key):
    """Each overlapping service produced records for BOTH tools independently."""
    assert len(_buf(tool_a_data, key)) > 0, f"Tool A missing overlapping service {key}"
    assert len(_buf(tool_b_data, key)) > 0, f"Tool B missing overlapping service {key}"


# ---- Tool B count proofs (deterministic services) --------------------------


def test_tool_b_dispatch_count_is_phase2(tool_b_data, phase2_dispatches):
    captured = len(_buf(tool_b_data, "kernel_dispatch"))
    assert (
        captured == phase2_dispatches
    ), f"expected {phase2_dispatches} dispatches (phase 2), got {captured}"


def test_tool_b_copy_count_is_phase2(tool_b_data, phase2_copies):
    captured = len(_buf(tool_b_data, "memory_copies"))
    assert (
        captured == phase2_copies
    ), f"expected {phase2_copies} memory copies (phase 2), got {captured}"


def test_tool_b_allocate_count_is_phase2(tool_b_data, phase2_allocates):
    allocates = [
        r
        for r in _buf(tool_b_data, "memory_allocations")
        if r["operation"] == OP_ALLOCATE
    ]
    assert (
        len(allocates) == phase2_allocates
    ), f"expected {phase2_allocates} allocate records (phase 2), got {len(allocates)}"


# ---- Tool B timing proof on the shared deterministic services --------------


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
