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
validate.py -- Two distinct tools registering late in one process.

Three tools, three disjoint service subsets:
  Tool A  (early): HIP            -> HIP records only, no dispatch/copy
  Tool B1 (late):  kernel dispatch -> dispatch records only (phase 2), no HIP/copy
  Tool B2 (late):  memory copy     -> copy records only (phase 2), no HIP/dispatch

Proves multiple force_configure() calls in one process each yield an independent context
that captures exactly its own service subset.
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _buf(data, key):
    return _sdk(data)["buffer_records"][key]


def test_structure(tool_a_data, tool_b1_data, tool_b2_data):
    for data in (tool_a_data, tool_b1_data, tool_b2_data):
        sdk = _sdk(data)
        assert "metadata" in sdk
        for key in ("pid", "init_time", "fini_time"):
            assert key in sdk["metadata"], f"metadata missing {key}"
        assert "buffer_records" in sdk


# ---- each tool captured ONLY its own service -------------------------------


def test_tool_a_hip_only(tool_a_data):
    assert len(_buf(tool_a_data, "hip_api_traces")) > 0, "Tool A missing HIP records"
    assert len(_buf(tool_a_data, "kernel_dispatch")) == 0, "Tool A leaked dispatches"
    assert len(_buf(tool_a_data, "memory_copies")) == 0, "Tool A leaked copies"


def test_tool_b1_dispatch_only(tool_b1_data):
    assert len(_buf(tool_b1_data, "kernel_dispatch")) > 0, "Tool B1 missing dispatches"
    assert len(_buf(tool_b1_data, "hip_api_traces")) == 0, "Tool B1 leaked HIP"
    assert len(_buf(tool_b1_data, "memory_copies")) == 0, "Tool B1 leaked copies"


def test_tool_b2_memcpy_only(tool_b2_data):
    assert len(_buf(tool_b2_data, "memory_copies")) > 0, "Tool B2 missing copies"
    assert len(_buf(tool_b2_data, "hip_api_traces")) == 0, "Tool B2 leaked HIP"
    assert len(_buf(tool_b2_data, "kernel_dispatch")) == 0, "Tool B2 leaked dispatches"


# ---- late tools captured exactly their phase-2 share ----------------------


def test_tool_b1_dispatch_count_is_phase2(tool_b1_data, phase2_dispatches):
    captured = len(_buf(tool_b1_data, "kernel_dispatch"))
    assert (
        captured == phase2_dispatches
    ), f"expected {phase2_dispatches} dispatches (phase 2), got {captured}"


def test_tool_b2_copy_count_is_phase2(tool_b2_data, phase2_copies):
    captured = len(_buf(tool_b2_data, "memory_copies"))
    assert (
        captured == phase2_copies
    ), f"expected {phase2_copies} memory copies (phase 2), got {captured}"


# ---- timing proof for both late tools -------------------------------------


def test_late_tools_records_after_registration(tool_b1_data, tool_b2_data):
    for data, key in ((tool_b1_data, "kernel_dispatch"), (tool_b2_data, "memory_copies")):
        sdk = _sdk(data)
        init_time = sdk["metadata"]["init_time"]
        for itr in sdk["buffer_records"][key]:
            assert (
                init_time < itr["start_timestamp"]
            ), f"{key} record before its tool registered (init_time={init_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
