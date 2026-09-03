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
validate.py -- PyTest validation for the early-tool + late-dispatch-tool test.

Proves the in-scope anytime-initialization behavior:
  Tool A (early, HIP-only) captured HIP API records and NO kernel dispatches.
  Tool B (late, kernel-dispatch) captured exactly the post-registration dispatches,
    including dispatches on streams that already existed when Tool B registered.
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _has_structure(sdk):
    assert "metadata" in sdk
    for key in ("pid", "main_tid", "init_time", "fini_time"):
        assert key in sdk["metadata"], f"metadata missing {key}"
    assert "callback_records" in sdk
    assert "buffer_records" in sdk


def test_tool_a_structure(tool_a_data):
    _has_structure(_sdk(tool_a_data))


def test_tool_b_structure(tool_b_data):
    _has_structure(_sdk(tool_b_data))


def test_tool_a_captured_hip_only(tool_a_data):
    """Tool A enabled HIP_API_BUFFERED only: HIP records present, no dispatches."""
    sdk = _sdk(tool_a_data)

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert len(hip) > 0, "Tool A captured no HIP API records"

    dispatches = sdk["buffer_records"]["kernel_dispatch"]
    assert len(dispatches) == 0, (
        f"Tool A captured {len(dispatches)} kernel dispatches but only enabled "
        "HIP_API_BUFFERED"
    )


def test_tool_b_captured_dispatch_only(tool_b_data):
    """Tool B enabled KERNEL_DISPATCH_BUFFERED: dispatches present, no HIP API."""
    sdk = _sdk(tool_b_data)

    dispatches = sdk["buffer_records"]["kernel_dispatch"]
    assert len(dispatches) > 0, "Tool B captured no kernel dispatch records"

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert len(hip) == 0, (
        f"Tool B captured {len(hip)} HIP API records but only enabled "
        "KERNEL_DISPATCH_BUFFERED"
    )


def test_tool_b_dispatch_count_is_phase2_only(tool_b_data, phase1_count, phase2_count):
    """Count proof: Tool B captured exactly the post-registration dispatches."""
    sdk = _sdk(tool_b_data)
    captured = len(sdk["buffer_records"]["kernel_dispatch"])

    assert captured == phase2_count, (
        f"expected exactly {phase2_count} dispatches (phase 2 only), got {captured}; "
        f"phase 1 ({phase1_count}) should have been missed"
    )


def test_tool_b_dispatches_after_registration(tool_b_data):
    """Timestamp proof: every Tool B dispatch falls after its registration."""
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    fini_time = sdk["metadata"]["fini_time"]

    dispatches = sdk["buffer_records"]["kernel_dispatch"]
    assert len(dispatches) > 0, "no dispatches to check timestamps on"

    for itr in dispatches:
        assert itr["start_timestamp"] < itr["end_timestamp"], f"bad span: {itr}"
        assert (
            init_time < itr["start_timestamp"]
        ), f"dispatch started before Tool B registered (init_time={init_time}): {itr}"
        assert (
            fini_time > itr["end_timestamp"]
        ), f"dispatch ended after finalize (fini_time={fini_time}): {itr}"


def test_tool_b_dispatch_ids_unique(tool_b_data):
    sdk = _sdk(tool_b_data)
    ids = [
        itr["dispatch_info"]["dispatch_id"]
        for itr in sdk["buffer_records"]["kernel_dispatch"]
    ]
    assert len(ids) == len(set(ids)), "duplicate dispatch ids found"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
