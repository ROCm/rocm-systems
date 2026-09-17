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
validate.py -- PyTest validation for the early-tool + late-memory-copy-tool test.

  Tool A (early, HIP-only) captured HIP API records and NO memory copies.
  Tool B (late, memory-copy) captured exactly the post-registration copies, on buffers
    associated with streams that already existed when Tool B registered.
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
    """Tool A enabled HIP_API_BUFFERED only: HIP records present, no copies."""
    sdk = _sdk(tool_a_data)

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert len(hip) > 0, "Tool A captured no HIP API records"

    copies = sdk["buffer_records"]["memory_copies"]
    assert (
        len(copies) == 0
    ), f"Tool A captured {len(copies)} memory copies but only enabled HIP_API_BUFFERED"


def test_tool_b_captured_memcpy_only(tool_b_data):
    """Tool B enabled MEMORY_COPY_BUFFERED only: copies present, no HIP API."""
    sdk = _sdk(tool_b_data)

    copies = sdk["buffer_records"]["memory_copies"]
    assert len(copies) > 0, "Tool B captured no memory copy records"

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert len(hip) == 0, (
        f"Tool B captured {len(hip)} HIP API records but only enabled "
        "MEMORY_COPY_BUFFERED"
    )


def test_tool_b_copy_count_is_phase2_only(tool_b_data, phase1_copies, phase2_copies):
    """Count proof: Tool B captured exactly the post-registration copies."""
    sdk = _sdk(tool_b_data)
    captured = len(sdk["buffer_records"]["memory_copies"])

    assert captured == phase2_copies, (
        f"expected exactly {phase2_copies} memory copies (phase 2 only), got {captured}; "
        f"phase 1 ({phase1_copies}) should have been missed"
    )


def test_tool_b_copies_after_registration(tool_b_data):
    """Timestamp proof: every Tool B copy falls after its registration."""
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    fini_time = sdk["metadata"]["fini_time"]

    copies = sdk["buffer_records"]["memory_copies"]
    assert len(copies) > 0, "no copies to check timestamps on"

    for itr in copies:
        assert itr["start_timestamp"] < itr["end_timestamp"], f"bad span: {itr}"
        assert (
            init_time < itr["start_timestamp"]
        ), f"copy started before Tool B registered (init_time={init_time}): {itr}"
        assert (
            fini_time > itr["end_timestamp"]
        ), f"copy ended after finalize (fini_time={fini_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
