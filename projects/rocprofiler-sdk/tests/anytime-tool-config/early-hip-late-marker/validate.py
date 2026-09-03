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
validate.py -- PyTest validation for the early-tool + late-marker-tool test.

  Tool A (early, HIP-only) captured HIP API records and NO markers.
  Tool B (late, marker) captured marker records from after it registered, and no HIP API.

Marker counts are not a clean phase multiple (roctx push+mark+pop = 3 records, and a range
can straddle the registration boundary), so the count proof here is `> 0` + a timing check
rather than an exact phase-2 count.
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
    """Tool A enabled HIP_API_BUFFERED only: HIP records present, no markers."""
    sdk = _sdk(tool_a_data)

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert len(hip) > 0, "Tool A captured no HIP API records"

    markers = sdk["buffer_records"]["marker_api_traces"]
    assert (
        len(markers) == 0
    ), f"Tool A captured {len(markers)} marker records but only enabled HIP_API_BUFFERED"


def test_tool_b_captured_marker_only(tool_b_data):
    """Tool B enabled MARKER_API_BUFFERED only: markers present, no HIP API."""
    sdk = _sdk(tool_b_data)

    markers = sdk["buffer_records"]["marker_api_traces"]
    assert len(markers) > 0, "Tool B captured no marker records"

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert (
        len(hip) == 0
    ), f"Tool B captured {len(hip)} HIP API records but only enabled MARKER_API_BUFFERED"


def test_tool_b_markers_after_registration(tool_b_data):
    """Timestamp proof: every Tool B marker record falls after its registration.

    roctx records carry a single timestamp field; both start and end timestamps of a
    range-pop must land after the tool's init_time.
    """
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    fini_time = sdk["metadata"]["fini_time"]

    markers = sdk["buffer_records"]["marker_api_traces"]
    assert len(markers) > 0, "no marker records to check timestamps on"

    for itr in markers:
        assert (
            init_time < itr["start_timestamp"]
        ), f"marker started before Tool B registered (init_time={init_time}): {itr}"
        assert (
            fini_time > itr["end_timestamp"]
        ), f"marker ended after finalize (fini_time={fini_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
