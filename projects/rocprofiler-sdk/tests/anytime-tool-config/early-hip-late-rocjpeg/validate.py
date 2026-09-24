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
validate.py -- PyTest validation for the early-tool + late-rocJPEG-tool test.

  Tool A (early, HIP-only) captured HIP API records and NO rocJPEG.
  Tool B (late, rocJPEG) captured rocJPEG API records from after it registered, and no HIP.
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


def test_tool_a_captured_hip_only(tool_a_data):
    assert len(_buf(tool_a_data, "hip_api_traces")) > 0, "Tool A missing HIP records"
    assert len(_buf(tool_a_data, "rocjpeg_api_traces")) == 0, "Tool A leaked rocJPEG"


def test_tool_b_captured_rocjpeg_only(tool_b_data):
    assert (
        len(_buf(tool_b_data, "rocjpeg_api_traces")) > 0
    ), "Tool B missing rocJPEG records"
    assert len(_buf(tool_b_data, "hip_api_traces")) == 0, "Tool B leaked HIP"


def test_tool_b_records_after_registration(tool_b_data):
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    records = _buf(tool_b_data, "rocjpeg_api_traces")
    assert len(records) > 0, "no rocJPEG records to check timestamps on"
    for itr in records:
        assert (
            init_time < itr["start_timestamp"]
        ), f"rocJPEG record before Tool B registered (init_time={init_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
