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
validate.py -- PyTest validation for the direct-HSA early + late tool test.

Both tools enable HSA_API_BUFFERED and the workload is driven entirely by direct HSA
calls (no HIP). Proves the HSA dispatch table is captured by a late-registered tool:

  Tool A (early) captured HSA API records across both phases.
  Tool B (late)  captured HSA API records too, ALL after its registration (Phase 2 only),
    and strictly fewer than Tool A (which also saw Phase 1).
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def _hsa(data):
    return _sdk(data)["buffer_records"]["hsa_api_traces"]


def test_structure(tool_a_data, tool_b_data):
    for data in (tool_a_data, tool_b_data):
        sdk = _sdk(data)
        assert "metadata" in sdk
        for key in ("pid", "main_tid", "init_time", "fini_time"):
            assert key in sdk["metadata"], f"metadata missing {key}"
        assert "buffer_records" in sdk


def test_both_tools_captured_hsa(tool_a_data, tool_b_data):
    """Both tools enabled HSA_API_BUFFERED and saw HSA records from the direct workload."""
    assert len(_hsa(tool_a_data)) > 0, "Tool A captured no HSA API records"
    assert len(_hsa(tool_b_data)) > 0, "Tool B captured no HSA API records"


def test_tool_b_did_not_enable_other_services(tool_b_data):
    """Tool B enabled only HSA: HIP / kernel-dispatch / memcpy must be empty."""
    b = _sdk(tool_b_data)["buffer_records"]
    for key in ("hip_api_traces", "kernel_dispatch", "memory_copies"):
        assert len(b[key]) == 0, f"Tool B unexpectedly captured {key}"


def test_late_tool_saw_fewer_than_early(tool_a_data, tool_b_data):
    """Tool B missed Phase 1, so it captured strictly fewer HSA records than Tool A."""
    a_count = len(_hsa(tool_a_data))
    b_count = len(_hsa(tool_b_data))
    assert b_count < a_count, (
        f"late tool B ({b_count}) should capture fewer HSA records than early tool A "
        f"({a_count}); it missed phase 1"
    )


def test_late_tool_records_after_registration(tool_b_data):
    """Timestamp proof: every Tool B HSA record falls after its registration."""
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    fini_time = sdk["metadata"]["fini_time"]

    records = _hsa(tool_b_data)
    assert len(records) > 0, "no HSA records to check timestamps on"

    for itr in records:
        assert (
            init_time < itr["start_timestamp"]
        ), f"HSA record before Tool B registered (init_time={init_time}): {itr}"
        assert (
            fini_time > itr["end_timestamp"]
        ), f"HSA record after finalize (fini_time={fini_time}): {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
