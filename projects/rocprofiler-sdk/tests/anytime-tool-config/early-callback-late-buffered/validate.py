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
validate.py -- callback vs buffered delivery path for kernel dispatch.

Tool A (early) uses KERNEL_DISPATCH_CALLBACK; its dispatch records land in
`callback_records.kernel_dispatch`. Tool B (late) uses KERNEL_DISPATCH_BUFFERED; its
records land in `buffer_records.kernel_dispatch`. Proves both delivery paths work, and that
the buffered path works for a late-registered tool.

Cross-checks the two paths don't bleed into each other: Tool A's buffer path is empty and
Tool B's callback path is empty.
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


def test_tool_a_used_callback_path(tool_a_data):
    """Tool A enabled KERNEL_DISPATCH_CALLBACK: dispatches in callback_records only."""
    sdk = _sdk(tool_a_data)

    cb = sdk["callback_records"]["kernel_dispatch"]
    assert len(cb) > 0, "Tool A captured no callback kernel dispatch records"

    buf = sdk["buffer_records"]["kernel_dispatch"]
    assert (
        len(buf) == 0
    ), f"Tool A has {len(buf)} buffered dispatch records but enabled the CALLBACK path"


def test_tool_b_used_buffered_path(tool_b_data):
    """Tool B enabled KERNEL_DISPATCH_BUFFERED: dispatches in buffer_records only."""
    sdk = _sdk(tool_b_data)

    buf = sdk["buffer_records"]["kernel_dispatch"]
    assert len(buf) > 0, "Tool B captured no buffered kernel dispatch records"

    cb = sdk["callback_records"]["kernel_dispatch"]
    assert (
        len(cb) == 0
    ), f"Tool B has {len(cb)} callback dispatch records but enabled the BUFFERED path"


def test_tool_b_buffered_count_is_phase2(tool_b_data, phase2_count):
    """Count proof: the late buffered tool captured exactly the phase-2 dispatches."""
    sdk = _sdk(tool_b_data)
    captured = len(sdk["buffer_records"]["kernel_dispatch"])
    assert (
        captured == phase2_count
    ), f"expected {phase2_count} buffered dispatches (phase 2), got {captured}"


def test_tool_b_dispatches_after_registration(tool_b_data):
    """Timestamp proof: every late buffered dispatch starts after registration."""
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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
