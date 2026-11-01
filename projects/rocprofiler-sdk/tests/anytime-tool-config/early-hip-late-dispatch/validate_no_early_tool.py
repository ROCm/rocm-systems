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
validate_no_early_tool.py -- PyTest validation for the late-dispatch tool running with
NO tool present when the queues were created.

The sibling validate.py covers the case where rocprofiler-sdk is already resident because
an early tool was preloaded, so HSA queues are intercepted at creation. This one removes
the early tool: nothing holds the SDK until the late tool calls
rocprofiler_force_configure(), by which time the application's queues already exist.

Those queues are invisible to hsa_queue_create interception and reachable only by doorbell
discovery, so the late tool records dispatches on them only if the queue is adopted into
the controller at discovery time. Without that, every phase-2 dispatch executes with the
write interceptor never running and the tool collects nothing -- the whole point of this
test is that it reports zero rather than passing quietly.
"""

import sys
import pytest


def _sdk(data):
    assert "rocprofiler-sdk-json-tool" in data
    return data["rocprofiler-sdk-json-tool"]


def test_structure(tool_b_data):
    sdk = _sdk(tool_b_data)
    assert "metadata" in sdk
    for key in ("pid", "main_tid", "init_time", "fini_time"):
        assert key in sdk["metadata"], f"metadata missing {key}"
    assert "buffer_records" in sdk


def test_dispatches_on_pre_existing_queues_are_captured(tool_b_data, phase2_count):
    """The regression this file exists for: zero here means queue adoption is absent."""
    sdk = _sdk(tool_b_data)
    captured = len(sdk["buffer_records"]["kernel_dispatch"])

    assert captured > 0, (
        "the late tool captured no kernel dispatches at all. Every phase-2 kernel ran on "
        "a queue that already existed when the tool registered, so this is the signature "
        "of those queues never being adopted into the queue controller"
    )
    assert (
        captured == phase2_count
    ), f"expected exactly {phase2_count} dispatches (phase 2 only), got {captured}"


def test_captured_dispatch_only(tool_b_data):
    """The late tool enabled KERNEL_DISPATCH_BUFFERED and nothing else."""
    sdk = _sdk(tool_b_data)

    hip = sdk["buffer_records"]["hip_api_traces"]
    assert (
        len(hip) == 0
    ), f"captured {len(hip)} HIP API records but only enabled KERNEL_DISPATCH_BUFFERED"


def test_dispatches_after_registration(tool_b_data):
    """Timestamp proof: every dispatch falls inside the tool's active window."""
    sdk = _sdk(tool_b_data)
    init_time = sdk["metadata"]["init_time"]
    fini_time = sdk["metadata"]["fini_time"]

    for itr in sdk["buffer_records"]["kernel_dispatch"]:
        assert itr["start_timestamp"] < itr["end_timestamp"], f"bad span: {itr}"
        assert (
            init_time < itr["start_timestamp"]
        ), f"dispatch started before the tool registered (init_time={init_time}): {itr}"
        assert (
            fini_time > itr["end_timestamp"]
        ), f"dispatch ended after finalize (fini_time={fini_time}): {itr}"


def test_dispatch_ids_unique(tool_b_data):
    sdk = _sdk(tool_b_data)
    ids = [
        itr["dispatch_info"]["dispatch_id"]
        for itr in sdk["buffer_records"]["kernel_dispatch"]
    ]
    assert len(ids) == len(set(ids)), "duplicate dispatch ids found"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
