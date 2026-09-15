#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# Test-plan box: "User OMPT tool runs while the SDK does non-OMPT tracing (no
# client OMPT service): SDK defers, user tool active."
#
# The coexist client did NOT request OMPT (COEXIST_OMPT unset), so the SDK
# defers the OMPT tool role; the mock OMPT tool (mode=own, via
# OMP_TOOL_LIBRARIES) becomes the runtime's OMPT tool. The SDK must still do its
# own non-OMPT (kernel-dispatch) tracing.

import sys
import pytest


def test_mock_became_the_tool(mock_data):
    """The SDK deferred, so the user/mock OMPT tool was made the OMPT tool."""
    m = mock_data["coexist-mock-tool"]
    assert (
        m.initialized is True
    ), "SDK did not defer: the mock OMPT tool was never initialized"


def test_mock_received_events(mock_data):
    m = mock_data["coexist-mock-tool"]
    ev = m.events
    assert ev.thread_begin >= 1, "mock OMPT tool received no thread_begin"
    assert ev.device_initialize >= 1, "mock OMPT tool received no device_initialize"
    assert (
        ev.target_submit_emi >= 1
    ), "mock OMPT tool received no target_submit_emi (no offload?)"


def test_sdk_deferred_but_still_traced(client_data):
    """The SDK kept doing its non-OMPT work and consumed no OMPT itself."""
    c = client_data["coexist-client"]
    assert (
        c.ompt_requested is False
    ), "client unexpectedly requested OMPT in the defer case"
    assert (
        c.kernel_dispatch_records > 0
    ), "SDK collected no kernel-dispatch records while deferring"
    assert c.ompt_records == 0, (
        "SDK deferred the OMPT role but the client still received OMPT records "
        f"({c.ompt_records}); the SDK should not be the OMPT tool here"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
