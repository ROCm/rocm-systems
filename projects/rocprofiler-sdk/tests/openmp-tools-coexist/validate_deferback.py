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

# Test-plan box: "User-tool defer-back (return rocprofiler_ompt_start_tool(...))
# arms SDK OMPT."
#
# The mock OMPT tool is preloaded FIRST (so the OpenMP runtime binds the mock's
# ompt_start_tool) and runs in mode=defer: it returns
# rocprofiler_ompt_start_tool(), handing the OMPT tool role back to the SDK. The
# coexist client requested OMPT (COEXIST_OMPT=1), so the SDK must arm OMPT and
# the client must collect OMPT records. The mock, having deferred, was never
# made the tool (initialize() never called).

import sys
import pytest


def test_defer_back_armed_sdk_ompt(client_data):
    c = client_data["coexist-client"]
    assert c.ompt_requested is True, "client did not request OMPT in the defer-back case"
    assert c.ompt_records > 0, (
        "defer-back failed: the mock handed the OMPT role back to the SDK but the SDK "
        "collected no OMPT records (OMPT was not armed)"
    )


def test_mock_deferred_not_initialized(mock_data):
    m = mock_data["coexist-mock-tool"]
    assert m.initialized is False, (
        "the mock returned the SDK's start-tool result (defer) but was itself initialized "
        "as the OMPT tool"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
