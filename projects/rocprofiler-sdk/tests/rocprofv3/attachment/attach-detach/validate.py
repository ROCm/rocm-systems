#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
import pytest


def test_agent_info(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    gpu_count = 0
    for agent in data["agents"]:
        assert agent["type"] in (1, 2)
        if agent["type"] == 2:
            gpu_count += 1

    assert gpu_count > 0, "No GPU agents found"


def test_kernel_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    assert (
        len(kernel_dispatch_data) > 0
    ), "No kernel dispatches captured before the interactive detach"

    for dispatch in kernel_dispatch_data:
        assert get_kind_name(dispatch["kind"]) == "KERNEL_DISPATCH"
        assert dispatch["correlation_id"]["internal"] > 0
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]
        assert dispatch["dispatch_info"]["queue_id"]["handle"] > 0


def test_hsa_api_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    hsa_api_data = data["buffer_records"]["hsa_api"]
    assert len(hsa_api_data) > 0, "No HSA API records captured during attachment"

    for api in hsa_api_data:
        assert get_kind_name(api["kind"]).startswith("HSA_")
        assert api["end_timestamp"] >= api["start_timestamp"]


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
