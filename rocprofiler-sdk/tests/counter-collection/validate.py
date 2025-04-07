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

import sys
import pytest


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)
    rocp_data = input_data
    node_exists("names", rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"])
    node_exists(
        "counter_collection", rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"]
    )


def test_counter_values(input_data):
    data = input_data["rocprofiler-sdk-json-tool"]
    agent_data = data["agents"]
    counter_info = data["counter_info"]
    counter_data = data["buffer_records"]["counter_collection"]

    for itr in counter_info:
        if itr["is_constant"] == 1:
            continue
        assert itr["id"]["handle"] > 0, f"{itr}"
        assert itr["is_constant"] in (0, 1), f"{itr}"
        assert itr["is_derived"] in (0, 1), f"{itr}"
        assert len(itr["name"]) >= 4, f"{itr}"
        assert len(itr["description"]) >= 4, f"{itr}"
        if itr["is_constant"] == 0:
            if itr["is_derived"] == 0:
                assert len(itr["block"]) > 0, f"{itr}"
            if itr["is_derived"] == 1:
                assert len(itr["expression"]) > 0, f"{itr}"

    def get_agent(agent_id):
        for itr in agent_data:
            if itr["id"]["handle"] == agent_id["handle"]:
                return itr
        return None

    def get_scaling_factor(agent_id):
        agent = get_agent(agent_id)
        assert agent is not None, f"id={agent_id}"
        if agent["type"] == 2 and agent["wave_front_size"] > 0:
            return 64 / agent["wave_front_size"]
        return 0

    for itr in counter_data:
        assert itr["num_records"] == len(itr["records"]), f"itr={itr}"
        agent_id = itr["dispatch_info"]["agent_id"]
        agent = get_agent(agent_id)
        scaling_factor = get_scaling_factor(agent_id)
        assert agent is not None, f"itr={itr}\nagent={agent}"
        assert itr["start_timestamp"] < itr["end_timestamp"]
        for ritr in itr["records"]:
            value = ritr["counter_value"]
            if int(round(value, 0)) > 0:
                assert int(round(value, 0)) == int(
                    round(1 * scaling_factor, 0)
                ), f"itr={itr}\nagent={agent}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
