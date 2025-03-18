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
import re


def get_type(id_type):
    if id_type == "absolute":
        return "node_id"
    elif id_type == "type-relative":
        return "logical_node_type_id"
    else:
        return "logical_node_id"


def check_kernel_traces(json_data, kernels_data, id_type):
    data = json_data["rocprofiler-sdk-tool"]
    agents = data["agents"]
    kernels = data["buffer_records"]["kernel_dispatch"]
    _type = get_type(id_type)

    for row in kernels_data:
        found = any(
            int(agent[_type]) == int(row["Agent_Id"].split(" ")[-1]) for agent in agents
        )

        assert found, f"Agent ID {row['Agent_Id']} of Type {id_type} not found"


def check_memory_allocation_traces(json_data, mem_alloc, id_type):
    data = json_data["rocprofiler-sdk-tool"]
    agents = data["agents"]
    mem_alloc_record = data["buffer_records"]["memory_allocation"]
    _type = get_type(id_type)

    for row in mem_alloc:

        found = row["Agent_Id"] != "" and any(
            int(agent[_type]) == int(row["Agent_Id"].split(" ")[-1]) for agent in agents
        )

        if row["Agent_Id"] != "":
            assert found, f"Agent ID {row['Agent_Id']} of Type {id_type} not found"


def check_memory_copy_traces(json_data, mem_copy, id_type):
    data = json_data["rocprofiler-sdk-tool"]
    agents = data["agents"]
    mem_copy_record = data["buffer_records"]["memory_copy"]
    _type = get_type(id_type)

    for row in mem_copy:

        found = any(
            int(agent[_type]) == int(row["Source_Agent_Id"].split(" ")[-1])
            for agent in agents
        )

        assert (
            found
        ), f"Src Agent ID {row['Destination_Agent_Id']} of Type {id_type} not found"

        found = any(
            int(agent[_type]) == int(row["Destination_Agent_Id"].split(" ")[-1])
            for agent in agents
        )

        assert (
            found
        ), f"Destination Agent ID {row['Destiantion_Agent_Id']} of Type {id_type} not found"

        if (
            _type == "logical_node_type_id"
            and row["Direction"] == "MEMORY_COPY_HOST_TO_DEVICE"
        ):
            assert row["Source_Agent_Id"].split(" ")[0] == "CPU"
            assert row["Destination_Agent_Id"].split(" ")[0] == "GPU"

        elif (
            _type == "logical_node_type_id"
            and row["Direction"] == "MEMORY_COPY_DEVICE_TO_HOST"
        ):
            assert row["Source_Agent_Id"].split(" ")[0] == "GPU"
            assert row["Destination_Agent_Id"].split(" ")[0] == "CPU"


def test_validate(
    csv_kernel_input,
    json_data,
    agent_index,
    csv_memory_allocation_input,
    csv_memory_copy_input,
):
    check_kernel_traces(json_data, csv_kernel_input, agent_index)
    check_memory_copy_traces(json_data, csv_memory_copy_input, agent_index)
    check_memory_allocation_traces(json_data, csv_memory_allocation_input, agent_index)


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
