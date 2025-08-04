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
import json

from collections import defaultdict


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["strings"]["buffer_records"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]
            else:
                for oidx, oname in enumerate(itr["operations"]):
                    if op_name == oname:
                        return oidx
    return None


def test_memory_allocation(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    memory_allocation_data = buffer_records["memory_allocation"]

    _, bf_op_names = get_operation(data, "MEMORY_ALLOCATION")

    assert len(bf_op_names) == 5

    # Op values:
    UNKNOWN_OP = 0
    HSA_MEMORY_ALLOCATE_OP = 1
    HSA_AMD_VMEM_HANDLE_CREATE_OP = 2
    HSA_MEMORY_FREE_OP = 3
    HSA_AMD_VMEM_HANDLE_RELEASE = 4
    TOTAL_MEM_OPS = 5

    ALLOCATE_OPS = (HSA_MEMORY_ALLOCATE_OP, HSA_AMD_VMEM_HANDLE_CREATE_OP)
    FREE_OPS = (HSA_MEMORY_FREE_OP, HSA_AMD_VMEM_HANDLE_RELEASE)

    valid_agent_ids = set()
    for row in data["agents"]:
        if "id" in row and "handle" in row.id:
            valid_agent_ids.add(row.id.handle)

    memory_alloc_cnt = dict(
        [
            (idx, {"agent": set(), "starting_addr": set(), "size": set(), "count": 0})
            for idx in range(1, TOTAL_MEM_OPS)
        ]
    )

    # check buffering data
    for node in memory_allocation_data:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node
        assert "thread_id" in node

        assert "agent_id" in node
        assert "address" in node
        assert "allocation_size" in node

        assert node.size > 0
        assert len(node.address) > 0
        assert node.thread_id > 0

        # Check that op ID is valid
        op_id = node.operation
        assert op_id > UNKNOWN_OP and op_id < TOTAL_MEM_OPS
        # Summarize info in dict
        memory_alloc_cnt[op_id]["count"] += 1
        memory_alloc_cnt[op_id]["starting_addr"].add(node.address)
        memory_alloc_cnt[op_id]["size"].add(node.allocation_size)
        memory_alloc_cnt[op_id]["agent"].add(node.agent_id.handle)

        # Check if agent is valid
        if op_id in ALLOCATE_OPS:
            assert node.agent_id.handle in valid_agent_ids
        else:
            assert node.agent_id.handle == 0  # free ops record agent id as null

        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp

        assert data.strings.buffer_records[node.kind].kind == "MEMORY_ALLOCATION"
        assert (
            data.strings.buffer_records[node.kind].operations[node.operation]
            in bf_op_names
        )

    # In the memory allocation test which generates this file
    # 6 hsa_memory_allocation calls with 1024 bytes were called
    # and 9 hsa_amd_memory_pool_allocations with 2048 bytes
    # were called
    assert memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["count"] == 15
    assert memory_alloc_cnt[HSA_MEMORY_FREE_OP]["count"] == 15

    # Check that allocation operations have corresponding free operations
    assert len(memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["starting_addr"]) == len(
        memory_alloc_cnt[HSA_MEMORY_FREE_OP]["starting_addr"]
    )
    for starting_addr in memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["starting_addr"]:
        assert starting_addr in memory_alloc_cnt[HSA_MEMORY_FREE_OP]["starting_addr"]

    # Confirm validation sizes are valid
    assert len(memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["size"]) == 2
    assert (
        len(memory_alloc_cnt[HSA_MEMORY_FREE_OP]["size"]) == 1
    )  # size for free ops are 0
    assert 1024 in memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["size"]
    assert 2048 in memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["size"]

    # Confirm that two agents were used
    assert len(memory_alloc_cnt[HSA_MEMORY_ALLOCATE_OP]["agent"]) == 2


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(otf2_data, json_data, ("memory_allocation",))


def test_csv_data(csv_data):
    assert len(csv_data) > 0, "Expected non-empty csv data"

    ALLOCATION_OPS = (
        "MEMORY_ALLOCATION_ALLOCATE",
        "ROCPROFILER_MEMORY_ALLOCATION_VMEM_ALLOCATE",
    )
    FREE_OPS = (
        "ROCPROFILER_MEMORY_ALLOCATION_FREE",
        "ROCPROFILER_MEMORY_ALLOCATION_VMEM_FREE",
    )

    memory_allocation_info = dict(
        {
            "agents": set(),
            "allocation_size": defaultdict(int),
            "address": defaultdict(lambda: defaultdict(int)),
        }
    )

    for row in csv_data:
        assert (
            "Kind" in row
        ), "'Kind' was not present in csv data for memory-allocation-trace"
        assert (
            "Operation" in row
        ), "'Operation' was not present in csv data for memory-allocation-trace"
        assert (
            "Agent_Id" in row
        ), "'Agent_Id' was not present in csv data for memory-allocation-trace"
        assert (
            "Allocation_Size" in row
        ), "'Allocation_Size' was not present in csv data for memory-allocation-trace"
        assert (
            "Address" in row
        ), "'Address' was not present in csv data for memory-allocation-trace"
        assert (
            "Correlation_Id" in row
        ), "'Correlation_Id' was not present in csv data for memory-allocation-trace"
        assert (
            "Start_Timestamp" in row
        ), "'Start_Timestamp' was not present in csv data for memory-allocation-trace"
        assert (
            "End_Timestamp" in row
        ), "'End_Timestamp' was not present in csv data for memory-allocation-trace"

        assert row["Kind"] == "MEMORY_ALLOCATION"
        assert row["Operation"] in (
            "MEMORY_ALLOCATION_ALLOCATE",
            "MEMORY_ALLOCATION_FREE",
        )
        assert int(row["Correlation_Id"]) > 0

        # Check if agent ID is assigned to correct row
        if row["Operation"] in ALLOCATION_OPS:
            assert row["Agent_Id"].split(" ")[0] == "Agent"
            memory_allocation_info["agents"].add(row["Agent_Id"])
        else:
            assert row["Agent_Id"] == ""  # free ops have no agent

        # Confirm allocation size is valid
        if row["Operation"] in ALLOCATION_OPS:
            assert int(row["Allocation_Size"]) in (
                1024,
                2048,
            )  # Test allocates 1024 and 2048 bytes only
            memory_allocation_info["allocation_size"][int(row["Allocation_Size"])] += 1
        else:
            row["Allocation_Size"] == 0  # Free ops record allocation size as 0

        # Confirm address is valid
        assert row["Address"][:2] == "0x"
        address = int(row["Address"], 16)
        if row["Operation"] in ALLOCATION_OPS:
            memory_allocation_info["address"][address]["allocation"] += 1
        else:
            memory_allocation_info["address"][address]["free"] += 1
        # Timestamp sanity check
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) < int(row["End_Timestamp"])
    # Test uses 2 agents with 6 allocations of 1024 bytes and 9 allocations of 2048 bytes
    assert len(memory_allocation_info["agents"]) == 2
    assert memory_allocation_info["allocation_size"][1024] == 6
    assert memory_allocation_info["allocation_size"][2048] == 9
    for address in memory_allocation_info["address"].values():
        assert (
            address["allocation"] == address["free"]
        )  # Free should have corresponding allocate


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
