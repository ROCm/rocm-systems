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

    valid_agent_ids = set()
    for row in data["agents"]:
        if "id" in row and "handle" in row.id:
            valid_agent_ids.add(row.id.handle)

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
        assert node.allocation_size >= 0
        assert len(node.address) > 0
        assert node.thread_id > 0

        op_id = node.operation
        assert op_id != UNKNOWN_OP
        if op_id == HSA_MEMORY_ALLOCATE_OP or op_id == HSA_AMD_VMEM_HANDLE_CREATE_OP:
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


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(otf2_data, json_data, ("memory_allocation",))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
