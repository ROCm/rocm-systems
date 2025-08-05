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


def test_scratch_memory(json_input_data, csv_input_data):
    data = json_input_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    scratch_memory_data = buffer_records["scratch_memory"]

    _, bf_op_names = get_operation(data, "SCRATCH_MEMORY")

    assert len(bf_op_names) == 4

    scratch_reported_agent_ids = set()
    detected_agents_ids = set(
        agent["id"]["handle"] for agent in data["agents"] if agent["type"] == 2
    )

    assert len(scratch_memory_data) > 0

    # check buffering data
    for node in scratch_memory_data:
        assert "size" in node
        assert "kind" in node
        assert "flags" in node
        assert "thread_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node

        assert "queue_id" in node
        assert "agent_id" in node
        assert "operation" in node
        assert "handle" in node["queue_id"]

        assert node.size > 0
        assert node.thread_id > 0
        assert node.agent_id.handle > 0
        assert node.queue_id.handle > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp

        assert data.strings.buffer_records[node.kind].kind == "SCRATCH_MEMORY"
        assert (
            data.strings.buffer_records[node.kind].operations[node.operation]
            in bf_op_names
        )

        scratch_reported_agent_ids.add(node["agent_id"]["handle"])

    assert 2**64 - 1 not in scratch_reported_agent_ids
    assert scratch_reported_agent_ids == detected_agents_ids

    assert len(csv_input_data) > 0

    for row in csv_input_data:
        assert "Kind" in row, "Kind header not present in csv for scratch memory trace."
        assert (
            "Operation" in row
        ), "Operation header not present in csv for scratch memory trace."
        assert (
            "Agent_Id" in row
        ), "Agent_Id header not present in csv for scratch memory trace."
        assert (
            "Queue_Id" in row
        ), "Queue_Id header not present in csv for scratch memory trace."
        assert (
            "Thread_Id" in row
        ), "Thread_Id header not present in csv for scratch memory trace."
        assert (
            "Alloc_Flags" in row
        ), "Alloc_Flags header not present in csv for scratch memory trace."
        assert (
            "Start_Timestamp" in row
        ), "Start_Timestamp header not present in csv for scratch memory trace."
        assert (
            "End_Timestamp" in row
        ), "End_Timestamp header not present in csv for scratch memory trace."

        assert row["Kind"] == "SCRATCH_MEMORY"
        assert row["Operation"] in ["SCRATCH_MEMORY_ALLOC", "SCRATCH_MEMORY_FREE"]
        assert int(row["Agent_Id"].split(" ")[-1]) >= 0
        assert int(row["Queue_Id"]) > 0
        assert int(row["Thread_Id"]) > 0
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) < int(row["End_Timestamp"])


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
