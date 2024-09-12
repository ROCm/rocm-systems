#!/usr/bin/env python3

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


def test_scratch_memory(input_data):
    data = input_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    scratch_memory_data = buffer_records["scratch_memory"]

    _, bf_op_names = get_operation(data, "SCRATCH_MEMORY")

    assert len(bf_op_names) == 4

    scratch_reported_agent_ids = set()
    detected_agents_ids = set(
        agent["id"]["handle"] for agent in data["agents"] if agent["type"] == 2
    )
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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
