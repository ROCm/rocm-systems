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


def test_memory_allocation(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    memory_allocation_data = buffer_records["memory_allocation"]

    _, bf_op_names = get_operation(data, "MEMORY_ALLOCATION")

    assert len(bf_op_names) == 5

    allocation_reported_agent_ids = set()
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
        assert node.agent_id.handle > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp

        assert data.strings.buffer_records[node.kind].kind == "MEMORY_ALLOCATION"
        assert (
            data.strings.buffer_records[node.kind].operations[node.operation]
            in bf_op_names
        )

        allocation_reported_agent_ids.add(node["agent_id"]["handle"])


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(otf2_data, json_data, ("memory_allocation",))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
