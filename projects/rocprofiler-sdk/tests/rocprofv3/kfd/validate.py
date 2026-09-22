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


def test_kfd_trace(json_data):
    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def assert_fields(record, fields):
        for field in fields:
            assert field in record

    def assert_agent(agent_ids, agent):
        assert agent["handle"] in agent_ids

    def assert_addresses(record):
        assert record["end_address"]["handle"] >= record["start_address"]["handle"]

    data = json_data["rocprofiler-sdk-tool"]

    # get list of valid agent IDs
    agent_ids = [agent["id"]["handle"] for agent in data["agents"]]
    # unknown agent is possible
    agent_ids.append(0)

    valid_kind = (
        "KFD_EVENT_PAGE_MIGRATE",
        "KFD_EVENT_PAGE_FAULT",
        "KFD_EVENT_QUEUE",
        "KFD_EVENT_UNMAP_FROM_GPU",
        "KFD_EVENT_DROPPED_EVENTS",
        "KFD_PAGE_MIGRATE",
        "KFD_PAGE_FAULT",
        "KFD_QUEUE",
    )
    paired_kind = (
        "KFD_PAGE_MIGRATE",
        "KFD_PAGE_FAULT",
        "KFD_QUEUE",
    )

    buffer_records = data["buffer_records"]
    kfd_data = buffer_records["kfd"]
    for record in kfd_data:
        # assert common fields across all KFD types
        assert_fields(record, ["size", "kind", "operation", "pid"])

        # assert invariants in common fields
        kind_id = record["kind"]
        kind = get_kind_name(kind_id)
        assert kind in valid_kind

        op_id = record["operation"]
        assert op_id >= 0 and op_id < len(
            data["strings"]["buffer_records"][kind_id]["operations"]
        )

        if kind in paired_kind:
            assert "start_timestamp" in record
            assert "end_timestamp" in record
            assert record["start_timestamp"] > 0
            assert record["end_timestamp"] >= record["start_timestamp"]
        else:
            assert "timestamp" in record
            assert record["timestamp"] > 0

        # per kind assertions
        if kind == "KFD_EVENT_PAGE_MIGRATE" or kind == "KFD_PAGE_MIGRATE":
            assert_fields(
                record,
                [
                    "start_address",
                    "end_address",
                    "src_agent",
                    "dst_agent",
                    "prefetch_agent",
                    "preferred_agent",
                    "error_code",
                ],
            )
            assert_agent(agent_ids, record["src_agent"])
            assert_agent(agent_ids, record["dst_agent"])
            assert_agent(agent_ids, record["prefetch_agent"])
            assert_agent(agent_ids, record["preferred_agent"])
            assert_addresses(record)

        elif kind == "KFD_EVENT_PAGE_FAULT" or kind == "KFD_PAGE_FAULT":
            assert_fields(record, ["agent_id", "address"])
            assert_agent(agent_ids, record["agent_id"])

        elif kind == "KFD_EVENT_QUEUE" or kind == "KFD_QUEUE":
            assert_fields(record, ["agent_id"])
            assert_agent(agent_ids, record["agent_id"])

        elif kind == "KFD_EVENT_UNMAP_FROM_GPU":
            assert_fields(record, ["agent_id", "start_address", "end_address"])
            assert_agent(agent_ids, record["agent_id"])
            assert_addresses(record)

        elif kind == "KFD_EVENT_DROPPED_EVENTS":
            assert_fields(record, ["count"])

        else:
            # unreachable
            assert 1 == 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
