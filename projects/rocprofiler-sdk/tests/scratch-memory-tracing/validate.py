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


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    data = input_data
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    node_exists("rocprofiler-sdk-json-tool", data)

    sdk_data = data["rocprofiler-sdk-json-tool"]

    num_agents = len([agent for agent in sdk_data["agents"] if agent["type"] == 2])

    node_exists("metadata", sdk_data)
    node_exists("pid", sdk_data["metadata"])
    node_exists("main_tid", sdk_data["metadata"])
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    node_exists("host_functions", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("hip_api_traces", sdk_data["callback_records"], 0)
    node_exists(
        "scratch_memory_traces", sdk_data["callback_records"], min_len=(2 * num_agents)
    )

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatch", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 0)
    node_exists("hsa_api_traces", sdk_data["buffer_records"])
    node_exists("hip_api_traces", sdk_data["buffer_records"], 0)
    node_exists("retired_correlation_ids", sdk_data["buffer_records"])
    node_exists(
        "scratch_memory_traces", sdk_data["buffer_records"], min_len=(1 * num_agents)
    )


def test_timestamps(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    cb_start = {}
    cb_end = {}
    for titr in ["hsa_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            cid = itr["correlation_id"]["internal"]
            phase = itr["phase"]
            if phase == 1:
                cb_start[cid] = itr["timestamp"]
            elif phase == 2:
                cb_end[cid] = itr["timestamp"]
                assert cb_start[cid] <= itr["timestamp"]
            else:
                assert phase == 1 or phase == 2

        for itr in sdk_data["buffer_records"][titr]:
            assert itr["start_timestamp"] <= itr["end_timestamp"]

    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["start_timestamp"] < itr["end_timestamp"], f"[{titr}] {itr}"
            assert itr["correlation_id"]["internal"] > 0, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert (
                sdk_data["metadata"]["init_time"] < itr["start_timestamp"]
            ), f"[{titr}] {itr}"
            assert (
                sdk_data["metadata"]["init_time"] < itr["end_timestamp"]
            ), f"[{titr}] {itr}"
            assert (
                sdk_data["metadata"]["fini_time"] > itr["start_timestamp"]
            ), f"[{titr}] {itr}"
            assert (
                sdk_data["metadata"]["fini_time"] > itr["end_timestamp"]
            ), f"[{titr}] {itr}"

            # TODO(Is this check applicable for scratch, which doesn't use any correlation id?)
            # api_start = cb_start[itr["correlation_id"]["internal"]]
            # api_end = cb_end[itr["correlation_id"]["internal"]]
            # assert api_start < itr["start_timestamp"]
            # assert api_end <= itr["end_timestamp"]


def test_internal_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in ["hsa_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

        for itr in sdk_data["buffer_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    for itr in sdk_data["buffer_records"]["memory_copies"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_external_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    extern_corr_ids = []
    for titr in ["hsa_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in ["hsa_api_traces", "hip_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert (
                itr["thread_id"] == itr["correlation_id"]["external"]
            ), f"[{titr}] {itr}"
            assert itr["thread_id"] in extern_corr_ids, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"

    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["names"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]
            else:
                for oidx, oname in enumerate(itr["operations"]):
                    if op_name == oname:
                        return oidx

    return None


# Tests above are identical to async-copy. Update as needed


def test_scratch_memory_tracking(input_data):
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    callback_records = sdk_data["callback_records"]
    buffer_records = sdk_data["buffer_records"]

    scratch_callback_data = callback_records["scratch_memory_traces"]
    scratch_buffer_data = buffer_records["scratch_memory_traces"]

    assert len(scratch_callback_data) == 2 * len(scratch_buffer_data)

    _, cb_op_names = get_operation(callback_records, "SCRATCH_MEMORY")
    _, bf_op_names = get_operation(buffer_records, "SCRATCH_MEMORY")

    assert len(cb_op_names) == 4
    assert len(bf_op_names) == 4

    # op name -> enum value
    assert cb_op_names == bf_op_names

    scratch_reported_agent_ids = set()
    detected_agents_ids = set(
        agent["id"]["handle"] for agent in sdk_data["agents"] if agent["type"] == 2
    )
    # check buffering data
    for node in scratch_buffer_data:
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

        assert node["start_timestamp"] > 0
        assert node["start_timestamp"] < node["end_timestamp"]

        scratch_reported_agent_ids.add(node["agent_id"]["handle"])

    assert 2**64 - 1 not in scratch_reported_agent_ids
    assert scratch_reported_agent_ids == detected_agents_ids

    # { thread-id -> [ events ],  ... }
    cb_threads = defaultdict(list)
    bf_threads = defaultdict(list)

    # fetch node["payload"]
    pl = lambda x: x["payload"]
    # fetch node
    rc = lambda x: x

    for node in scratch_callback_data:
        cb_threads[rc(node)["thread_id"]].append(node)

    for node in scratch_buffer_data:
        bf_threads[node["thread_id"]].append(node)

    for thread_id, nodes in cb_threads.items():
        assert thread_id > 0

        # sort based on timestamp
        nodes = sorted(nodes, key=lambda x: x["timestamp"])

        # start must be followed by end
        for inx in range(0, len(nodes), 2):
            this_node = nodes[inx]
            next_node = nodes[inx + 1]

            assert (
                rc(this_node)["phase"] + 1 == rc(next_node)["phase"]
            ), f"this:\n{this_node}\n\nnext:\n{next_node}"
            assert (
                rc(this_node)["thread_id"] == rc(next_node)["thread_id"]
            ), f"this:\n{this_node}\n\nnext:\n{next_node}"
            assert (
                this_node["timestamp"] < next_node["timestamp"]
            ), f"this:\n{this_node}\n\nnext:\n{next_node}"

            # alloc has more data vs free and async reclaim
            scratch_alloc_node = cb_op_names[this_node["operation"]]
            if scratch_alloc_node == "SCRATCH_MEMORY_ALLOC":
                assert (
                    pl(this_node)["queue_id"]["handle"]
                    == pl(next_node)["queue_id"]["handle"]
                )
                assert (
                    this_node["args"]["dispatch_id"] == next_node["args"]["dispatch_id"]
                )
                assert "size" in pl(next_node) and pl(next_node)["size"] > 0
                assert (
                    "num_slots" in next_node["args"]
                    and next_node["args"]["num_slots"] > 0
                )
                assert "flags" in pl(next_node)

    # callback data and buffer data must agree with each other
    for bf_thr, bf_nodes in bf_threads.items():
        cb_nodes = cb_threads[bf_thr]

        for bf_node_inx in range(len(bf_nodes)):
            # All these 3 should have same data
            # timestamps are not same as callback records them at
            # a different instant in time. Callback timestamp
            # should be more than buffer timestamp
            bf_node = bf_nodes[bf_node_inx]
            cb_enter = cb_nodes[bf_node_inx * 2]
            cb_exit = cb_nodes[bf_node_inx * 2 + 1]

            assert (
                bf_node["operation"]
                == rc(cb_enter)["operation"]
                == rc(cb_exit)["operation"]
            )
            assert (
                bf_op_names[bf_node["operation"]]
                == cb_op_names[rc(cb_enter)["operation"]]
                == cb_op_names[rc(cb_exit)["operation"]]
            )

            assert bf_node["flags"] == pl(cb_exit)["flags"]

            assert (
                bf_node["thread_id"]
                == rc(cb_enter)["thread_id"]
                == rc(cb_exit)["thread_id"]
            )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
