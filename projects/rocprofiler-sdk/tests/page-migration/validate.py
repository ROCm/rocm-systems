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

from collections import defaultdict
import sys
import pytest


test_api_traces = [
    "hsa_api_traces",
    "marker_api_traces",
    "hip_api_traces",
    "rccl_api_traces",
    "scratch_memory_traces",
]


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def to_dict(key_values):
    a = defaultdict()
    for kv in key_values:
        a[kv["key"]] = kv["value"]
    return a


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


def dict_from_value_key(d):
    ret_d = defaultdict()

    for k, v in d.items():
        assert v not in ret_d
        ret_d[v] = k
    return ret_d


def sort_by_timestamp(lines):
    timestamp_line_map = {}

    for log_line in lines:
        timestamp = log_line.split(" ")[1]
        timestamp_line_map[timestamp] = log_line

    timestamps_sorted = sorted([l.split(" ")[1] for l in lines])
    return timestamps_sorted, timestamp_line_map


# ------------------------------ Tests ------------------------------ #


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    data = input_data

    node_exists("rocprofiler-sdk-json-tool", data)

    sdk_data = data["rocprofiler-sdk-json-tool"]

    node_exists("metadata", sdk_data)
    node_exists("pid", sdk_data["metadata"])
    node_exists("main_tid", sdk_data["metadata"])
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])
    node_exists("validate_page_migration", sdk_data["metadata"])

    assert sdk_data["metadata"]["validate_page_migration"] is True

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    node_exists("host_functions", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"], 0)
    node_exists("hip_api_traces", sdk_data["callback_records"])
    node_exists("marker_api_traces", sdk_data["callback_records"], 0)

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatch", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 0)
    node_exists("hsa_api_traces", sdk_data["buffer_records"], 0)
    node_exists("hip_api_traces", sdk_data["buffer_records"])
    node_exists("marker_api_traces", sdk_data["buffer_records"], 0)
    node_exists("retired_correlation_ids", sdk_data["buffer_records"])
    node_exists("page_migration", sdk_data["buffer_records"])


def test_timestamps(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    cb_start = {}
    cb_end = {}
    for titr in test_api_traces:
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
            assert itr["start_timestamp"] < itr["end_timestamp"]
            assert itr["correlation_id"]["internal"] > 0
            assert itr["correlation_id"]["external"] > 0
            assert sdk_data["metadata"]["init_time"] < itr["start_timestamp"]
            assert sdk_data["metadata"]["init_time"] < itr["end_timestamp"]
            assert sdk_data["metadata"]["fini_time"] > itr["start_timestamp"]
            assert sdk_data["metadata"]["fini_time"] > itr["end_timestamp"]

            # api_start = cb_start[itr["correlation_id"]["internal"]]
            # api_end = cb_end[itr["correlation_id"]["internal"]]
            # assert api_start < itr["start_timestamp"]
            # assert api_end <= itr["end_timestamp"]


def test_internal_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in test_api_traces:
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
    for titr in test_api_traces:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in test_api_traces:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            assert itr["thread_id"] in extern_corr_ids
            assert itr["correlation_id"]["external"] in extern_corr_ids

    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        assert itr["correlation_id"]["external"] > 0
        assert itr["correlation_id"]["external"] in extern_corr_ids

    for itr in sdk_data["buffer_records"]["memory_copies"]:
        assert itr["correlation_id"]["external"] > 0
        assert itr["correlation_id"]["external"] in extern_corr_ids


def test_kernel_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    symbol_info = {}
    for itr in sdk_data["callback_records"]["kernel_symbols"]:
        phase = itr["phase"]
        payload = itr["payload"]
        kern_id = payload["kernel_id"]

        assert phase == 1 or phase == 2
        assert kern_id > 0
        if phase == 1:
            assert len(payload["kernel_name"]) > 0
            symbol_info[kern_id] = payload
        elif phase == 2:
            assert payload["kernel_id"] in symbol_info.keys()
            assert payload["kernel_name"] == symbol_info[kern_id]["kernel_name"]

    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        assert itr["dispatch_info"]["kernel_id"] in symbol_info.keys()

    for itr in sdk_data["callback_records"]["kernel_dispatch"]:
        assert itr["payload"]["dispatch_info"]["kernel_id"] in symbol_info.keys()


def get_allocated_pages(callback_records):
    # Get how many pages we allocated
    op_idx = get_operation(callback_records, "HIP_RUNTIME_API", "hipHostRegister")
    rt_idx, rt_data = get_operation(callback_records, "HIP_RUNTIME_API")

    assert op_idx is not None, f"{rt_idx}:\n{rt_data}"

    host_register_record = []
    for itr in callback_records["hip_api_traces"]:
        if itr["kind"] == rt_idx and itr["operation"] == op_idx and itr["phase"] == 2:
            assert "sizeBytes" in itr["args"].keys(), f"{itr}"
            assert "hostPtr" in itr["args"].keys(), f"{itr}"
            host_register_record.append(itr)

    num_host_register_calls = len(host_register_record)
    assert num_host_register_calls == 5, "Expected 5 hipHostRegister calls in test"

    ret = []
    for i in range(num_host_register_calls):
        alloc_size = int(host_register_record[0]["args"]["sizeBytes"], 10)
        start_addr = int(host_register_record[0]["args"]["hostPtr"], 16)
        end_addr = start_addr + alloc_size
        ret.append((start_addr, end_addr))
    return ret


def validate_node(id, nodes):
    assert id.handle in nodes


def test_page_migration_data(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]
    buffer_records = sdk_data.buffer_records
    callback_records = sdk_data.callback_records
    page_migtation_buffers = buffer_records.page_migration

    _, bf_op_names = get_operation(buffer_records, "PAGE_MIGRATION")
    assert bf_op_names[0] == "PAGE_MIGRATION_NONE"

    for op_name in bf_op_names:
        assert "PAGE_MIGRATION" in op_name

    assert len(bf_op_names) == 9

    nodes = set(x.id.handle for x in sdk_data.agents)
    allocations = get_allocated_pages(callback_records)

    for start_addr, end_addr in allocations:

        assert (
            start_addr < end_addr
        ), "Expected start address less than end address for mmap range"
        alloc_size = end_addr - start_addr
        assert int(alloc_size) == 512 * 4096  # We allocated 512 pages in the test

        # PID must be same
        assert len(set(r.pid for r in page_migtation_buffers)) == 1

        for r in page_migtation_buffers:
            op = r.operation

            assert r.size == 160
            assert op != 0 and bf_op_names[op] != "PAGE_MIGRATION_NONE"
            assert bf_op_names[op].lower().replace("page_migration_", "") in r.keys()

            if "page_fault_start" in r:
                arg = r.page_fault_start
                assert arg.read_fault < 2
                validate_node(arg.agent_id, nodes)
                assert arg.address > 0

            if "page_fault_end" in r:
                arg = r.page_fault_end
                assert arg.migrated < 2
                validate_node(arg.agent_id, nodes)
                assert arg.address > 0

            if "page_migrate_start" in r:
                arg = r.page_migrate_start
                assert (
                    0 < arg.start_addr < arg.end_addr
                ), "Expected start addr to be less than end addr"
                if arg.start_addr == start_addr:
                    assert arg.end_addr == end_addr
                validate_node(arg.from_agent, nodes)
                validate_node(arg.to_agent, nodes)
                validate_node(arg.prefetch_agent, nodes)
                validate_node(arg.preferred_agent, nodes)
                assert 0 <= arg.trigger < 4

            if "page_migrate_end" in r:
                arg = r.page_migrate_end
                assert (
                    0 < arg.start_addr < arg.end_addr
                ), "Expected start addr to be less than end addr"
                if arg.start_addr == start_addr:
                    assert arg.end_addr == end_addr
                validate_node(arg.from_agent, nodes)
                validate_node(arg.to_agent, nodes)
                assert 0 <= arg.trigger < 4

            if "queue_eviction" in r:
                arg = r.queue_eviction
                validate_node(arg.agent_id, nodes)
                assert 0 <= arg.trigger < 6

            if "queue_restore" in r:
                arg = r.queue_restore
                assert arg.rescheduled < 2
                validate_node(arg.agent_id, nodes)

            if "unmap_from_gpu" in r:
                arg = r.unmap_from_gpu
                assert (
                    0 < arg.start_addr < arg.end_addr
                ), "Expected start addr to be less than end addr"
                if arg.start_addr == start_addr:
                    assert arg.end_addr == end_addr
                validate_node(arg.agent_id, nodes)
                assert 0 <= arg.trigger < 3

            if "dropped_event" in r:
                arg = r.dropped_event
                # We shouldn't get any dropped events. If we do, our test needs to be redesigned.
                assert arg.dropped_events_count == 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
