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
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len, f"{name}:\n{data}"


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

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("memory_allocations", sdk_data["callback_records"])

    node_exists("names", sdk_data["buffer_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("memory_allocations", sdk_data["buffer_records"])


def test_size_entries(input_data):
    # check that size fields are > 0 but account for function arguments
    # which are named "size"
    def check_size(data, bt):
        if "size" in data.keys():
            if isinstance(data["size"], str) and bt.endswith('["args"]'):
                pass
            else:
                assert data["size"] > 0, f"origin: {bt}"

    # recursively check the entire data structure
    def iterate_data(data, bt):
        if isinstance(data, (list, tuple)):
            for i, itr in enumerate(data):
                if isinstance(itr, dict):
                    check_size(itr, f"{bt}[{i}]")
                iterate_data(itr, f"{bt}[{i}]")
        elif isinstance(data, dict):
            check_size(data, f"{bt}")
            for key, itr in data.items():
                iterate_data(itr, f'{bt}["{key}"]')

    # start recursive check over entire JSON dict
    iterate_data(input_data, "input_data")


def test_timestamps(input_data):
    """Verify starting timestamps are less than ending timestamps"""
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    cb_start = {}
    cb_end = {}
    for titr in ["hsa_api_traces"]:
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

    for titr in ["memory_allocations"]:
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

            api_start = cb_start[itr["correlation_id"]["internal"]]
            # api_end = cb_end[itr["correlation_id"]["internal"]]
            assert api_start < itr["start_timestamp"], f"[{titr}] {itr}"
            # assert api_end <= itr["end_timestamp"], f"[{titr}] {itr}"


def test_internal_correlation_ids(input_data):
    """Assure correlation ids are unique"""
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in ["hsa_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

        for itr in sdk_data["buffer_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    for itr in sdk_data["buffer_records"]["memory_allocations"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_external_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    extern_corr_ids = []
    for titr in ["hsa_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in ["hsa_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert (
                itr["thread_id"] == itr["correlation_id"]["external"]
            ), f"[{titr}] {itr}"
            assert itr["thread_id"] in extern_corr_ids, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"

    for titr in ["memory_allocations"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"

        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"


def test_memory_alloc_sizes(input_data):
    """Ensure trace file memory allocation operations match up with the memory allocation operations performed in hsa-memory-allocation"""
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    # Op values:
    UNKNOWN_OP = 0
    HSA_MEMORY_ALLOCATE_OP = 1
    HSA_AMD_VMEM_HANDLE_CREATE_OP = 2
    HSA_MEMORY_FREE_OP = 3
    HSA_AMD_VMEM_HANDLE_RELEASE = 4
    TOTAL_MEM_OPS = 5

    ALLOCATE_OPS = (HSA_MEMORY_ALLOCATE_OP, HSA_AMD_VMEM_HANDLE_CREATE_OP)
    FREE_OPS = (HSA_MEMORY_FREE_OP, HSA_AMD_VMEM_HANDLE_RELEASE)

    memory_alloc_cnt = dict(
        [
            (idx, {"agent": set(), "starting_addr": set(), "size": set(), "count": 0})
            for idx in range(1, 5)
        ]
    )
    for itr in sdk_data["buffer_records"]["memory_allocations"]:
        op_id = itr["operation"]
        assert op_id > UNKNOWN_OP and op_id < TOTAL_MEM_OPS, f"{itr}"
        memory_alloc_cnt[op_id]["count"] += 1
        memory_alloc_cnt[op_id]["starting_addr"].add(itr.address)
        memory_alloc_cnt[op_id]["size"].add(itr.allocation_size)
        memory_alloc_cnt[op_id]["agent"].add(itr.agent_id.handle)

    for itr in sdk_data["callback_records"]["memory_copies"]:
        op_id = itr.operation
        assert op_id > UNKNOWN_OP and op_id < TOTAL_MEM_OPS, f"{itr}"
        memory_alloc_cnt[op_id]["count"] += 1

        phase = itr.phase
        pitr = itr.payload

        assert phase is not None, f"{itr}"
        assert pitr is not None, f"{itr}"

        if phase == 1:
            assert pitr.start_timestamp == 0, f"{itr}"
            assert pitr.end_timestamp == 0, f"{itr}"
        elif phase == 2:
            assert pitr.start_timestamp > 0, f"{itr}"
            assert pitr.end_timestamp > 0, f"{itr}"
            assert pitr.end_timestamp >= pitr.start_timestamp, f"{itr}"

            memory_alloc_cnt[op_id]["starting_addr"].add(pitr.address)
            memory_alloc_cnt[op_id]["size"].add(pitr.allocation_size)
            memory_alloc_cnt[op_id]["agent"].add(pitr.agent_id.handle)
        else:
            assert phase == 1 or phase == 2, f"{itr}"

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


def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in ["hsa_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in api_corr_ids.keys()
            api_corr_ids[corr_id] = itr

    alloc_corr_ids = {}
    for titr in ["memory_allocations"]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in alloc_corr_ids.keys()
            alloc_corr_ids[corr_id] = itr

    retired_corr_ids = {}
    for itr in sdk_data["buffer_records"]["retired_correlation_ids"]:
        corr_id = itr["internal_correlation_id"]
        assert corr_id not in retired_corr_ids.keys()
        retired_corr_ids[corr_id] = itr

    api_corr_ids = _sort_dict(api_corr_ids)
    alloc_corr_ids = _sort_dict(alloc_corr_ids)
    retired_corr_ids = _sort_dict(retired_corr_ids)

    for cid, itr in alloc_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        assert (retired_ts - end_ts) > 0, f"correlation-id: {cid}, data: {itr}"

    for cid, itr in api_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        assert (retired_ts - end_ts) > 0, f"correlation-id: {cid}, data: {itr}"

    assert len(api_corr_ids.keys()) == (len(retired_corr_ids.keys()))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
