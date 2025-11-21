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
        assert len(data[name]) >= min_len, f"{name}:\n{data}"


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


def get_operation_name(record, kind_idx, op_idx):
    for idx, itr in enumerate(record["names"]):
        if idx == kind_idx:
            return itr["operations"][op_idx]

    return None


def groupby_corr_id(trace_item, op_id=None):
    """
    If op_id is not none, returns records only with that operation ID

    {
        corr_id-1: record with internal corr_id = corr_id-1
        corr_id-2: record with internal corr_id = corr_id-2
        ...
    }
    """
    from rocprofiler_sdk.pytest_utils.dotdict import dotdict

    ret = {}

    for x in trace_item:
        if op_id is not None and x.operation != op_id:
            continue

        corr_id = x.correlation_id["internal"]

        if corr_id in ret.keys():
            assert False, f"Duplicate internal corr_id {corr_id}"
        else:
            ret[corr_id] = x

    return dotdict(ret)


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
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    node_exists("host_functions", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("hip_api_traces", sdk_data["callback_records"], 0)
    node_exists("marker_api_traces", sdk_data["callback_records"])
    node_exists("kernel_dispatch", sdk_data["callback_records"])
    node_exists("memory_copies", sdk_data["callback_records"], 24)

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatch", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 12)
    node_exists("hsa_api_traces", sdk_data["buffer_records"])
    node_exists("hip_api_traces", sdk_data["buffer_records"], 0)
    node_exists("marker_api_traces", sdk_data["buffer_records"])
    node_exists("retired_correlation_ids", sdk_data["buffer_records"])


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

        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"


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


def test_kernel_dispatch_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    num_dispatches = len(sdk_data["buffer_records"]["kernel_dispatch"])
    num_cb_dispatches = len(sdk_data["callback_records"]["kernel_dispatch"])

    assert num_cb_dispatches == (3 * num_dispatches)

    bf_seq_ids = []
    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        bf_seq_ids.append(itr["dispatch_info"]["dispatch_id"])

    cb_seq_ids = []
    for itr in sdk_data["callback_records"]["kernel_dispatch"]:
        cb_seq_ids.append(itr["payload"]["dispatch_info"]["dispatch_id"])

    bf_seq_ids = sorted(bf_seq_ids)
    cb_seq_ids = sorted(cb_seq_ids)

    assert (3 * len(bf_seq_ids)) == len(cb_seq_ids)

    assert bf_seq_ids[0] == cb_seq_ids[0]
    assert bf_seq_ids[-1] == cb_seq_ids[-1]

    def get_uniq(data):
        return list(set(data))

    bf_seq_ids_uniq = get_uniq(bf_seq_ids)
    cb_seq_ids_uniq = get_uniq(cb_seq_ids)

    assert bf_seq_ids == bf_seq_ids_uniq
    assert len(cb_seq_ids) == (3 * len(cb_seq_ids_uniq))
    assert len(bf_seq_ids) == num_dispatches
    assert len(bf_seq_ids_uniq) == num_dispatches
    assert len(cb_seq_ids_uniq) == num_dispatches


def test_async_copy_direction(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    # Direction values:
    #   0 == ??? (unknown)
    #   1 == H2H (host to host)
    #   2 == H2D (host to device)
    #   3 == D2H (device to host)
    #   4 == D2D (device to device)
    default_async_dir_cnt = dict([(idx, 0) for idx in range(0, 5)])
    thread_async_dir_cnt = {}
    for itr in sdk_data.buffer_records.memory_copies:
        tid = itr.thread_id
        if tid not in thread_async_dir_cnt.keys():
            thread_async_dir_cnt[tid] = default_async_dir_cnt
        op_id = itr.operation
        assert op_id > 1, f"{itr}"
        assert op_id < 4, f"{itr}"
        thread_async_dir_cnt[tid][op_id] += 1

    for itr in sdk_data.callback_records.memory_copies:
        tid = itr.thread_id
        if tid not in thread_async_dir_cnt.keys():
            thread_async_dir_cnt[tid] = default_async_dir_cnt
        op_id = itr.operation
        assert op_id > 1, f"{itr}"
        assert op_id < 4, f"{itr}"
        thread_async_dir_cnt[tid][op_id] += 1

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
        else:
            assert phase == 1 or phase == 2, f"{itr}"

    # in the transpose test which generates the input file,
    # two threads each perform one H2D + one D2H memory copy.
    # there are at least two callback records (phase start +
    # phase end) and one buffer record for each memory copy,
    # i.e., at least 3 records per memory copy
    assert len(thread_async_dir_cnt) == 2, f"{thread_async_dir_cnt}"
    for tid, async_dir_cnt in thread_async_dir_cnt.items():
        min_copy_records = 3
        assert async_dir_cnt[0] == 0
        assert async_dir_cnt[1] == 0
        assert async_dir_cnt[2] >= min_copy_records, f"TID={tid}:\n\t{async_dir_cnt}"
        assert async_dir_cnt[3] >= min_copy_records, f"TID={tid}:\n\t{async_dir_cnt}"
        assert async_dir_cnt[4] == 0
        # HIP memory copies may be decomposed into more than one
        # memory copy at the HSA level so require it to be a multiple
        # of min_copy_records
        assert (
            async_dir_cnt[2] % min_copy_records
        ) == 0, f"TID={tid}:\n\t{async_dir_cnt}"
        assert (
            async_dir_cnt[3] % min_copy_records
        ) == 0, f"TID={tid}:\n\t{async_dir_cnt}"


def test_ancestor_ids(input_data):
    """
    This test ensures that each memcpy can be traced back to
    a hipMemcpyAsync through ancestor IDs
    """
    from rocprofiler_sdk.pytest_utils.dotdict import dotdict

    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]
    buffer_records = sdk_data.buffer_records
    memcopies = buffer_records.memory_copies

    _, hip_op_ids = get_operation(buffer_records, "HIP_RUNTIME_API")
    hip_memcopy_id = get_operation(buffer_records, "HIP_RUNTIME_API", "hipMemcpyAsync")

    # dict with { internal id : record }
    hip_memcopies = groupby_corr_id(buffer_records.hip_api_traces, hip_memcopy_id)

    hsa_records = groupby_corr_id(buffer_records.hsa_api_traces)
    hip_records = groupby_corr_id(buffer_records.hip_api_traces)

    accounted_for_hip_ids = []

    for memcpy in memcopies:
        parent_hsa_call = hsa_records[memcpy.correlation_id.internal]
        parent_hip_call = hip_records[parent_hsa_call.correlation_id.ancestor]

        assert (
            parent_hip_call.thread_id == parent_hsa_call.thread_id
        ), "Expected hsa and hip calls to be on the same thread"
        assert hip_op_ids[parent_hip_call.operation] == "hipMemcpyAsync"
        accounted_for_hip_ids.append(parent_hip_call.correlation_id.internal)

    # Ensure we looked through all HIP entries
    assert (
        set(accounted_for_hip_ids) == set(hip_memcopies.keys()),
        "Expected to account for all HIP memcpy calls through ancestor ID lookup",
    )


def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]
    buffer_records = sdk_data["buffer_records"]
    api_name_info = {}

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in test_api_traces:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            name = get_operation_name(buffer_records, itr["kind"], itr["operation"])

            assert corr_id not in api_corr_ids.keys()
            assert name is not None, f"{itr}"

            api_corr_ids[corr_id] = itr
            api_name_info[corr_id] = name

    async_corr_ids = {}
    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in async_corr_ids.keys()
            async_corr_ids[corr_id] = itr

    retired_corr_ids = {}
    for itr in sdk_data["buffer_records"]["retired_correlation_ids"]:
        corr_id = itr["internal_correlation_id"]
        assert corr_id not in retired_corr_ids.keys()
        retired_corr_ids[corr_id] = itr

    api_corr_ids = _sort_dict(api_corr_ids)
    async_corr_ids = _sort_dict(async_corr_ids)
    retired_corr_ids = _sort_dict(retired_corr_ids)

    #
    # verify all the correlation ids were retired
    #
    num_api_corr_ids = len(api_corr_ids.keys())
    num_retired_corr_ids = len(retired_corr_ids.keys())

    missing_retired_corr_ids = [
        itr for itr in api_corr_ids.keys() if itr not in retired_corr_ids.keys()
    ]
    # log in case of failure
    sys.stderr.flush()
    for itr in missing_retired_corr_ids:
        name = api_name_info[itr]
        info = api_corr_ids[itr]
        sys.stderr.write(f"- unretired corr id: {itr} :: {name} :: {info}\n")
    sys.stderr.flush()

    assert (
        num_api_corr_ids == num_retired_corr_ids
    ), f"correlation ids not retired:\n\t{missing_retired_corr_ids}"

    #
    #   verify the retirement timestamp is >= the end timestamp of the records
    #
    for cid, itr in api_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        name = api_name_info[cid]
        assert (
            retired_ts - end_ts
        ) >= 0, f"\n\tcorr: {cid}\n\tname: {name}\n\tdata: {itr}"

    # allow the retired timestamp to be 10 usec earlier than async end timestamp
    # since the async timestamps undergo conversion from the GPU clock domain to
    # the CPU clock domain. 10 microseconds was arbitrarily chosen to be an
    # acceptable amount of inaccuracy -- in an ideal world, retired_ts should
    # always be >= end_ts
    usec = 1000
    supported_fuzzing = 10 * usec

    for cid, itr in async_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid]["timestamp"]
        end_ts = itr["end_timestamp"]
        name = api_name_info[cid]
        assert (
            retired_ts - end_ts
        ) >= -supported_fuzzing, f"\n\tcorr: {cid}\n\tname: {name}\n\tdata: {itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
