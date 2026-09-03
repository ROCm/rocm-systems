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
        assert len(data[name]) >= min_len


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

    node_exists("names", sdk_data.callback_records)
    node_exists("code_objects", sdk_data.callback_records)
    node_exists("kernel_symbols", sdk_data.callback_records)
    node_exists("hsa_api_traces", sdk_data.callback_records)
    node_exists("hip_api_traces", sdk_data.callback_records, 0)
    node_exists("marker_api_traces", sdk_data.callback_records)
    node_exists("rccl_api_traces", sdk_data.callback_records, 0)
    node_exists("ompt_traces", sdk_data.callback_records)
    node_exists("kernel_dispatch", sdk_data.callback_records)

    node_exists("names", sdk_data.buffer_records)
    node_exists("kernel_dispatch", sdk_data.buffer_records)
    node_exists("memory_copies", sdk_data.buffer_records, 0)
    node_exists("hsa_api_traces", sdk_data.buffer_records)
    node_exists("hip_api_traces", sdk_data.buffer_records, 0)
    node_exists("marker_api_traces", sdk_data.buffer_records)
    node_exists("rccl_api_traces", sdk_data.buffer_records, 0)
    node_exists("ompt_traces", sdk_data.buffer_records)
    node_exists("retired_correlation_ids", sdk_data.buffer_records)


def test_size_entries(input_data):
    # check that size fields are > 0 but account for function arguments
    # which are named "size"
    def check_size(data, bt):
        if "size" in data.keys():
            if isinstance(data.size, str) and bt.endswith('["args"]'):
                pass
            else:
                assert data.size > 0, f"origin: {bt}"

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
    for titr in [
        "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
        "ompt_traces",
    ]:
        for itr in sdk_data.callback_records[titr]:
            cid = itr.correlation_id.internal
            phase = itr.phase
            if phase == 1:
                cb_start[cid] = itr.timestamp
            elif phase == 2:
                cb_end[cid] = itr.timestamp
                assert cb_start[cid] <= itr.timestamp

        for itr in sdk_data.buffer_records[titr]:
            assert itr.start_timestamp <= itr.end_timestamp

    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data.buffer_records[titr]:
            assert itr.start_timestamp < itr.end_timestamp, f"[{titr}] {itr}"
            assert itr.correlation_id.internal > 0, f"[{titr}] {itr}"
            assert itr.correlation_id.external > 0, f"[{titr}] {itr}"
            assert sdk_data.metadata.init_time < itr.start_timestamp, f"[{titr}] {itr}"
            assert sdk_data.metadata.init_time < itr.end_timestamp, f"[{titr}] {itr}"
            assert sdk_data.metadata.fini_time > itr.start_timestamp, f"[{titr}] {itr}"
            assert sdk_data.metadata.fini_time > itr.end_timestamp, f"[{titr}] {itr}"

            api_start = cb_start[itr.correlation_id.internal]
            # api_end = cb_end[itr.correlation_id.internal]
            assert api_start < itr.start_timestamp, f"[{titr}] {itr}"
            # assert api_end <= itr.end_timestamp, f"[{titr}] {itr}"


def test_total_runtime(input_data):
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    runtime_data = []
    for itr in sdk_data.buffer_records.kernel_dispatch:
        elapsed = itr.end_timestamp - itr.start_timestamp
        runtime_data.append(elapsed)  # in nanoseconds

    expected_runtime = 1.0e-6  # one millisecond

    assert sum(runtime_data) >= expected_runtime


def test_internal_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in [
        "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
        "ompt_traces",
    ]:
        for itr in sdk_data.callback_records[titr]:
            api_corr_ids.append(itr.correlation_id.internal)

        for itr in sdk_data.buffer_records[titr]:
            api_corr_ids.append(itr.correlation_id.internal)

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    for itr in sdk_data.buffer_records.kernel_dispatch:
        assert itr.correlation_id.internal in api_corr_ids_unique

    for itr in sdk_data.buffer_records.memory_copies:
        assert itr.correlation_id.internal in api_corr_ids_unique

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in [
        "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
        "ompt_traces",
    ]:
        for itr in sdk_data.buffer_records[titr]:
            corr_id = itr.correlation_id.internal
            assert corr_id not in api_corr_ids.keys()
            api_corr_ids[corr_id] = itr

    async_corr_ids = {}
    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data.buffer_records[titr]:
            corr_id = itr.correlation_id.internal
            assert corr_id not in async_corr_ids.keys()
            async_corr_ids[corr_id] = itr

    retired_corr_ids = {}
    for itr in sdk_data.buffer_records.retired_correlation_ids:
        corr_id = itr.internal_correlation_id
        assert corr_id not in retired_corr_ids.keys()
        retired_corr_ids[corr_id] = itr

    api_corr_ids = _sort_dict(api_corr_ids)
    async_corr_ids = _sort_dict(async_corr_ids)
    retired_corr_ids = _sort_dict(retired_corr_ids)

    for cid, itr in async_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid].timestamp
        end_ts = itr.end_timestamp
        assert (retired_ts - end_ts) > 0, f"correlation-id: {cid}, data: {itr}"

    for cid, itr in api_corr_ids.items():
        assert cid in retired_corr_ids.keys()
        retired_ts = retired_corr_ids[cid].timestamp
        end_ts = itr.end_timestamp
        assert (retired_ts - end_ts) > 0, f"correlation-id: {cid}, data: {itr}"

    assert len(api_corr_ids.keys()) == (len(retired_corr_ids.keys()))


def test_external_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    extern_corr_ids = []
    for titr in [
        "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
        "ompt_traces",
    ]:
        for itr in sdk_data.callback_records[titr]:
            assert itr.correlation_id.external > 0
            assert itr.thread_id == itr.correlation_id.external
            extern_corr_ids.append(itr.correlation_id.external)

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in [
        "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
        "ompt_traces",
    ]:
        for itr in sdk_data.buffer_records[titr]:
            assert itr.correlation_id.external > 0, f"[{titr}] {itr}"
            assert itr.thread_id == itr.correlation_id.external, f"[{titr}] {itr}"
            assert itr.thread_id in extern_corr_ids, f"[{titr}] {itr}"
            assert itr.correlation_id.external in extern_corr_ids, f"[{titr}] {itr}"

    for titr in ["kernel_dispatch", "memory_copies"]:
        for itr in sdk_data.buffer_records[titr]:
            assert itr.correlation_id.external > 0, f"[{titr}] {itr}"
            assert itr.correlation_id.external in extern_corr_ids, f"[{titr}] {itr}"


def test_kernel_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    symbol_info = {}
    for itr in sdk_data.callback_records.kernel_symbols:
        phase = itr.phase
        payload = itr.payload
        kern_id = payload.kernel_id

        assert phase == 1 or phase == 2
        assert kern_id > 0
        if phase == 1:
            assert len(payload.kernel_name) > 0
            symbol_info[kern_id] = payload
        elif phase == 2:
            assert payload.kernel_id in symbol_info.keys()
            assert payload.kernel_name == symbol_info[kern_id].kernel_name

    for itr in sdk_data.buffer_records.kernel_dispatch:
        assert itr.dispatch_info.kernel_id in symbol_info.keys()

    for itr in sdk_data.callback_records.kernel_dispatch:
        assert itr.payload.dispatch_info.kernel_id in symbol_info.keys()


def test_kernel_dispatch_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    num_dispatches = len(sdk_data.buffer_records.kernel_dispatch)
    num_cb_dispatches = len(sdk_data.callback_records.kernel_dispatch)

    assert num_cb_dispatches == (3 * num_dispatches)

    bf_seq_ids = []
    for itr in sdk_data.buffer_records.kernel_dispatch:
        bf_seq_ids.append(itr.dispatch_info.dispatch_id)

    cb_seq_ids = []
    for itr in sdk_data.callback_records.kernel_dispatch:
        cb_seq_ids.append(itr.payload.dispatch_info.dispatch_id)

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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
