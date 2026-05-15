#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    node_exists("rocjpeg_api_traces", sdk_data["callback_records"])

    node_exists("names", sdk_data["buffer_records"])
    node_exists("rocjpeg_api_traces", sdk_data["buffer_records"])


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
    for titr in ["rocjpeg_api_traces"]:
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


def test_internal_correlation_ids(input_data):
    """Assure correlation ids are unique"""
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in ["hsa_api_traces", "hip_api_traces", "rocjpeg_api_traces"]:
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
    for titr in ["rocjpeg_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in ["rocjpeg_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert (
                itr["thread_id"] == itr["correlation_id"]["external"]
            ), f"[{titr}] {itr}"
            assert itr["thread_id"] in extern_corr_ids, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"

        for itr in sdk_data["callback_records"][titr]:
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


def test_rocjpeg_traces(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    callback_records = sdk_data["callback_records"]
    buffer_records = sdk_data["buffer_records"]

    rocjpeg_bf_traces = sdk_data["buffer_records"]["rocjpeg_api_traces"]
    rocjpeg_api_bf_ops = get_operation(buffer_records, "ROCJPEG_API")
    assert len(rocjpeg_api_bf_ops[1]) == 9

    rocjpeg_cb_traces = sdk_data["callback_records"]["rocjpeg_api_traces"]
    rocjpeg_api_cb_ops = get_operation(callback_records, "ROCJPEG_API")

    # If rocJPEG tracing is not supported, end early
    if len(rocjpeg_bf_traces) <= 2:
        return pytest.skip("rocdecode tracing unavailable")
    assert (
        rocjpeg_api_bf_ops[1] == rocjpeg_api_cb_ops[1] and len(rocjpeg_api_cb_ops[1]) == 9
    )

    # check that buffer and callback records agree
    phase_enter_count = 0
    phase_end_count = 0

    api_calls = []

    for api_call in rocjpeg_cb_traces:
        if api_call["phase"] == 1:
            phase_enter_count += 1
            api_calls.append(rocjpeg_api_cb_ops[1][api_call["operation"]])
        if api_call["phase"] == 2:
            phase_end_count += 1

    assert phase_enter_count == phase_end_count == len(rocjpeg_bf_traces)

    for call in [
        "rocJpegCreate",
        "rocJpegStreamCreate",
        "rocJpegStreamParse",
        "rocJpegGetImageInfo",
        "rocJpegDecode",
        "rocJpegDestroy",
        "rocJpegStreamDestroy",
    ]:
        assert call in api_calls


def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in ["hsa_api_traces", "hip_api_traces", "rocjpeg_api_traces"]:
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
