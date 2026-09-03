#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

EXPECTED_CALLS = [
    "hipFileGetVersion",
    "hipFileGetOpErrorString",
    "hipFileUseCount",
    "hipFileGetParameterSizeT",
    "hipFileSetParameterSizeT",
    "hipFileGetParameterBool",
    "hipFileSetParameterBool",
    "hipFileDriverOpen",
]


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

            for oidx, oname in enumerate(itr["operations"]):
                if op_name == oname:
                    return oidx
    return None


def operation_name(records, entry):
    return records["names"][entry["kind"]]["operations"][entry["operation"]]


def test_data_structure(input_data):
    data = input_data

    node_exists("rocprofiler-sdk-json-tool", data)

    sdk_data = data["rocprofiler-sdk-json-tool"]

    node_exists("metadata", sdk_data)
    node_exists("pid", sdk_data["metadata"])
    node_exists("main_tid", sdk_data["metadata"])
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])

    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("hipfile_api_traces", sdk_data["callback_records"])

    node_exists("names", sdk_data["buffer_records"])
    node_exists("hipfile_api_traces", sdk_data["buffer_records"])
    node_exists("hipfile_api_ext_traces", sdk_data["buffer_records"])


def test_size_entries(input_data):
    def check_size(data, bt):
        if "size" in data.keys():
            if isinstance(data["size"], str) and bt.endswith('["args"]'):
                pass
            else:
                assert data["size"] > 0, f"origin: {bt}"

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

    iterate_data(input_data, "input_data")


def test_hipfile_traces(input_data):
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    callback_records = sdk_data["callback_records"]
    buffer_records = sdk_data["buffer_records"]

    hipfile_cb_traces = callback_records["hipfile_api_traces"]
    hipfile_bf_traces = buffer_records["hipfile_api_traces"]
    hipfile_ext_traces = buffer_records["hipfile_api_ext_traces"]

    _, cb_op_names = get_operation(callback_records, "HIPFILE_API")
    _, bf_op_names = get_operation(buffer_records, "HIPFILE_API")
    _, ext_op_names = get_operation(buffer_records, "HIPFILE_API_EXT")

    assert cb_op_names == bf_op_names == ext_op_names
    assert len(cb_op_names) == (34 if "hipFileGetStatsL3" in cb_op_names else 31)
    assert len(hipfile_bf_traces) > 0
    assert len(hipfile_ext_traces) == len(hipfile_bf_traces)

    phase_enter_count = 0
    phase_exit_count = 0
    observed_operations = set()

    for api_call in hipfile_cb_traces:
        assert api_call["thread_id"] > 0
        assert api_call["timestamp"] > 0

        if api_call["phase"] == 1:
            phase_enter_count += 1
        elif api_call["phase"] == 2:
            phase_exit_count += 1
            observed_operations.add(operation_name(callback_records, api_call))
        else:
            assert api_call["phase"] in [1, 2]

    assert phase_enter_count == phase_exit_count == len(hipfile_bf_traces)

    for call in EXPECTED_CALLS:
        assert call in observed_operations

    for api_call in hipfile_bf_traces:
        assert api_call["thread_id"] > 0
        assert api_call["start_timestamp"] > 0
        assert api_call["end_timestamp"] > 0
        assert api_call["start_timestamp"] < api_call["end_timestamp"]
        assert operation_name(buffer_records, api_call) in cb_op_names

    observed_args = False
    for api_call in hipfile_ext_traces:
        assert api_call["thread_id"] > 0
        assert api_call["start_timestamp"] > 0
        assert api_call["end_timestamp"] > 0
        assert api_call["start_timestamp"] < api_call["end_timestamp"]
        assert operation_name(buffer_records, api_call) in ext_op_names
        assert "args" in api_call
        assert "retval" in api_call
        observed_args = observed_args or len(api_call["args"]) > 0

    assert observed_args


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
