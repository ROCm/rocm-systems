#!/usr/bin/env python3

import sys
import pytest


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    data = input_data

    node_exists("rocprofiler-sdk-json-tool", data)

    sdk_data = data["rocprofiler-sdk-json-tool"]

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("hip_api_traces", sdk_data["callback_records"], 0)
    node_exists("marker_api_traces", sdk_data["callback_records"])

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatches", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 0)
    node_exists("hsa_api_traces", sdk_data["buffer_records"])
    node_exists("hip_api_traces", sdk_data["buffer_records"], 0)
    node_exists("marker_api_traces", sdk_data["buffer_records"])


def test_timestamps(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    cb_start = {}
    cb_end = {}
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            cid = itr["record"]["correlation_id"]["internal"]
            phase = itr["record"]["phase"]
            if phase == 1:
                cb_start[cid] = itr["timestamp"]
            elif phase == 2:
                cb_end[cid] = itr["timestamp"]
                assert cb_start[cid] <= itr["timestamp"]
            else:
                assert phase == 1 or phase == 2

        for itr in sdk_data["buffer_records"][titr]:
            assert itr["start_timestamp"] <= itr["end_timestamp"]

    for itr in sdk_data["buffer_records"]["memory_copies"]:
        assert itr["start_timestamp"] <= itr["end_timestamp"]

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
        assert itr["start_timestamp"] < itr["end_timestamp"]
        assert itr["correlation_id"]["internal"] > 0
        assert itr["correlation_id"]["external"] > 0

        api_start = cb_start[itr["correlation_id"]["internal"]]
        api_end = cb_end[itr["correlation_id"]["internal"]]
        assert api_start < itr["start_timestamp"]
        assert api_end <= itr["end_timestamp"]


def test_internal_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    api_corr_ids = []
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            api_corr_ids.append(itr["record"]["correlation_id"]["internal"])

        for itr in sdk_data["buffer_records"][titr]:
            api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
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
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["record"]["correlation_id"]["external"] > 0
            assert (
                itr["record"]["thread_id"] == itr["record"]["correlation_id"]["external"]
            )
            extern_corr_ids.append(itr["record"]["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in ["hsa_api_traces", "marker_api_traces", "hip_api_traces"]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            assert itr["thread_id"] in extern_corr_ids
            assert itr["correlation_id"]["external"] in extern_corr_ids

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
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
        phase = itr["record"]["phase"]
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

    for itr in sdk_data["buffer_records"]["kernel_dispatches"]:
        assert itr["kernel_id"] in symbol_info.keys()


def test_async_copy_direction(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    # Direction values:
    #   0 == ??? (unknown)
    #   1 == H2H (host to host)
    #   2 == H2D (host to device)
    #   3 == D2H (device to host)
    #   4 == D2D (device to device)
    async_dir_cnt = dict([(idx, 0) for idx in range(0, 5)])
    for itr in sdk_data["buffer_records"]["memory_copies"]:
        op_id = itr["operation"]
        async_dir_cnt[op_id] += 1

    # in the reproducible-runtime test which generates the input file,
    # we don't expect any async memory copy operations
    assert async_dir_cnt[0] == 0
    assert async_dir_cnt[1] == 0
    assert async_dir_cnt[2] == 0
    assert async_dir_cnt[3] == 0
    assert async_dir_cnt[4] == 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
