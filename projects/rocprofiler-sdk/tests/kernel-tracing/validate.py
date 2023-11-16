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

    node_exists("kernel-tracing-test-tool", data)
    node_exists("agents", data["kernel-tracing-test-tool"])
    node_exists("call_stack", data["kernel-tracing-test-tool"])
    node_exists("callback_records", data["kernel-tracing-test-tool"])
    node_exists("buffer_records", data["kernel-tracing-test-tool"])

    node_exists("names", data["kernel-tracing-test-tool"]["callback_records"])
    node_exists("code_objects", data["kernel-tracing-test-tool"]["callback_records"])
    node_exists("kernel_symbols", data["kernel-tracing-test-tool"]["callback_records"])
    node_exists("hsa_api_traces", data["kernel-tracing-test-tool"]["callback_records"])

    node_exists("names", data["kernel-tracing-test-tool"]["buffer_records"])
    node_exists("kernel_dispatches", data["kernel-tracing-test-tool"]["buffer_records"])
    node_exists("memory_copies", data["kernel-tracing-test-tool"]["buffer_records"], 0)
    node_exists("hsa_api_traces", data["kernel-tracing-test-tool"]["buffer_records"])


def test_timestamps(input_data):
    data = input_data

    cb_start = {}
    cb_end = {}
    for itr in data["kernel-tracing-test-tool"]["callback_records"]["hsa_api_traces"]:
        cid = itr["record"]["correlation_id"]["internal"]
        phase = itr["record"]["phase"]
        if phase == 1:
            cb_start[cid] = itr["timestamp"]
        elif phase == 2:
            cb_end[cid] = itr["timestamp"]
            assert cb_start[cid] <= itr["timestamp"]
        else:
            assert phase == 1 or phase == 2

    for itr in data["kernel-tracing-test-tool"]["buffer_records"]["hsa_api_traces"]:
        assert itr["start_timestamp"] <= itr["end_timestamp"]

    for itr in data["kernel-tracing-test-tool"]["buffer_records"]["kernel_dispatches"]:
        assert itr["start_timestamp"] < itr["end_timestamp"]
        assert itr["correlation_id"]["internal"] > 0
        assert itr["correlation_id"]["external"] > 0

        api_start = cb_start[itr["correlation_id"]["internal"]]
        api_end = cb_end[itr["correlation_id"]["internal"]]
        assert api_start < itr["start_timestamp"]
        assert api_end <= itr["end_timestamp"]


def test_internal_correlation_ids(input_data):
    data = input_data

    api_corr_ids = []
    for itr in data["kernel-tracing-test-tool"]["callback_records"]["hsa_api_traces"]:
        api_corr_ids.append(itr["record"]["correlation_id"]["internal"])

    for itr in data["kernel-tracing-test-tool"]["buffer_records"]["hsa_api_traces"]:
        api_corr_ids.append(itr["correlation_id"]["internal"])

    api_corr_ids_sorted = sorted(api_corr_ids)
    api_corr_ids_unique = list(set(api_corr_ids))

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_external_correlation_ids(input_data):
    data = input_data

    for itr in data["kernel-tracing-test-tool"]["callback_records"]["hsa_api_traces"]:
        assert itr["record"]["thread_id"] == itr["record"]["correlation_id"]["external"]

    for itr in data["kernel-tracing-test-tool"]["buffer_records"]["hsa_api_traces"]:
        assert itr["thread_id"] == itr["correlation_id"]["external"]


def test_kernel_ids(input_data):
    data = input_data

    symbol_info = {}
    for itr in data["kernel-tracing-test-tool"]["callback_records"]["kernel_symbols"]:
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

    for itr in data["kernel-tracing-test-tool"]["buffer_records"]["kernel_dispatches"]:
        assert itr["kernel_id"] in symbol_info.keys()


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
