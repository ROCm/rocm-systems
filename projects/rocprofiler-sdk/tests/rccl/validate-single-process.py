#!/usr/bin/env python3

from collections import defaultdict
import os
import sys
import pytest


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

    node_exists("agents", sdk_data)
    node_exists("call_stack", sdk_data)
    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)

    node_exists("names", sdk_data["callback_records"])
    node_exists("code_objects", sdk_data["callback_records"])
    node_exists("kernel_symbols", sdk_data["callback_records"])
    # Disabled for rccl
    # node_exists("hsa_api_traces", sdk_data["callback_records"])
    node_exists("hip_api_traces", sdk_data["callback_records"], 0)
    node_exists("marker_api_traces", sdk_data["callback_records"], 0)
    node_exists("rccl_api_traces", sdk_data["callback_records"], 0)

    node_exists("names", sdk_data["buffer_records"])
    node_exists("kernel_dispatch", sdk_data["buffer_records"])
    node_exists("memory_copies", sdk_data["buffer_records"], 0)
    # Disabled for rccl
    # node_exists("hsa_api_traces", sdk_data["buffer_records"])
    node_exists("hip_api_traces", sdk_data["buffer_records"], 0)
    node_exists("marker_api_traces", sdk_data["buffer_records"], 0)
    node_exists("rccl_api_traces", sdk_data["buffer_records"], 0)
    node_exists("retired_correlation_ids", sdk_data["buffer_records"])


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
    ]:
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
    for titr in [
        # "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
    ]:
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

    for itr in sdk_data["buffer_records"]["memory_allocations"]:
        assert itr["correlation_id"]["internal"] in api_corr_ids_unique

    len_corr_id_unq = len(api_corr_ids_unique)
    assert len(api_corr_ids) != len_corr_id_unq
    assert max(api_corr_ids_sorted) == len_corr_id_unq


def test_external_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    extern_corr_ids = []
    for titr in [
        # "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
    ]:
        for itr in sdk_data["callback_records"][titr]:
            assert itr["correlation_id"]["external"] > 0
            assert itr["thread_id"] == itr["correlation_id"]["external"]
            extern_corr_ids.append(itr["correlation_id"]["external"])

    extern_corr_ids = list(set(sorted(extern_corr_ids)))
    for titr in [
        # "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
    ]:
        for itr in sdk_data["buffer_records"][titr]:
            assert itr["correlation_id"]["external"] > 0, f"[{titr}] {itr}"
            assert (
                itr["thread_id"] == itr["correlation_id"]["external"]
            ), f"[{titr}] {itr}"
            assert itr["thread_id"] in extern_corr_ids, f"[{titr}] {itr}"
            assert itr["correlation_id"]["external"] in extern_corr_ids, f"[{titr}] {itr}"

    for titr in ["kernel_dispatch", "memory_copies", "memory_allocations"]:
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


def test_rccl_sp_api_traces(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    gpu_count = len([x for x in sdk_data["agents"] if x["type"] == 2])

    callback_records = sdk_data["callback_records"]
    buffer_records = sdk_data["buffer_records"]

    rccl_bf_traces = sdk_data["buffer_records"]["rccl_api_traces"]
    rccl_api_bf_ops = get_operation(buffer_records, "RCCL_API_EXT")
    assert len(rccl_api_bf_ops[1]) == 37

    rccl_cb_traces = sdk_data["callback_records"]["rccl_api_traces"]
    rccl_api_cb_ops = get_operation(callback_records, "RCCL_API")

    assert rccl_api_bf_ops[1] == rccl_api_cb_ops[1] and len(rccl_api_cb_ops[1]) == 37

    # check that buffer and callback records agree
    phase_enter_count = 0
    phase_end_count = 0

    api_calls = []

    for api_call in rccl_cb_traces:
        if api_call["phase"] == 1:
            phase_enter_count += 1
            api_calls.append(rccl_api_cb_ops[1][api_call["operation"]])
        if api_call["phase"] == 2:
            phase_end_count += 1

    assert phase_enter_count == phase_end_count == len(rccl_bf_traces)

    for call in [
        "ncclCommInitAll",
        "ncclGetUniqueId",
        "ncclGroupStart",
        "ncclGroupEnd",
        "ncclAllReduce",
        "ncclCommDestroy",
    ]:
        assert call in api_calls

    # check for buffer args
    # these checks must be in sync with json tool test's validate
    alloc_size = 64 * 1024 * 1024  # 64MB in the test
    elem_count = alloc_size // 4  # sizeof(float)

    def validate_nccl_count(value):
        assert int(value) == elem_count, f"Expected {elem_count} elements, got {value}"

    def validate_nccl_dtype(value):
        assert value == "7"  # ncclFloat = 7

    def validate_nccl_op(value):
        assert value == "0"  # ncclSum = 0

    for record in rccl_bf_traces:
        op_name = rccl_api_bf_ops[1][record["operation"]]

        if op_name == "ncclAllReduce":
            checked_args = 0

            for arg in record["args"]:

                if arg["name"] == "count":
                    checked_args += 1
                    validate_nccl_count(arg["value"])

                if arg["name"] == "datatype":
                    checked_args += 1
                    validate_nccl_dtype(arg["value"])

                if arg["name"] == "op":
                    checked_args += 1
                    validate_nccl_op(arg["value"])

            assert (
                checked_args == 3
            ), f"Expected to validate 3 args, found only {checked_args}"

    # check cakkback records args
    for record in rccl_cb_traces:
        op_name = rccl_api_cb_ops[1][record["operation"]]

        if op_name == "ncclAllReduce" and record["phase"] == 2:
            checked_args = 0

            for arg_name, value in record["args"].items():

                if arg_name == "count":
                    checked_args += 1
                    validate_nccl_count(value)

                if arg_name == "datatype":
                    checked_args += 1
                    validate_nccl_dtype(value)

                if arg_name == "op":
                    checked_args += 1
                    validate_nccl_op(value)

            assert (
                checked_args == 3
            ), f"Expected to validate 3 args, found only {checked_args}"


@pytest.mark.skip("Temporarily disabled")
def test_retired_correlation_ids(input_data):
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    def _sort_dict(inp):
        return dict(sorted(inp.items()))

    api_corr_ids = {}
    for titr in [
        # "hsa_api_traces",
        "marker_api_traces",
        "hip_api_traces",
        "rccl_api_traces",
    ]:
        for itr in sdk_data["buffer_records"][titr]:
            corr_id = itr["correlation_id"]["internal"]
            assert corr_id not in api_corr_ids.keys()
            api_corr_ids[corr_id] = itr

    async_corr_ids = {}
    for titr in ["kernel_dispatch", "memory_copies", "memory_allocation"]:
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

    for cid, itr in async_corr_ids.items():
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
