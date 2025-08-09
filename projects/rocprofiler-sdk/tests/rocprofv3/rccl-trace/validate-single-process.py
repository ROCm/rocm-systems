#!/usr/bin/env python3

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


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["strings"]["buffer_records"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]
            else:
                for oidx, oname in enumerate(itr["operations"]):
                    if op_name == oname:
                        return oidx
    return None


def _test_rccl_api_json_traces(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    callback_records = data["callback_records"]
    buffer_records = data["buffer_records"]

    rccl_bf_traces = buffer_records["rccl_api"]
    rccl_api_bf_ops = get_operation(data, "RCCL_API")
    assert len(rccl_api_bf_ops[1]) == 37

    api_calls = []

    for api_call in rccl_bf_traces:
        api_calls.append(rccl_api_bf_ops[1][api_call["operation"]])

    parent_process = False
    if "ncclGetUniqueId" in api_calls:
        parent_process = True

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
    # these checks must be in sync with rocprofv3 test's validate
    alloc_size = 64 * 1024 * 1024  # 64MB in the test
    elem_count = alloc_size // 4  # sizeof(float)

    def validate_nccl_count(arg):
        assert (
            int(arg["value"]) == elem_count
        ), f'Expected {elem_count} elements, got {arg["value"]}'

    def validate_nccl_dtype(arg):
        assert arg["value"] == "7"  # ncclFloat = 7

    def validate_nccl_op(arg):
        assert arg["value"] == "0"  # ncclSum = 0

    for record in rccl_bf_traces:
        op_name = rccl_api_bf_ops[1][record["operation"]]

        if op_name == "ncclAllReduce":
            checked_args = 0

            for arg in record["args"]:

                if arg["name"] == "count":
                    checked_args += 1
                    validate_nccl_count(arg)

                if arg["name"] == "datatype":
                    checked_args += 1
                    validate_nccl_dtype(arg)

                if arg["name"] == "op":
                    checked_args += 1
                    validate_nccl_op(arg)

            assert (
                checked_args == 3
            ), f"Expected to validate 3 args, found only {checked_args}"

    return parent_process


def _test_rccl_api_csv_traces(csv_data):
    assert len(csv_data) > 0, "Expected non-empty csv data"

    api_calls = []

    for row in csv_data:
        assert "Domain" in row, "'Domain' was not present in csv data for rccl-trace"
        assert "Function" in row, "'Function' was not present in csv data for rccl-trace"
        assert (
            "Process_Id" in row
        ), "'Process_Id' was not present in csv data for rccl-trace"
        assert (
            "Thread_Id" in row
        ), "'Thread_Id' was not present in csv data for rccl-trace"
        assert (
            "Correlation_Id" in row
        ), "'Correlation_Id' was not present in csv data for rccl-trace"
        assert (
            "Start_Timestamp" in row
        ), "'Start_Timestamp' was not present in csv data for rccl-trace"
        assert (
            "End_Timestamp" in row
        ), "'End_Timestamp' was not present in csv data for rccl-trace"

        api_calls.append(row["Function"])

        assert row["Domain"] == "RCCL_API_EXT"
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) > 0
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) < int(row["End_Timestamp"])

    for call in [
        "ncclAllReduce",
        "ncclCommInitAll",
        "ncclCommGetAsyncError",
    ]:
        assert call in api_calls

    parent_process = False
    if "ncclGetUniqueId" in api_calls:
        parent_process = True

    return parent_process


def test_rccl_sp_trace(json_sp_input_data, csv_sp_input_data):
    assert len(json_sp_input_data) != 0, "Expected non-zero json output files"
    assert len(csv_sp_input_data) != 0, "Expected non-zero csv output files"
    assert len(json_sp_input_data) == len(csv_sp_input_data)

    def test_data(func, data):
        """
        func: should return True if data is from a parent process
        data: list of json or csv data to pass to func
        """
        num_parents = 0
        num_children = 0

        for _data in data:
            is_parent = func(_data)
            if is_parent:
                num_parents += 1
            else:
                num_children += 1

        assert num_parents == 1, "Expected one parent process"
        assert num_children + num_parents == len(
            json_sp_input_data
        ), "Expected parent + child processes to be same as number of files"

    test_data(_test_rccl_api_json_traces, json_sp_input_data)
    test_data(_test_rccl_api_csv_traces, csv_sp_input_data)


if __name__ == "__main__":
    print(sys.argv[1:], sys.stderr)
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
