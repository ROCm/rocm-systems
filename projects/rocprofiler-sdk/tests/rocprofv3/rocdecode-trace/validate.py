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


def test_rocdeocde(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    rocdecode_data = buffer_records["rocdecode_api"]
    assert len(rocdecode_data) > 0
    # Ensure that the input file path has the correct string saved
    assert len(rocdecode_data[0].args) > 1
    assert rocdecode_data[0].args[1].name == "input_file_path"
    assert (
        rocdecode_data[0].args[1].value.split("/")[-1]
        == "AMD_driving_virtual_20-H265.265"
    )

    _, bf_op_names = get_operation(data, "ROCDECODE_API")

    assert len(bf_op_names) == 16

    rocdecode_reported_agent_ids = set()
    # check buffering data
    for node in rocdecode_data:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node
        assert "thread_id" in node
        assert "args" in node
        assert "retval" in node

        assert node.size > 0
        assert node.thread_id > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp
        assert len(node.args) > 0

        assert (
            data.strings.buffer_records[node.kind].kind == "ROCDECODE_API"
            or data.strings.buffer_records[node.kind].kind == "ROCDECODE_API_EXT"
        )
        assert (
            data.strings.buffer_records[node.kind].operations[node.operation]
            in bf_op_names
        )


def test_csv_data(csv_data):
    assert len(csv_data) > 0, "Expected non-empty csv data"

    api_calls = []

    for row in csv_data:
        assert "Domain" in row, "'Domain' was not present in csv data for rocdecode-trace"
        assert (
            "Function" in row
        ), "'Function' was not present in csv data for rocdecode-trace"
        assert (
            "Process_Id" in row
        ), "'Process_Id' was not present in csv data for rocdecode-trace"
        assert (
            "Thread_Id" in row
        ), "'Thread_Id' was not present in csv data for rocdecode-trace"
        assert (
            "Correlation_Id" in row
        ), "'Correlation_Id' was not present in csv data for rocdecode-trace"
        assert (
            "Start_Timestamp" in row
        ), "'Start_Timestamp' was not present in csv data for rocdecode-trace"
        assert (
            "End_Timestamp" in row
        ), "'End_Timestamp' was not present in csv data for rocdecode-trace"

        api_calls.append(row["Function"])

        assert row["Domain"] in ("ROCDECODE_API", "ROCDECODE_API_EXT")
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) > 0
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) < int(row["End_Timestamp"])

    for call in [
        "rocDecCreateBitstreamReader",
        "rocDecGetBitstreamCodecType",
        "rocDecGetBitstreamBitDepth",
        "rocDecCreateVideoParser",
        "rocDecGetBitstreamPicData",
        "rocDecGetDecoderCaps",
        "rocDecCreateDecoder",
        "rocDecDecodeFrame",
        "rocDecParseVideoData",
        "rocDecGetVideoFrame",
        "rocDecGetDecodeStatus",
        "rocDecDestroyBitstreamReader",
    ]:
        assert call in api_calls


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("rocdecode_api",),
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data,
        json_data,
        ("rocdecode_api",),
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
