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


def get_operation(record, kind_name, op_name=None):
    for idx, itr in enumerate(record["strings"]["buffer_records"]):
        if kind_name == itr["kind"]:
            if op_name is None:
                return idx, itr["operations"]

            for oidx, oname in enumerate(itr["operations"]):
                if op_name == oname:
                    return oidx
    return None


def test_hipfile(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    hipfile_data = buffer_records["hipfile_api"]
    if len(hipfile_data) == 0:
        return pytest.skip("hipFILE tracing unavailable")

    _, bf_op_names = get_operation(data, "HIPFILE_API")

    assert len(bf_op_names) == 31

    observed_operations = set()
    for node in hipfile_data:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node
        assert "thread_id" in node

        assert node.size > 0
        assert node.thread_id > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp

        assert data.strings.buffer_records[node.kind].kind == "HIPFILE_API"

        operation_name = data.strings.buffer_records[node.kind].operations[node.operation]
        assert operation_name in bf_op_names
        observed_operations.add(operation_name)

    for call in [
        "hipFileGetVersion",
        "hipFileGetOpErrorString",
        "hipFileUseCount",
        "hipFileGetParameterSizeT",
        "hipFileSetParameterSizeT",
        "hipFileGetParameterBool",
        "hipFileSetParameterBool",
        "hipFileGetParameterString",
        "hipFileSetParameterString",
        "hipFileDriverOpen",
    ]:
        assert call in observed_operations


def test_csv_data(csv_data):
    if len(csv_data) == 0:
        return pytest.skip("hipFILE tracing unavailable")

    api_calls = []

    for row in csv_data:
        assert "Domain" in row, "'Domain' was not present in csv data for hipfile-trace"
        assert "Function" in row, "'Function' was not present in csv data for hipfile-trace"
        assert (
            "Process_Id" in row
        ), "'Process_Id' was not present in csv data for hipfile-trace"
        assert "Thread_Id" in row, "'Thread_Id' was not present in csv data for hipfile-trace"
        assert (
            "Correlation_Id" in row
        ), "'Correlation_Id' was not present in csv data for hipfile-trace"
        assert (
            "Start_Timestamp" in row
        ), "'Start_Timestamp' was not present in csv data for hipfile-trace"
        assert (
            "End_Timestamp" in row
        ), "'End_Timestamp' was not present in csv data for hipfile-trace"

        api_calls.append(row["Function"])

        assert row["Domain"] == "HIPFILE_API"
        assert int(row["Process_Id"]) > 0
        assert int(row["Thread_Id"]) > 0
        assert int(row["Start_Timestamp"]) > 0
        assert int(row["End_Timestamp"]) > 0
        assert int(row["Start_Timestamp"]) < int(row["End_Timestamp"])

    for call in [
        "hipFileGetVersion",
        "hipFileGetOpErrorString",
        "hipFileUseCount",
        "hipFileDriverOpen",
    ]:
        assert call in api_calls


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    if len(json_data["rocprofiler-sdk-tool"]["buffer_records"]["hipfile_api"]) == 0:
        return pytest.skip("hipFILE tracing unavailable")

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("hipfile_api",),
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    if len(json_data["rocprofiler-sdk-tool"]["buffer_records"]["hipfile_api"]) == 0:
        return pytest.skip("hipFILE tracing unavailable")

    rocprofv3.test_otf2_data(
        otf2_data,
        json_data,
        ("hipfile_api",),
    )


def test_rocpd_data(rocpd_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    if len(json_data["rocprofiler-sdk-tool"]["buffer_records"]["hipfile_api"]) == 0:
        return pytest.skip("hipFILE tracing unavailable")

    rocprofv3.test_rocpd_data(rocpd_data, json_data, ("hipfile_api",))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
