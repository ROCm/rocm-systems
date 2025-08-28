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


def test_stream_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    memory_copies_data = buffer_records["memory_copy"]
    assert len(memory_copies_data) > 0

    # Expect non-null stream ids to be set between 1 and 6 inclusive
    expected_stream_ids = set([i for i in range(1, 7)])
    memory_copy_streams = defaultdict(int)
    # check buffering data
    for node in memory_copies_data:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node
        assert "thread_id" in node
        assert "stream_id" in node

        assert node.size > 0
        assert node.thread_id > 0
        assert node.start_timestamp < node.end_timestamp

        stream_id = node.stream_id.handle
        memory_copy_streams[stream_id] += 1
    # 2 memory copies with null stream and 4 with hipStreamPerThread
    assert memory_copy_streams[0] == 6
    # Exactly 1 memory copy to device and 1 memory copy to host
    for i in expected_stream_ids:
        assert memory_copy_streams[i] == 2


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    assert pftrace_data.empty == False
    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("kernel", "memory_copy"),
    )


def test_csv_data(memory_copy_csv_data):
    assert len(memory_copy_csv_data) > 0, "Expected non-empty memory copy csv data"

    expected_stream_ids = set([i for i in range(1, 7)])
    memory_copy_streams = defaultdict(int)
    for row in memory_copy_csv_data:
        assert "Stream_Id" in row

        stream_id = int(row["Stream_Id"])
        memory_copy_streams[stream_id] += 1

    # 2 memory copies with null stream and 4 with hipStreamPerThread
    assert memory_copy_streams[0] == 6
    # Exactly 1 memory copy to device and 1 memory copy to host
    for i in expected_stream_ids:
        assert memory_copy_streams[i] == 2


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
