#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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


def test_rocshmem(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    rocshmem_data = buffer_records["rocshmem_api"]
    # If rocSHMEM tracing is not supported, end early
    if len(rocshmem_data) == 0:
        return pytest.skip("rocshmem tracing unavailable")

    _, bf_op_names = get_operation(data, "ROCSHMEM_API")

    # rocSHMEM dispatch table currently has 9 host-stream operations
    assert len(bf_op_names) == 9

    # check buffering data
    for node in rocshmem_data:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "end_timestamp" in node
        assert "start_timestamp" in node
        assert "thread_id" in node
        # rocprofv3 consumes the extended (EXT) buffer records, so every record
        # carries the function argument(s) and return value.
        assert "args" in node
        assert "retval" in node

        assert node.size > 0
        assert node.thread_id > 0
        assert node.start_timestamp > 0
        assert node.end_timestamp > 0
        assert node.start_timestamp < node.end_timestamp

        # every traced rocSHMEM host-stream API takes at least one argument, so the
        # args payload must be present and well-formed ({type, name, value}).
        assert len(node.args) > 0
        for arg in node.args:
            assert "type" in arg
            assert "name" in arg
            assert "value" in arg

        # the buffer service registers the EXT domain, so records report the
        # ROCSHMEM_API_EXT kind (the basic ROCSHMEM_API kind shares the same ops).
        assert data.strings.buffer_records[node.kind].kind in (
            "ROCSHMEM_API",
            "ROCSHMEM_API_EXT",
        )
        assert (
            data.strings.buffer_records[node.kind].operations[node.operation]
            in bf_op_names
        )

    # The structure checks above prove every EXT record carries well-formed
    # args/retval; this additionally pins a known argument name + value (parity
    # with rocdecode-trace, which asserts args[1].name == "input_file_path").
    # The rocshmem-demo always transfers kPerPeBytes (= 64) via
    # putmem_on_stream(dest, source, nelems, pe, stream).
    putmem_records = [
        node
        for node in rocshmem_data
        if data.strings.buffer_records[node.kind].operations[node.operation]
        == "putmem_on_stream"
    ]
    assert len(putmem_records) > 0, "expected at least one putmem_on_stream record"

    putmem_args = {arg.name: arg.value for arg in putmem_records[0].args}
    for expected in ("dest", "source", "nelems", "pe", "stream"):
        assert expected in putmem_args, f"putmem_on_stream missing arg '{expected}'"

    # kPerPeBytes is a compile-time constant (64) in the rocshmem-demo app.
    assert int(putmem_args["nelems"], 0) == 64, putmem_args["nelems"]


def test_rocpd_data(rocpd_conn, json_data):
    # rocpd (SQLite) is the default rocprofv3 output format, so --rocshmem-trace
    # must populate the database as well. Skip in lock-step with the other format
    # tests when rocSHMEM tracing is unavailable.
    if len(json_data["rocprofiler-sdk-tool"]["buffer_records"]["rocshmem_api"]) == 0:
        return pytest.skip("rocshmem tracing unavailable")

    cur = rocpd_conn.cursor()

    # rocSHMEM host-stream APIs are ranged operations, so rocprofv3 stores them as
    # rows in the rocpd `regions` view under the ROCSHMEM_API[_EXT] category (the
    # category string is set in source/lib/output/domain_type.cpp; rocprofv3
    # consumes the EXT buffer records, so the category is ROCSHMEM_API_EXT).
    region_rows = cur.execute(
        "SELECT name, tid, start, end FROM regions "
        "WHERE category IN ('ROCSHMEM_API', 'ROCSHMEM_API_EXT')"
    ).fetchall()
    assert len(region_rows) > 0, "rocpd database contains no rocSHMEM regions"

    op_names = set()
    for name, tid, start, end in region_rows:
        assert tid > 0
        assert start > 0
        assert end > 0
        assert start < end
        op_names.add(name)

    # Every traced rocSHMEM host-stream API should surface as a named region in
    # the rocpd database (parity with the json/csv per-call assertions above).
    for call in [
        "barrier_all_on_stream",
        "quiet_on_stream",
        "sync_all_on_stream",
        "alltoallmem_on_stream",
        "broadcastmem_on_stream",
        "getmem_on_stream",
        "putmem_on_stream",
        "putmem_signal_on_stream",
        "signal_wait_until_on_stream",
    ]:
        assert call in op_names, f"rocpd missing rocSHMEM operation '{call}'"

    # Function arguments are stored in rocpd_arg and surfaced (joined to their
    # region) by the `region_args` view. Pin the putmem_on_stream signature and its
    # known nelems value (= kPerPeBytes = 64) for parity with the json/csv checks.
    putmem_arg_names = {
        row[0]
        for row in cur.execute(
            "SELECT DISTINCT ra.name FROM region_args ra "
            "INNER JOIN regions r ON r.id = ra.id AND r.guid = ra.guid "
            "WHERE r.category IN ('ROCSHMEM_API', 'ROCSHMEM_API_EXT') "
            "AND r.name = 'putmem_on_stream'"
        ).fetchall()
    }
    for expected in ("dest", "source", "nelems", "pe", "stream"):
        assert (
            expected in putmem_arg_names
        ), f"putmem_on_stream missing arg '{expected}' in rocpd"

    nelems_values = [
        row[0]
        for row in cur.execute(
            "SELECT ra.value FROM region_args ra "
            "INNER JOIN regions r ON r.id = ra.id AND r.guid = ra.guid "
            "WHERE r.category IN ('ROCSHMEM_API', 'ROCSHMEM_API_EXT') "
            "AND r.name = 'putmem_on_stream' AND ra.name = 'nelems'"
        ).fetchall()
    ]
    assert nelems_values, "no nelems arg recorded for putmem_on_stream in rocpd"
    assert any(int(value, 0) == 64 for value in nelems_values), nelems_values


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
