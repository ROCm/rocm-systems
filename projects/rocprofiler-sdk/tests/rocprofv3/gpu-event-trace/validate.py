#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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


def test_gpu_event_json(json_data):
    """Validate GPU events exist in JSON output."""
    data = json_data["rocprofiler-sdk-tool"]
    buffer_records = data["buffer_records"]

    assert "gpu_events" in buffer_records
    gpu_events = buffer_records["gpu_events"]
    assert len(gpu_events) > 0, "No GPU event buffer records found"

    for node in gpu_events:
        assert "size" in node
        assert "kind" in node
        assert "operation" in node
        assert "correlation_id" in node
        assert "start_timestamp" in node
        assert "end_timestamp" in node
        assert "thread_id" in node
        assert "event_info" in node

        assert node.size > 0
        assert node.thread_id > 0


def test_gpu_event_timestamps(json_data):
    """Validate timestamps are ordered and fall within the profiling window."""
    data = json_data["rocprofiler-sdk-tool"]
    init_time = data["metadata"]["init_time"]
    fini_time = data["metadata"]["fini_time"]

    for itr in data["buffer_records"]["gpu_events"]:
        assert itr.start_timestamp < itr.end_timestamp, f"Bad timestamps: {itr}"
        assert itr.start_timestamp > init_time, f"start before init: {itr}"
        assert itr.end_timestamp < fini_time, f"end after fini: {itr}"


def test_gpu_event_operations(json_data):
    """Verify both RECORD_COMPLETE and WAIT_COMPLETE present."""
    data = json_data["rocprofiler-sdk-tool"]
    operations = set()
    for itr in data["buffer_records"]["gpu_events"]:
        operations.add(itr.operation)

    # WAIT_COMPLETE = 2, RECORD_COMPLETE = 4
    assert 2 in operations, f"Missing WAIT_COMPLETE in {operations}"
    assert 4 in operations, f"Missing RECORD_COMPLETE in {operations}"


def test_gpu_event_info(json_data):
    """Verify event_info fields in buffer records."""
    data = json_data["rocprofiler-sdk-tool"]

    for itr in data["buffer_records"]["gpu_events"]:
        info = itr.event_info
        assert info.size > 0, f"Bad event_info size: {itr}"
        assert info.event_id > 0, f"Bad event_id: {itr}"
        assert info.issue_id > 0, f"Bad issue_id: {itr}"


def test_gpu_event_correlation_ids(json_data):
    """Verify correlation IDs are valid in buffer records.

    COMPLETE operations (2=WAIT_COMPLETE, 4=RECORD_COMPLETE) fire asynchronously
    on the GPU without a host-side HIP API call, so they have no external
    correlation ID. Only ENQUEUE-originated records carry one.
    """
    data = json_data["rocprofiler-sdk-tool"]

    for itr in data["buffer_records"]["gpu_events"]:
        assert itr.correlation_id.internal > 0, f"Bad internal corr_id: {itr}"
        if itr.operation not in (2, 4):
            assert itr.correlation_id.external > 0, f"Bad external corr_id: {itr}"


def test_rocpd_gpu_events(rocpd_data, json_data):
    """Verify rocpd event_operations table matches JSON buffer records."""
    js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"]["gpu_events"]

    rpd_data = rocpd_data.execute("SELECT * FROM event_operations").fetchall()

    assert len(rpd_data) == len(js_data), (
        f"rocpd event_operations has {len(rpd_data)} rows, "
        f"JSON gpu_events has {len(js_data)} records"
    )

    num_total = rocpd_data.execute(
        "SELECT COUNT(*) FROM event_operations"
    ).fetchone()[0]
    num_unique_start = rocpd_data.execute(
        "SELECT COUNT(DISTINCT(start)) FROM event_operations"
    ).fetchone()[0]
    num_unique_end = rocpd_data.execute(
        "SELECT COUNT(DISTINCT(end)) FROM event_operations"
    ).fetchone()[0]

    assert num_total == num_unique_start == num_unique_end, (
        f"Duplicate records in event_operations: total={num_total}, "
        f"unique starts={num_unique_start}, unique ends={num_unique_end}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
