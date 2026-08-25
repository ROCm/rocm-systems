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

# Validates hip event tracing when HIP API tracing is NOT enabled.
# This exercises the correlation ID self-construction path and verifies
# that hip event tracing works independently of HIP API tracing.

HIP_EVENT_RECORD = 1
HIP_EVENT_WAIT = 2
BUFFER_TRACING_HIP_EVENT = 38


def test_no_hip_api_records_present(json_data):
    """Verify HIP API records are absent (confirming HIP API tracing is off)."""
    data = json_data["rocprofiler-sdk-tool"]
    hip_api = data["buffer_records"]["hip_api"]
    assert len(hip_api) == 0, (
        f"Expected no HIP API records when running without --hip-runtime-trace, "
        f"but found {len(hip_api)}"
    )


def test_no_hip_api_hip_event_records_present(json_data):
    """Verify hip_event records exist without HIP API tracing."""
    data = json_data["rocprofiler-sdk-tool"]
    assert "hip_event" in data["buffer_records"]
    assert len(data["buffer_records"]["hip_event"]) > 0, "No hip_event buffer records"


def test_no_hip_api_both_operations(json_data):
    """Verify both RECORD and WAIT operations are present without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    operations = set(r.operation for r in records)

    assert HIP_EVENT_RECORD in operations, f"Missing RECORD (1) in {operations}"
    assert HIP_EVENT_WAIT in operations, f"Missing WAIT (2) in {operations}"


def test_no_hip_api_fields(json_data):
    """Verify all required fields are present and valid without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    for r in records:
        assert r.size > 0
        assert r.kind == BUFFER_TRACING_HIP_EVENT
        assert r.thread_id > 0
        assert r.agent_id.handle > 0
        assert r.queue_id.handle > 0
        assert r.hip_event_handle > 0
        assert r.correlation_id.internal > 0


def test_no_hip_api_cross_stream(json_data):
    """Verify WAIT records show cross-stream dependencies without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]

    cross_stream = [
        r for r in wait_records if r.queue_id.handle != r.source_queue_id.handle
    ]
    assert len(cross_stream) > 0, "No cross-stream WAIT records found"


def test_no_hip_api_no_same_stream_wait(json_data):
    """Verify same-stream waits do not produce WAIT records without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]

    same_stream_waits = [
        r for r in wait_records if r.queue_id.handle == r.source_queue_id.handle
    ]
    assert (
        len(same_stream_waits) == 0
    ), f"Found {len(same_stream_waits)} same-stream WAIT records"


def test_no_hip_api_destroy_cleanup(json_data):
    """Verify destroy with pending wait does not leak without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    record_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_RECORD
    )
    wait_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_WAIT
    )

    record_only = record_handles - wait_handles
    assert len(record_only) >= 1, (
        f"Expected at least one event handle with RECORD but no WAIT completions "
        f"(destroy_event scenario). Found {len(record_only)}: {record_only}"
    )


def test_no_hip_api_graph_capture_exclusion(json_data):
    """Verify graph capture records are excluded without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    record_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_RECORD
    )

    assert (
        len(record_handles) == 4
    ), f"Expected 4 unique RECORD handles, found {len(record_handles)}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
