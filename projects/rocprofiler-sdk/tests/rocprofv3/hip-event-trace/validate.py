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

# The hip-events test app does 4 iterations of:
#   kernel on stream0 -> hipEventRecord(event0, stream0)
#   hipStreamWaitEvent(stream1, event0) -> kernel on stream1
#   hipEventRecord(event1, stream1) -> hipStreamWaitEvent(stream0, event1)
#
# This produces at least 8 hipEventRecord and 8 hipStreamWaitEvent calls.
# The first iteration may produce an extra barrier from initialization.

# rocprofiler_hip_event_operation_t values
HIP_EVENT_RECORD = 1
HIP_EVENT_WAIT = 2

BUFFER_TRACING_HIP_EVENT = 38

# rocprofiler_hip_runtime_api_id_t values. These are positional in the enum
# defined in source/include/rocprofiler-sdk/hip/runtime_api_id.h and will
# shift if entries are inserted before them.
HIP_EVENT_RECORD_API_ID = 81
HIP_STREAM_WAIT_EVENT_API_ID = 348


def test_hip_event_json_structure(json_data):
    """Verify hip_event records exist in JSON buffer_records."""
    data = json_data["rocprofiler-sdk-tool"]
    assert "buffer_records" in data
    assert "hip_event" in data["buffer_records"]
    assert len(data["buffer_records"]["hip_event"]) > 0, "No hip_event buffer records"


def test_hip_event_operations(json_data):
    """Verify both RECORD and WAIT operations are present."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    operations = set(r.operation for r in records)

    assert HIP_EVENT_RECORD in operations, f"Missing RECORD (1) in {operations}"
    assert HIP_EVENT_WAIT in operations, f"Missing WAIT (2) in {operations}"


def test_hip_event_record_count(json_data):
    """Verify expected number of RECORD and WAIT completions."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    record_count = sum(1 for r in records if r.operation == HIP_EVENT_RECORD)
    wait_count = sum(1 for r in records if r.operation == HIP_EVENT_WAIT)

    assert record_count >= 12, f"Expected >= 12 RECORD completions, got {record_count}"
    assert wait_count >= 9, f"Expected >= 9 WAIT completions, got {wait_count}"


def test_hip_event_timestamps(json_data):
    """Verify timestamps are ordered and within the profiling window."""
    data = json_data["rocprofiler-sdk-tool"]
    init_time = data["metadata"]["init_time"]
    fini_time = data["metadata"]["fini_time"]

    for itr in data["buffer_records"]["hip_event"]:
        assert (
            itr.start_timestamp < itr.end_timestamp
        ), f"start >= end: {itr.start_timestamp} >= {itr.end_timestamp}"
        assert (
            itr.start_timestamp > init_time
        ), f"start {itr.start_timestamp} before init {init_time}"
        assert (
            itr.end_timestamp < fini_time
        ), f"end {itr.end_timestamp} after fini {fini_time}"


def test_hip_event_fields(json_data):
    """Verify all required fields are present and valid."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    for r in records:
        assert r.size > 0
        assert r.kind == BUFFER_TRACING_HIP_EVENT
        assert r.thread_id > 0
        assert r.agent_id.handle > 0
        assert r.queue_id.handle > 0
        assert r.hip_event_handle > 0
        assert r.stream_id.handle > 0
        assert r.correlation_id.internal > 0


def test_hip_event_cross_stream(json_data):
    """Verify WAIT operations show cross-stream dependencies."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]

    cross_stream = [
        r for r in wait_records if r.queue_id.handle != r.source_queue_id.handle
    ]
    assert len(cross_stream) > 0, "No cross-stream WAIT records found"

    for r in cross_stream:
        assert r.source_queue_id.handle > 0, "source_queue_id is zero"


def test_hip_event_handle_consistency(json_data):
    """Verify the same hip_event_handle appears in both RECORD and WAIT records."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    record_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_RECORD
    )
    wait_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_WAIT
    )

    shared = record_handles & wait_handles
    assert len(shared) > 0, (
        f"No shared event handles between RECORD and WAIT: "
        f"record={record_handles}, wait={wait_handles}"
    )


def test_hip_event_record_source_queue(json_data):
    """Verify RECORD operations have source_queue_id == queue_id."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    record_records = [r for r in records if r.operation == HIP_EVENT_RECORD]

    for r in record_records:
        assert r.source_queue_id.handle == r.queue_id.handle, (
            f"RECORD source_queue_id ({r.source_queue_id.handle}) != "
            f"queue_id ({r.queue_id.handle})"
        )


def test_hip_event_correlation(json_data):
    """Verify hip_event correlation_id.internal values match HIP API calls."""
    data = json_data["rocprofiler-sdk-tool"]
    hip_event_records = data["buffer_records"]["hip_event"]
    hip_api_records = data["buffer_records"]["hip_api"]

    event_corr_ids = set(r.correlation_id.internal for r in hip_event_records)

    hip_event_api_corr_ids = set(
        r.correlation_id.internal
        for r in hip_api_records
        if r.operation in (HIP_EVENT_RECORD_API_ID, HIP_STREAM_WAIT_EVENT_API_ID)
    )

    matched = event_corr_ids & hip_event_api_corr_ids
    assert len(matched) > 0, (
        f"No correlation ID matches between hip_event and HIP API records. "
        f"event corr_ids={event_corr_ids}, api corr_ids={hip_event_api_corr_ids}"
    )


def test_hip_event_coalescing(json_data):
    """Verify that consecutive hipEventRecord calls on a hipEventDisableTiming
    event each produce a completion record, even when CLR coalesces barriers."""
    data = json_data["rocprofiler-sdk-tool"]
    hip_event_records = data["buffer_records"]["hip_event"]
    hip_api_records = data["buffer_records"]["hip_api"]

    # Find the coalesce_event handle: it's the event created with
    # hipEventCreateWithFlags (API ID for hipEventCreateWithFlags is distinct
    # from hipEventCreate). We identify it as the event handle that appears
    # in exactly 3 consecutive hipEventRecord API calls at the end.
    # Simpler approach: find RECORD buffer records grouped by hip_event_handle,
    # and check that one handle has >= 3 records.
    record_records = [r for r in hip_event_records if r.operation == HIP_EVENT_RECORD]

    handle_counts = {}
    for r in record_records:
        h = r.hip_event_handle
        handle_counts[h] = handle_counts.get(h, 0) + 1

    coalesced_handles = [h for h, c in handle_counts.items() if c >= 3]
    assert len(coalesced_handles) > 0, (
        f"Expected at least one event handle with >= 3 RECORD completions "
        f"(coalescing scenario). Counts per handle: {handle_counts}"
    )

    coalesced_handle = coalesced_handles[0]
    coalesced_records = [
        r for r in record_records if r.hip_event_handle == coalesced_handle
    ]

    corr_ids = set(r.correlation_id.internal for r in coalesced_records)
    assert len(corr_ids) >= 3, (
        f"Coalesced RECORD completions should have distinct correlation IDs, "
        f"got {len(corr_ids)} unique out of {len(coalesced_records)} records"
    )


def test_rocpd_hip_events(rocpd_data, json_data):
    """Verify rocpd hip_events table matches JSON buffer records."""
    js_data = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    rpd_data = rocpd_data.execute("SELECT * FROM hip_events").fetchall()

    assert len(rpd_data) == len(js_data), (
        f"rocpd hip_events has {len(rpd_data)} rows, "
        f"JSON hip_event has {len(js_data)} records"
    )

    assert len(rpd_data) > 0, "No records in rocpd hip_events"


def test_rocpd_hip_events_cross_stream(rocpd_data):
    """Verify rocpd hip_events contains cross-stream WAIT records."""
    cross_stream = rocpd_data.execute(
        "SELECT COUNT(*) FROM hip_events "
        "WHERE name LIKE '%HIP_EVENT_WAIT%' AND queue_id != source_queue_id"
    ).fetchone()[0]

    assert cross_stream > 0, "No cross-stream WAIT records in rocpd"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
