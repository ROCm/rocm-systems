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

# Records produced on the legacy default stream carry stream_id 0: get_stream_id in
# hip/stream.cpp resolves hipStreamLegacy to nullptr and looks it up in the stream map,
# which only ever holds streams passed to hipStreamCreate. The app records default_event
# twice and legacy_event once on the default stream, so exactly that many records may carry
# a zero stream_id.
DEFAULT_STREAM_RECORDS = 3


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

    # stream_id is checked separately: zero is legitimate for the default stream.
    zero_stream = [r for r in records if r.stream_id.handle == 0]
    assert len(zero_stream) <= DEFAULT_STREAM_RECORDS, (
        f"{len(zero_stream)} records carry stream_id 0, more than the "
        f"{DEFAULT_STREAM_RECORDS} default-stream records the app produces"
    )
    assert len(zero_stream) < len(records), "every record carries stream_id 0"


def test_no_hip_api_timestamps(json_data):
    """Verify timestamps are ordered when HIP API tracing is disabled."""
    data = json_data["rocprofiler-sdk-tool"]
    init_time = data["metadata"]["init_time"]

    for itr in data["buffer_records"]["hip_event"]:
        assert (
            itr.start_timestamp <= itr.end_timestamp
        ), f"start > end: {itr.start_timestamp} > {itr.end_timestamp}"
        assert (
            itr.start_timestamp > init_time
        ), f"start {itr.start_timestamp} before init {init_time}"


def test_no_hip_api_cross_stream(json_data):
    """Verify WAIT records show cross-stream dependencies without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]

    cross_stream = [
        r for r in wait_records if r.queue_id.handle != r.source_queue_id.handle
    ]
    assert len(cross_stream) > 0, "No cross-stream WAIT records found"


def test_no_hip_api_source_correlation(json_data):
    """Verify source_correlation_id links each WAIT to the RECORD it depends on.

    A RECORD reports its own internal correlation id. A WAIT reports the internal
    correlation id of the RECORD that recorded the awaited event, or zero when that
    record was not seen, in which case source_queue_id is zero too.
    """
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    record_records = {
        r.correlation_id.internal: r for r in records if r.operation == HIP_EVENT_RECORD
    }
    for r in record_records.values():
        assert r.source_correlation_id == r.correlation_id.internal, (
            f"RECORD (correlation {r.correlation_id.internal}) reports "
            f"source_correlation_id {r.source_correlation_id}, expected its own"
        )

    linked = 0
    for w in (r for r in records if r.operation == HIP_EVENT_WAIT):
        if w.source_queue_id.handle == 0:
            assert w.source_correlation_id == 0, (
                f"WAIT (correlation {w.correlation_id.internal}) has no source queue but "
                f"reports source_correlation_id {w.source_correlation_id}"
            )
            continue

        rec = record_records.get(w.source_correlation_id)
        assert rec is not None, (
            f"WAIT (correlation {w.correlation_id.internal}) reports "
            f"source_correlation_id {w.source_correlation_id}, which is not a RECORD"
        )
        assert rec.hip_event_handle == w.hip_event_handle, (
            f"WAIT (correlation {w.correlation_id.internal}) on event "
            f"{w.hip_event_handle:#x} links to a RECORD on event {rec.hip_event_handle:#x}"
        )
        assert rec.queue_id.handle == w.source_queue_id.handle, (
            f"WAIT (correlation {w.correlation_id.internal}) source_queue_id "
            f"{w.source_queue_id.handle} differs from its RECORD's queue {rec.queue_id.handle}"
        )
        assert rec.correlation_id.internal < w.correlation_id.internal, (
            f"WAIT (correlation {w.correlation_id.internal}) links to a later RECORD "
            f"(correlation {rec.correlation_id.internal})"
        )
        assert w.end_timestamp >= rec.end_timestamp, (
            f"WAIT (correlation {w.correlation_id.internal}) ended at {w.end_timestamp}, "
            f"before its RECORD ended at {rec.end_timestamp}"
        )
        linked += 1

    assert linked > 0, "No WAIT record links to a RECORD through source_correlation_id"


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


def test_no_hip_api_record_source_queue(json_data):
    """Verify RECORD operations have source_queue_id == queue_id."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    record_records = [r for r in records if r.operation == HIP_EVENT_RECORD]

    for r in record_records:
        assert r.source_queue_id.handle == r.queue_id.handle, (
            f"RECORD source_queue_id ({r.source_queue_id.handle}) != "
            f"queue_id ({r.queue_id.handle})"
        )


def test_no_hip_api_graph_capture_exclusion(json_data):
    """Verify graph capture records are excluded without HIP API tracing."""
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]

    record_handles = set(
        r.hip_event_handle for r in records if r.operation == HIP_EVENT_RECORD
    )

    # The binary creates 12 events; 11 are recorded outside graph capture (event0, event1,
    # destroy_event, default_event, same_stream_event, ordered_event, marker_event, ipc_event,
    # legacy_event, per_thread_event, coalesce_event) and capture_event must be excluded.
    # The observed count can be lower than 11: hipEventDestroy frees an address that a later
    # hipEventCreate can reuse, so two distinct events may report the same handle. Only the
    # upper bound is meaningful -- exceeding it means a record was emitted for the captured
    # event.
    assert len(record_handles) <= 11, (
        f"Expected at most 11 unique RECORD handles (capture_event should be excluded). "
        f"Found {len(record_handles)}: {record_handles}. "
        f"If 12 handles, graph capture exclusion may have failed."
    )
    # Loose floor: handle reuse makes an exact count unreliable, but tracing being
    # broken entirely would show far fewer than this.
    assert len(record_handles) >= 4, (
        f"Expected at least 4 unique RECORD handles. Found {len(record_handles)}: "
        f"{record_handles}."
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
