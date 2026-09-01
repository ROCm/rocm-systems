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
# Followed by:
#   hipEventRecordWithFlags exercise
#   Deferred wait: large kernel -> record -> wait -> kernel on other stream
#   Coalescing: 3x hipEventRecord on a hipEventDisableTiming event
#
# This produces at least 8 hipEventRecord and 8 hipStreamWaitEvent calls.
# The first iteration may produce an extra barrier from initialization.

# rocprofiler_hip_event_operation_t values
HIP_EVENT_RECORD = 1
HIP_EVENT_WAIT = 2

BUFFER_TRACING_HIP_EVENT = 38

HIP_EVENT_API_NAMES = {
    "hipEventRecord",
    "hipEventRecordWithFlags",
    "hipStreamWaitEvent",
}


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


RECORD_API_NAMES = {"hipEventRecord", "hipEventRecord_spt", "hipEventRecordWithFlags"}
WAIT_API_NAMES = {"hipStreamWaitEvent", "hipStreamWaitEvent_spt"}


def test_hip_event_record_count(json_data):
    """Verify RECORD and WAIT completion counts are plausible relative to API calls.

    Not all API calls produce completions:
    - hipEventRecord during graph capture produces no barrier (no completion)
    - hipStreamWaitEvent on the same stream, already-complete event, or during
      graph capture produces no barrier (no completion)

    We verify:
    - RECORD completions > 0
    - RECORD completions <= hipEventRecord API call count (never more than calls)
    - WAIT completions > 0
    - WAIT completions <= hipStreamWaitEvent API call count
    """
    data = json_data["rocprofiler-sdk-tool"]
    records = data["buffer_records"]["hip_event"]
    hip_api = data["buffer_records"]["hip_api"]
    string_table = data["strings"]["buffer_records"]

    def get_operation_name(kind_id, op_id):
        return string_table[kind_id]["operations"][op_id]

    record_count = sum(1 for r in records if r.operation == HIP_EVENT_RECORD)
    wait_count = sum(1 for r in records if r.operation == HIP_EVENT_WAIT)

    api_record_count = sum(
        1 for r in hip_api if get_operation_name(r.kind, r.operation) in RECORD_API_NAMES
    )
    api_wait_count = sum(
        1 for r in hip_api if get_operation_name(r.kind, r.operation) in WAIT_API_NAMES
    )

    assert record_count > 0, "No RECORD completions found"
    assert record_count <= api_record_count, (
        f"RECORD completions ({record_count}) > hipEventRecord API calls "
        f"({api_record_count})"
    )
    assert wait_count > 0, "No WAIT completions found"
    assert wait_count <= api_wait_count, (
        f"WAIT completions ({wait_count}) > hipStreamWaitEvent API calls "
        f"({api_wait_count})"
    )


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
        # end_timestamp is a CPU timestamp taken when the async handler fires,
        # which can occur after fini_time is recorded on a different thread.
        # Unlike kernel dispatch (which uses GPU profiling timestamps), barrier
        # completion timestamps are not bounded by fini_time.


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
    string_table = data["strings"]["buffer_records"]

    def get_operation_name(kind_id, op_id):
        return string_table[kind_id]["operations"][op_id]

    event_corr_ids = set(r.correlation_id.internal for r in hip_event_records)

    hip_event_api_corr_ids = set(
        r.correlation_id.internal
        for r in hip_api_records
        if get_operation_name(r.kind, r.operation) in HIP_EVENT_API_NAMES
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


def test_hip_event_deferred_wait(json_data):
    """Verify that hipStreamWaitEvent produces WAIT completion records even when
    CLR does not emit a standalone barrier (the deferred wait path).

    The test binary launches a large kernel on stream0, records an event, then
    immediately calls hipStreamWaitEvent on stream1 followed by a kernel launch
    on stream1. Because the event is unlikely to have completed, CLR folds the
    dependency into the next dispatch's barrier as a dep_signal rather than
    emitting a standalone wait barrier. The profiler should detect this and
    produce a WAIT completion record on the waiting stream's queue.

    We verify that at least one WAIT record exists where:
    - queue_id != source_queue_id (cross-stream)
    - The WAIT record's queue_id matches a queue that also has kernel dispatch
      records (proving the WAIT was resolved via a kernel dispatch barrier)
    """
    data = json_data["rocprofiler-sdk-tool"]
    hip_event_records = data["buffer_records"]["hip_event"]
    kernel_records = data["buffer_records"]["kernel_dispatch"]

    wait_records = [r for r in hip_event_records if r.operation == HIP_EVENT_WAIT]
    cross_stream_waits = [
        r for r in wait_records if r.queue_id.handle != r.source_queue_id.handle
    ]

    assert len(cross_stream_waits) > 0, "No cross-stream WAIT completion records"

    kernel_queues = set(r.dispatch_info.queue_id.handle for r in kernel_records)

    waits_on_dispatch_queues = [
        r for r in cross_stream_waits if r.queue_id.handle in kernel_queues
    ]
    assert len(waits_on_dispatch_queues) > 0, (
        f"No WAIT completion records on queues that have kernel dispatches. "
        f"WAIT queues: {set(r.queue_id.handle for r in cross_stream_waits)}, "
        f"kernel queues: {kernel_queues}"
    )


def test_hip_event_wait_queue_identity(json_data):
    """Verify WAIT records report the waiting queue, not the recording queue.

    For a cross-stream wait where event was recorded on stream0 and
    hipStreamWaitEvent was called on stream1, the WAIT record must have:
    - queue_id = stream1's queue (the waiting queue)
    - source_queue_id = stream0's queue (the recording queue)

    We verify:
    - WAIT queue_id is a queue with kernel dispatches (the waiting stream)
    - WAIT source_queue_id is a queue with RECORD records (the recording stream)
    - The two are different
    """
    data = json_data["rocprofiler-sdk-tool"]
    records = data["buffer_records"]["hip_event"]
    kernel_records = data["buffer_records"]["kernel_dispatch"]

    record_queues = set(
        r.queue_id.handle for r in records if r.operation == HIP_EVENT_RECORD
    )
    kernel_queues = set(r.dispatch_info.queue_id.handle for r in kernel_records)

    wait_records = [
        r
        for r in records
        if r.operation == HIP_EVENT_WAIT and r.queue_id.handle != r.source_queue_id.handle
    ]

    for w in wait_records:
        assert w.queue_id.handle in kernel_queues, (
            f"WAIT record queue_id {w.queue_id.handle} is not a kernel dispatch "
            f"queue. This suggests the WAIT is reporting the recording queue "
            f"instead of the waiting queue. Kernel queues: {kernel_queues}"
        )
        assert w.source_queue_id.handle in record_queues, (
            f"WAIT record source_queue_id {w.source_queue_id.handle} is not a "
            f"known recording queue. Record queues: {record_queues}"
        )
        assert w.queue_id.handle != w.source_queue_id.handle, (
            f"WAIT record queue_id == source_queue_id ({w.queue_id.handle}), "
            f"expected different queues for cross-stream wait"
        )


def test_hip_event_duplicate_wait(json_data):
    """Verify that multiple hipStreamWaitEvent calls on the same event do not
    cause crashes, ref-count leaks, or lost records.

    The test binary records event0 on stream0, then calls
    hipStreamWaitEvent on both stream1 and stream2. Depending on GPU timing,
    one or both waits may produce WAIT records: CLR short-circuits when the
    event has already completed. Whether or not both produce records, the
    process must exit cleanly (no ref-count leaks from the multimap path or
    the cleanup-on-destroy path).

    We verify that WAIT records exist on at least 2 different queues in total
    (from the main cross-stream loop and/or the duplicate wait scenario),
    which is a timing-independent invariant.
    """
    data = json_data["rocprofiler-sdk-tool"]
    records = data["buffer_records"]["hip_event"]

    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]
    all_wait_queues = set(w.queue_id.handle for w in wait_records)

    assert len(all_wait_queues) >= 2, (
        f"Expected WAIT records on at least 2 different queues "
        f"(cross-stream dependencies from main loop and/or duplicate wait). "
        f"Found queues: {all_wait_queues}"
    )


def test_hip_event_no_same_stream_wait(json_data):
    """Verify same-stream waits do not produce WAIT completion records.

    The test binary calls hipStreamWaitEvent(stream0, event0, 0) where event0
    was recorded on stream0. CLR short-circuits this (same queue), so no
    barrier is dispatched and no WAIT record should appear with
    queue_id == source_queue_id.
    """
    records = json_data["rocprofiler-sdk-tool"]["buffer_records"]["hip_event"]
    wait_records = [r for r in records if r.operation == HIP_EVENT_WAIT]

    same_stream_waits = [
        r for r in wait_records if r.queue_id.handle == r.source_queue_id.handle
    ]
    assert len(same_stream_waits) == 0, (
        f"Found {len(same_stream_waits)} WAIT records with queue_id == source_queue_id "
        f"(same-stream waits should not produce completion records)"
    )


def test_hip_event_destroy_cleanup(json_data):
    """Verify that destroying an event with a pending wait does not leak.

    The test binary records destroy_event on stream0, calls
    hipStreamWaitEvent(stream1, destroy_event) to register a pending wait,
    then destroys the event before any kernel on stream1 consumes it. The
    pending wait should be cleaned up by erase_event_info. We verify:
    - At least one event handle has RECORD completions but zero WAIT completions
      (the destroyed event's wait was cleaned up, not consumed)
    - The process exits cleanly (implicit: no hang from leaked ref counts)
    """
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
        f"(destroy_event scenario). record_handles={record_handles}, "
        f"wait_handles={wait_handles}"
    )


def test_hip_event_graph_capture_exclusion(json_data):
    """Verify that hipEventRecord during graph capture does not produce records.

    The test binary records capture_event on a stream that is in graph capture
    mode. No barrier is dispatched during capture, so no RECORD completion should
    appear for that event. We verify that RECORD completions are strictly fewer
    than total hipEventRecord API calls — the capture call is excluded.
    """
    data = json_data["rocprofiler-sdk-tool"]
    records = data["buffer_records"]["hip_event"]
    hip_api = data["buffer_records"]["hip_api"]
    string_table = data["strings"]["buffer_records"]

    def get_operation_name(kind_id, op_id):
        return string_table[kind_id]["operations"][op_id]

    record_count = sum(1 for r in records if r.operation == HIP_EVENT_RECORD)
    api_record_count = sum(
        1 for r in hip_api if get_operation_name(r.kind, r.operation) in RECORD_API_NAMES
    )

    assert record_count < api_record_count, (
        f"Expected RECORD completions ({record_count}) to be strictly less than "
        f"hipEventRecord API calls ({api_record_count}): "
        f"at least one call was made during graph capture and should be excluded."
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
