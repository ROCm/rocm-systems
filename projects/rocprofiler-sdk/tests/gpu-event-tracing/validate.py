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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import pytest


def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """Verify GPU events data exists in both callback and buffer records."""
    data = input_data
    sdk_data = data["rocprofiler-sdk-json-tool"]

    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)
    node_exists("gpu_events", sdk_data["callback_records"])
    node_exists("gpu_events", sdk_data["buffer_records"])


def test_gpu_event_operations(input_data):
    """Verify both RECORD_COMPLETE and WAIT_COMPLETE operations appear in buffer records."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    bf_records = sdk_data["buffer_records"]["gpu_events"]

    operations = set()
    for itr in bf_records:
        operations.add(itr["operation"])

    # ROCPROFILER_GPU_EVENT_WAIT_COMPLETE = 2
    # ROCPROFILER_GPU_EVENT_RECORD_COMPLETE = 4
    assert 2 in operations, f"Missing WAIT_COMPLETE (2) in operations: {operations}"
    assert 4 in operations, f"Missing RECORD_COMPLETE (4) in operations: {operations}"


def test_callback_phases(input_data):
    """Verify callback records have expected phase patterns.

    GPU events callbacks:
    - ENTER (phase=1) + EXIT (phase=2) for ENQUEUE operations
    - NONE (phase=0) for COMPLETE operations
    """
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    cb_records = sdk_data["callback_records"]["gpu_events"]

    phases_by_op = {}
    for itr in cb_records:
        op = itr["operation"]
        phase = itr["phase"]
        if op not in phases_by_op:
            phases_by_op[op] = set()
        phases_by_op[op].add(phase)

    # RECORD_ENQUEUE (3) must be present with ENTER(1) and EXIT(2)
    assert 3 in phases_by_op, "Missing RECORD_ENQUEUE (3) in callback records"
    assert 1 in phases_by_op[3], "Missing ENTER phase for RECORD_ENQUEUE"
    assert 2 in phases_by_op[3], "Missing EXIT phase for RECORD_ENQUEUE"

    # WAIT_ENQUEUE (1) may be present; if so it must have ENTER(1) and EXIT(2)
    if 1 in phases_by_op:
        assert 1 in phases_by_op[1], "Missing ENTER phase for WAIT_ENQUEUE"
        assert 2 in phases_by_op[1], "Missing EXIT phase for WAIT_ENQUEUE"

    # COMPLETE ops (2=WAIT_COMPLETE, 4=RECORD_COMPLETE) have NONE(0)
    assert 4 in phases_by_op, "Missing RECORD_COMPLETE (4) in callback records"
    assert 0 in phases_by_op[4], "Missing NONE phase for RECORD_COMPLETE"

    if 2 in phases_by_op:
        assert 0 in phases_by_op[2], "Missing NONE phase for WAIT_COMPLETE"


def test_timestamps(input_data):
    """Verify timestamps are valid for buffer records."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    init_time = sdk_data["metadata"]["init_time"]
    fini_time = sdk_data["metadata"]["fini_time"]

    for itr in sdk_data["buffer_records"]["gpu_events"]:
        assert itr["start_timestamp"] < itr["end_timestamp"], f"Bad timestamps: {itr}"
        assert itr["start_timestamp"] > init_time, f"start before init: {itr}"
        assert itr["end_timestamp"] < fini_time, f"end after fini: {itr}"


def test_event_info(input_data):
    """Verify event_info fields in buffer records."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    for itr in sdk_data["buffer_records"]["gpu_events"]:
        info = itr["event_info"]
        assert info["size"] > 0, f"Bad event_info size: {itr}"
        assert info["event_id"] > 0, f"Bad event_id: {itr}"
        assert info["issue_id"] > 0, f"Bad issue_id: {itr}"


def test_correlation_ids(input_data):
    """Verify correlation IDs are valid in buffer records.

    COMPLETE operations (2=WAIT_COMPLETE, 4=RECORD_COMPLETE) fire asynchronously
    on the GPU without a host-side HIP API call, so they have no external
    correlation ID. Only ENQUEUE-originated records carry one.
    """
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    for itr in sdk_data["buffer_records"]["gpu_events"]:
        assert itr["correlation_id"]["internal"] > 0, f"Bad internal corr_id: {itr}"
        if itr["operation"] not in (2, 4):
            assert itr["correlation_id"]["external"] > 0, (
                f"Bad external corr_id: {itr}"
            )


def test_size_entries(input_data):
    """Check that size fields are > 0."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    for itr in sdk_data["buffer_records"]["gpu_events"]:
        assert itr["size"] > 0

    for itr in sdk_data["callback_records"]["gpu_events"]:
        assert itr["payload"]["size"] > 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
