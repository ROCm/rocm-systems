#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import re
import sys
import pytest
import json


class TimeWindow(object):

    def __init__(self, beg, end):
        self.offset = beg
        self.duration = end - beg

    def in_region(self, val):
        return val >= self.offset and val <= (self.offset + self.duration)

    def __repr__(self):
        return f"[{self.offset}:{self.offset+self.duration}]"


def compute_guard(period, transition):
    # Collection doesn't flip the instant start()/stop() is called, so exclude a guard
    # band around each transition. The call duration is a rough proxy for machine load;
    # 8x gives the runtime headroom to apply the change, while the 2 ms floor covers
    # timestamp jitter on fast idle machines.
    if period is None or transition not in period.keys():
        return int(2e6)

    call_span = period[transition].stop - period[transition].start
    return max(8 * call_span, int(2e6))


def test_collection_period_trace(json_data, collection_period_data):
    # off_cores: genuinely-off (delay) time, shrunk by guard -- must not contain
    # sustained collection from any thread.
    # on_cores: genuinely-on (collection) time, shrunk by guard -- records expected.
    #
    # Guards are local to each boundary. A single slow start/stop call can make its
    # adjacent core untestable, but must not erase otherwise stable windows elsewhere.
    off_cores = []
    on_cores = []
    guards = []
    previous_period = None
    for period in collection_period_data:
        start_guard = compute_guard(period, "start")
        stop_guard = compute_guard(period, "stop")
        previous_stop_guard = compute_guard(previous_period, "stop")
        guards.append(
            {
                "previous_stop": previous_stop_guard,
                "start": start_guard,
                "stop": stop_guard,
            }
        )

        if "delay" in period.keys():
            beg = period.delay.start + previous_stop_guard
            end = period.delay.stop - start_guard
            if end > beg:
                off_cores.append(TimeWindow(beg, end))

        if "duration" in period.keys():
            beg = period.duration.start + start_guard
            end = period.duration.stop - stop_guard
            if end > beg:
                on_cores.append(TimeWindow(beg, end))

        previous_period = period

    assert off_cores, f"no stable off-window cores (guards={guards})"
    assert on_cores, f"no stable on-window cores (guards={guards})"

    data = json_data["rocprofiler-sdk-tool"]

    on_core_records = [0] * len(on_cores)
    off_core_records = {}
    for itr in ["hsa_api", "hip_api", "marker_api", "rccl_api"]:
        grp = data.buffer_records[itr]
        for record in grp:
            ts = record.start_timestamp

            for index, window in enumerate(off_cores):
                if window.in_region(ts):
                    key = (index, record.thread_id)
                    off_core_records.setdefault(key, []).append((itr, record))

            for index, window in enumerate(on_cores):
                if window.in_region(ts):
                    on_core_records[index] += 1

    # A thread may read the active context just before stop_context and then be
    # descheduled before recording the API call's start time. When it resumes, that one
    # in-flight call can appear in the next off window. Each thread can have only one
    # such call, so seeing another means tracing continued after the context stopped.
    for (window_index, thread_id), records in off_core_records.items():
        assert len(records) <= 1, (
            "multiple records from one thread were collected while tracing was off "
            f"(window={off_cores[window_index]}, thread_id={thread_id}, "
            f"records={records}, guards={guards})"
        )
        itr, record = records[0]
        offset = record.start_timestamp - off_cores[window_index].offset
        print(
            "tolerated in-flight record while tracing was off "
            f"(window={off_cores[window_index]}, thread_id={thread_id}, "
            f"category={itr}, offset={offset / 1e6:.3f} ms)"
        )

    # Collection must have captured data inside every active window.
    empty_on_cores = [
        window for window, count in zip(on_cores, on_core_records) if count == 0
    ]
    assert not empty_on_cores, (
        f"no records were collected inside active collection window(s) {empty_on_cores} "
        f"(counts={on_core_records}, guards={guards})"
    )


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data, json_data, ("hip", "hsa", "marker", "kernel", "memory_copy")
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
