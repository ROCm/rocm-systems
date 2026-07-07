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


def compute_guard(collection_period_data):
    # Collection switches between off windows (delay) and on windows (duration). Right
    # at each switch, a record's timestamp could belong to either window, because
    # collection doesn't flip the instant start()/stop() is called -- the runtime applies
    # it a little later, and by a varying amount. So the assertions ignore a guard band
    # (in ns) on both sides of every switch, and this returns its width.
    #
    # The width is loose on purpose: windows are seconds long but the flip latency is
    # sub-millisecond, so anything from a few ms to tens of ms hides the ambiguity yet
    # still leaves nearly the whole window testable. It scales with the start()/stop()
    # call duration (stop minus start), a rough proxy for machine load, since a slower
    # call means a slower flip; 8x is empirically enough headroom. The 2 ms floor covers
    # fast idle machines, where 8x a tiny call duration would undershoot timestamp jitter.
    call_spans = []
    for period in collection_period_data:
        for key in ("start", "stop"):
            if key in period.keys():
                call_spans.append(period[key].stop - period[key].start)

    return max([8 * span for span in call_spans] + [int(2e6)])


def test_collection_period_trace(json_data, collection_period_data):
    guard = compute_guard(collection_period_data)

    # off_cores: genuinely-off (delay) time, shrunk by guard -- must contain no records.
    # on_cores:  genuinely-on (collection) time, shrunk by guard -- records expected.
    off_cores = []
    on_cores = []
    for period in collection_period_data:
        if "delay" in period.keys():
            beg = period.delay.start + guard
            end = period.delay.stop - guard
            if end > beg:
                off_cores.append(TimeWindow(beg, end))

        if "duration" in period.keys():
            beg = period.duration.start + guard
            end = period.duration.stop - guard
            if end > beg:
                on_cores.append(TimeWindow(beg, end))

    data = json_data["rocprofiler-sdk-tool"]

    on_core_records = 0
    for itr in ["hsa_api", "hip_api", "marker_api", "rccl_api"]:
        grp = data.buffer_records[itr]
        for record in grp:
            ts = record.start_timestamp

            off_hit = [w for w in off_cores if w.in_region(ts)]
            assert len(off_hit) == 0, (
                f"\nrecord collected while tracing was off:\n\t{record}\n"
                f"found in off-window(s):\n{off_hit}\nguard={guard} ns"
            )

            if any(w.in_region(ts) for w in on_cores):
                on_core_records += 1

    # Sanity check: collection must have actually captured data inside the active
    # windows (guards against the feature silently collecting nothing).
    assert on_core_records > 0, (
        "no records were collected inside any active collection window "
        f"(on_cores={on_cores}, guard={guard} ns)"
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
