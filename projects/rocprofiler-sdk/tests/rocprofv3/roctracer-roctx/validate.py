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

import sys
import pytest


def test_marker_api_trace(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kind_name(kind_id):
        return data["strings"]["buffer_records"][kind_id]["kind"]

    def get_region_name(corr_id):
        for itr in data["strings"]["marker_api"]:
            if itr.key == corr_id:
                return itr.value
        return None

    valid_domain = (
        "MARKER_CORE_API",
        "MARKER_CONTROL_API",
        "MARKER_NAME_API",
        "MARKER_CORE_RANGE_API",
    )

    buffer_records = data["buffer_records"]
    marker_data = buffer_records["marker_api"]
    tot_data = {}
    thr_data = {}
    for marker in marker_data:
        assert get_kind_name(marker["kind"]) in valid_domain
        assert marker.thread_id >= data["metadata"]["pid"]
        assert marker.end_timestamp >= marker.start_timestamp

        if marker.thread_id not in thr_data.keys():
            thr_data[marker.thread_id] = {}

        corr_id = marker.correlation_id.internal
        assert corr_id > 0, f"{marker}"
        name = get_region_name(corr_id)
        if not name.startswith("roctracer/roctx"):
            assert "run" in name, f"{marker}"
            if name not in thr_data[marker.thread_id].keys():
                thr_data[marker.thread_id][name] = 1
            else:
                thr_data[marker.thread_id][name] += 1

        if name not in tot_data.keys():
            tot_data[name] = 1
        else:
            tot_data[name] += 1

    assert tot_data["roctracer/roctx v4.1"] == 1
    assert tot_data["run"] == 2
    assert tot_data["run/iteration"] == 1000
    assert tot_data["run/iteration/sync"] == 100
    assert tot_data["run/rank-0/thread-0/device-0/begin"] == 1
    assert tot_data["run/rank-0/thread-0/device-0/end"] == 1
    assert len(tot_data.keys()) >= 8

    for tid, titr in thr_data.items():
        assert titr["run"] == 1
        assert titr["run/iteration"] == 500
        assert titr["run/iteration/sync"] == 50
        assert len(titr.keys()) >= 5


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(pftrace_data, json_data, ("memory_copy", "marker"))


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(otf2_data, json_data, ("memory_copy", "marker"))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
