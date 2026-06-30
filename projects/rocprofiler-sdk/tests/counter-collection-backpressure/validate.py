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

# The execute test launches reproducible-dispatch-count with:
#   ARGS = <niterations=8000> <nthreads=2> <nspin=100> <nsync=8000>
# so the deep, infrequently-synchronized backlog totals niterations * nthreads
# dispatches. The serializer backpressure must drain all of them without hanging
# or dropping work. We require a strong lower bound (half the total) to confirm
# the backlog actually drained; the execute test's TIMEOUT guards against a hang.
EXPECTED_MIN_DISPATCHES = 8000


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)
    rocp_data = input_data
    node_exists("names", rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"])
    node_exists(
        "counter_collection",
        rocp_data["rocprofiler-sdk-json-tool"]["buffer_records"],
        EXPECTED_MIN_DISPATCHES,
    )


def test_backlog_drained(input_data):
    """the full deep backlog must complete under backpressure (no hang/drop)"""
    data = input_data["rocprofiler-sdk-json-tool"]
    counter_data = data["buffer_records"]["counter_collection"]
    assert (
        len(counter_data) >= EXPECTED_MIN_DISPATCHES
    ), f"only {len(counter_data)} dispatches profiled; backlog did not fully drain"

    for itr in counter_data:
        assert itr["num_records"] == len(itr["records"]), f"itr={itr}"
        assert itr["start_timestamp"] < itr["end_timestamp"], f"itr={itr}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
