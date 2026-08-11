#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Workload geometry, kept in sync with the execute step's `ARGS 8 8 30`
# (hip-graph-forkjoin <width> <depth> <replays>). Each replay dispatches one
# root node plus, per layer, `width` branch kernels and one join kernel.
WIDTH = 8
DEPTH = 8
REPLAYS = 30
EXPECTED_DISPATCHES = REPLAYS * (1 + DEPTH * (WIDTH + 1))


def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """Minimum expected structure is present."""
    node_exists("rocprofiler-sdk-json-tool", input_data)

    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    node_exists("metadata", sdk_data)
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])

    node_exists("agents", sdk_data)
    node_exists("buffer_records", sdk_data)
    node_exists("kernel_dispatch", sdk_data["buffer_records"])


def test_dispatch_count(input_data):
    """Every graph kernel dispatch was captured. A dropped completion (the
    failure mode of the old handler pool) would leave records missing."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    dispatches = sdk_data["buffer_records"]["kernel_dispatch"]

    assert (
        len(dispatches) == EXPECTED_DISPATCHES
    ), f"expected {EXPECTED_DISPATCHES} dispatches, got {len(dispatches)}"


def test_dispatch_records_well_formed(input_data):
    """Each captured dispatch has ordered timestamps within the session bounds
    and valid correlation ids."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    init_time = sdk_data["metadata"]["init_time"]
    fini_time = sdk_data["metadata"]["fini_time"]

    for itr in sdk_data["buffer_records"]["kernel_dispatch"]:
        assert itr["start_timestamp"] < itr["end_timestamp"], f"{itr}"
        assert itr["correlation_id"]["internal"] > 0, f"{itr}"
        assert init_time < itr["start_timestamp"], f"{itr}"
        assert fini_time > itr["end_timestamp"], f"{itr}"


def test_correlation_ids_unique(input_data):
    """Internal correlation ids are unique across captured dispatches; a
    duplicate would indicate a completion accounted for more than once."""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    cids = [
        itr["correlation_id"]["internal"]
        for itr in sdk_data["buffer_records"]["kernel_dispatch"]
    ]

    assert len(cids) == len(set(cids)), "duplicate correlation ids"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
