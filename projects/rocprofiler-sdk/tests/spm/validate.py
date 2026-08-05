#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
from collections import defaultdict


# helper function
def node_exists(name, data, min_len=1):
    assert name in data
    assert data[name] is not None
    assert len(data[name]) >= min_len


def is_buffered(input_data):
    data = input_data["rocprofiler-sdk-json-tool"]
    buffer_records = data.get("buffer_records", {})
    return len(buffer_records.get("spm_counter_collection", [])) > 0


def test_data_structure(input_data):
    """verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)
    data = input_data["rocprofiler-sdk-json-tool"]
    if is_buffered(input_data):
        node_exists("buffer_records", data)
        node_exists("spm_counter_collection", data["buffer_records"])
    else:
        node_exists("names", data["callback_records"])
        node_exists("spm_records", data["callback_records"])


def test_spm_counter_values(input_data):
    data = input_data["rocprofiler-sdk-json-tool"]

    if is_buffered(input_data):
        _validate_buffered_counter_values(data)
    else:
        _validate_callback_counter_values(data)


def _get_counter_value(counters, name):
    for itr in counters:
        if itr["name"] == name:
            return itr["value"]


def _validate_dispatch_counters(dispatch_id, counters):
    assert float(_get_counter_value(counters, "TA_TA_BUSY")) > _get_counter_value(
        counters, "TA_TOTAL_WAVEFRONTS"
    )

    assert (
        100
        * _get_counter_value(counters, "SQC_ICACHE_MISSES")
        / _get_counter_value(counters, "SQC_ICACHE_REQ")
    ) < 100
    assert (
        100
        * _get_counter_value(counters, "SQC_ICACHE_HITS")
        / _get_counter_value(counters, "SQC_ICACHE_REQ")
    ) < 100


def _accumulate_counter(counter_map, name, value):
    if name in counter_map:
        counter_map[name]["value"] += value
    else:
        counter_map[name] = {"name": name, "value": value}


def _validate_buffered_counter_values(data):
    counter_info = data["counter_info"]
    spm_data = data["buffer_records"]["spm_counter_collection"]
    assert len(spm_data) > 0, "No buffered SPM dispatch records found"

    def get_name(counter_id):
        for itr in counter_info:
            if itr["id"]["handle"] == counter_id:
                return itr["name"]

    headers = {}
    all_records = []
    for dispatch_group in spm_data:
        assert "dispatch_info" in dispatch_group
        assert "records" in dispatch_group

        dispatch_info = dispatch_group["dispatch_info"]
        assert dispatch_info["agent_id"]["handle"] > 0
        assert dispatch_info["dispatch_id"] > 0

        headers[dispatch_info["dispatch_id"]] = dispatch_info
        all_records.extend(dispatch_group["records"])

    assert len(all_records) > 0, "No SPM counter records found in buffered output"

    dispatch_counter_map = defaultdict(dict)
    for record in all_records:
        assert "counter_id" in record
        assert "dispatch_id" in record
        assert (
            record["dispatch_id"] in headers
        ), f"Record dispatch_id {record['dispatch_id']} has no matching header"

        dispatch_id = record["dispatch_id"]
        counter_name = get_name(record["counter_id"]["handle"])
        _accumulate_counter(
            dispatch_counter_map[dispatch_id], counter_name, record["value"]
        )

    for dispatch_id, counter_map in dispatch_counter_map.items():
        _validate_dispatch_counters(dispatch_id, list(counter_map.values()))


def _validate_callback_counter_values(data):
    counter_info = data["counter_info"]
    counter_data = data["callback_records"]["spm_records"]

    def get_name(counter_id):
        for itr in counter_info:
            if itr["id"]["handle"] == counter_id:
                return itr["name"]

    dispatch_counter_map = defaultdict(dict)
    for record in counter_data:
        dispatch_id = record["dispatch_id"]
        counter_name = get_name(record["counter_id"]["handle"])
        _accumulate_counter(
            dispatch_counter_map[dispatch_id], counter_name, record["value"]
        )

    for dispatch_id, counter_map in dispatch_counter_map.items():
        _validate_dispatch_counters(dispatch_id, list(counter_map.values()))


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
