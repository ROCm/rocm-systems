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


def test_counter_collection_with_att(json_data):
    data = json_data["rocprofiler-sdk-tool"]
    assert data["strings"]["att_filenames"]
    counter_names = {
        counter["id"]["handle"]: counter["name"] for counter in data["counters"]
    }
    callbacks = data["callback_records"]["counter_collection"]
    assert callbacks

    found_positive_value = False
    for entry in callbacks:
        dispatch = entry["dispatch_data"]
        assert dispatch["dispatch_info"]["dispatch_id"] > 0
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]
        assert entry["records"]
        for record in entry["records"]:
            assert counter_names[record["counter_id"]["handle"]] == "SQ_WAVES"
            assert record["value"] >= 0
            found_positive_value = found_positive_value or record["value"] > 0
    assert found_positive_value


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-x", __file__] + sys.argv[1:]))
