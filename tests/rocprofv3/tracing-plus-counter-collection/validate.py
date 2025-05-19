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

import os
import sys
import pytest


def test_validate_counter_collection_plus_tracing(
    json_data, counter_input_data, hsa_input_data
):

    # check if either kernel-name/FUNCTION is present
    assert (
        "Kernel_Name" in counter_input_data.columns
        or "Function" in counter_input_data.columns
    )

    data = json_data["rocprofiler-sdk-tool"]
    hsa_api = data["buffer_records"]["hsa_api"]
    assert len(hsa_input_data) == len(hsa_api)

    counter_collection_data = data["callback_records"]["counter_collection"]
    assert len(counter_collection_data) > 0


def test_validate_counter_collection_plus_tracing_dispatch_data(json_data):
    """
    Validates that the dispatch data from kernel tracing and counter collection match
    """

    data = json_data["rocprofiler-sdk-tool"]
    argv0 = os.path.basename(data["metadata"]["command"][0])
    kernel_ids = []
    kernel_dispatch_data = {}
    counter_collection_data = {}

    for itr in data["buffer_records"]["kernel_dispatch"]:
        kern_id = itr["dispatch_info"]["kernel_id"]
        kernel_dispatch_data[kern_id] = itr
        kernel_dispatch_data[kern_id]["stream_id"] = itr["stream_id"]["handle"]
        if kern_id not in kernel_ids:
            kernel_ids += [kern_id]

    for itr in data["callback_records"]["counter_collection"]:
        kern_id = itr["dispatch_data"]["dispatch_info"]["kernel_id"]
        counter_collection_data[kern_id] = itr["dispatch_data"]
        counter_collection_data[kern_id]["thread_id"] = itr["thread_id"]
        counter_collection_data[kern_id]["stream_id"] = itr["stream_id"]["handle"]
        if kern_id not in kernel_ids:
            kernel_ids += [kern_id]

    kernel_ids = sorted(kernel_ids)

    for kern_id in kernel_ids:
        for itr in ["thread_id", "stream_id", "start_timestamp", "end_timestamp"]:
            assert (
                counter_collection_data[kern_id][itr]
                == kernel_dispatch_data[kern_id][itr]
            ), f"kernel_id={kern_id}, check={itr}"
            if itr == "stream_id":
                if argv0 == "simple-transpose":
                    assert (
                        kernel_dispatch_data[kern_id][itr] == 0
                    ), f"kernel_id={kern_id}, check={itr}"
                elif argv0 == "transpose":
                    assert (
                        kernel_dispatch_data[kern_id][itr] > 0
                    ), f"kernel_id={kern_id}, check={itr}"
                else:
                    assert False, f"no stream_id handling for '{argv0}'"
            else:
                assert (
                    kernel_dispatch_data[kern_id][itr] > 0
                ), f"kernel_id={kern_id}, check={itr}"

        assert (
            counter_collection_data[kern_id]["correlation_id"]["internal"]
            == kernel_dispatch_data[kern_id]["correlation_id"]["internal"]
        )
        assert (
            counter_collection_data[kern_id]["correlation_id"]["external"]
            == kernel_dispatch_data[kern_id]["correlation_id"]["external"]
        )


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("hip", "hsa", "marker", "kernel", "memory_copy", "counter_collection"),
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
