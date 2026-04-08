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


def node_exists(name, data, min_len=1):
    """Helper function to verify expected data structure"""
    assert name in data
    assert data[name] is not None
    if isinstance(data[name], (list, tuple, dict, set)):
        assert len(data[name]) >= min_len


def test_data_structure(input_data):
    """Verify minimum amount of expected data is present"""
    node_exists("rocprofiler-sdk-json-tool", input_data)

    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    node_exists("metadata", sdk_data)
    node_exists("pid", sdk_data["metadata"])
    node_exists("main_tid", sdk_data["metadata"])
    node_exists("init_time", sdk_data["metadata"])
    node_exists("fini_time", sdk_data["metadata"])

    node_exists("callback_records", sdk_data)
    node_exists("buffer_records", sdk_data)
    node_exists("kernel_dispatch", sdk_data["callback_records"])
    node_exists("kernel_dispatch", sdk_data["buffer_records"])


def test_kernel_dispatch_count(input_data, expected_kernels, expected_iterations):
    """Verify correct number of kernel dispatches"""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    expected_dispatch_count = expected_kernels * expected_iterations

    # Check buffer records (this is what we use for bubble detection)
    buf_dispatches = sdk_data["buffer_records"]["kernel_dispatch"]
    assert len(buf_dispatches) == expected_dispatch_count, (
        f"Expected {expected_dispatch_count} buffer records, "
        f"got {len(buf_dispatches)}"
    )


def test_kernel_timestamps(input_data):
    """Verify timestamps are valid"""
    sdk_data = input_data["rocprofiler-sdk-json-tool"]

    for idx, dispatch in enumerate(sdk_data["buffer_records"]["kernel_dispatch"]):
        assert (
            dispatch["start_timestamp"] < dispatch["end_timestamp"]
        ), f"Dispatch {idx}: start_timestamp must be < end_timestamp"

        assert (
            dispatch["correlation_id"]["internal"] > 0
        ), f"Dispatch {idx}: internal correlation_id must be > 0"

        assert (
            dispatch["correlation_id"]["external"] > 0
        ), f"Dispatch {idx}: external correlation_id must be > 0"


def test_no_bubbles(input_data, expected_kernels, expected_iterations):
    """
    Validate that there are no excessive gaps ("bubbles") between consecutive
    kernel dispatches within a single hipGraphLaunch execution.

    NOTE: This test cannot reliably group by hipGraphLaunch iterations using
    the json-tool because external correlation IDs are thread IDs, not unique
    per graph launch. This test is primarily for rocprofv3 which uses --hip-trace
    to get proper correlation IDs. For json-tool, we skip iteration grouping
    and just verify kernels executed.
    """
    import pytest

    sdk_data = input_data["rocprofiler-sdk-json-tool"]
    dispatches = sdk_data["buffer_records"]["kernel_dispatch"]

    assert len(dispatches) > 0, (
        "No kernel dispatches found in buffer records. "
        "Check that the test binary is executing kernels."
    )

    expected_dispatch_count = expected_kernels * expected_iterations
    assert len(dispatches) == expected_dispatch_count, (
        f"Expected {expected_dispatch_count} total dispatches, " f"got {len(dispatches)}"
    )

    # json-tool doesn't provide unique correlation IDs per hipGraphLaunch,
    # so we can't do bubble detection. Skip the rest of this test.
    pytest.skip(
        "Bubble detection requires unique correlation IDs per graph launch (use rocprofv3 test)"
    )

    # For each graph launch, check for bubbles within that execution
    all_gaps = []

    for corr_id, kernel_list in iterations.items():
        # Sort by start timestamp within this graph launch
        sorted_kernels = sorted(kernel_list, key=lambda k: k["start_ns"])

        # Verify correct number of kernels per graph launch
        assert len(sorted_kernels) == expected_kernels, (
            f"External correlation ID {corr_id}: expected {expected_kernels} "
            f"dispatches per graph launch, got {len(sorted_kernels)}"
        )

        # Calculate gaps between consecutive dispatches within this graph launch
        for i in range(len(sorted_kernels) - 1):
            curr_end = sorted_kernels[i]["end_ns"]
            next_start = sorted_kernels[i + 1]["start_ns"]
            gap = next_start - curr_end
            all_gaps.append(gap)

    # Analyze gaps
    if not all_gaps:
        # No gaps means each iteration had only 1 kernel, which is fine
        return

    all_gaps_sorted = sorted(all_gaps)
    max_gap = max(all_gaps)
    p99_9_gap = all_gaps_sorted[int(len(all_gaps) * 0.999)]

    # Check for batching patterns (most gaps are small, but some are very large)
    # This detects scenarios where kernels are grouped into batches with large
    # gaps between batches, indicating a profiler regression
    OUTLIER_THRESHOLD_NS = 50_000  # 50 microseconds

    if p99_9_gap > OUTLIER_THRESHOLD_NS:
        raise AssertionError(
            f"Bubble detected: batching pattern with large inter-batch gaps. "
            f"99.9th percentile gap is {p99_9_gap}ns ({p99_9_gap/1000:.2f}µs), "
            f"which exceeds the threshold of {OUTLIER_THRESHOLD_NS}ns ({OUTLIER_THRESHOLD_NS/1000}µs). "
            f"Max gap: {max_gap}ns ({max_gap/1000:.2f}µs). "
            f"Total gaps analyzed: {len(all_gaps)}. "
            f"This indicates kernels are being dispatched in batches rather than continuously."
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
