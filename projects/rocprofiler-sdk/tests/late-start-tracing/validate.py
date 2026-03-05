#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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


def test_data_structure(input_data):
    """Verify the JSON output has the expected structure"""
    data = input_data

    assert "late-start-tracing" in data
    assert data["late-start-tracing"] is not None

    sdk_data = data["late-start-tracing"]

    assert "traces" in sdk_data
    assert sdk_data["traces"] is not None
    assert isinstance(sdk_data["traces"], list)


def test_pre_init_not_traced(input_data):
    """Verify that pre-init HIP calls were NOT traced

    The test first calls:
    - hipGetDeviceCount
    - hipSetDevice
    - hipGetDeviceProperties

    These should NOT appear in the traces because they happened before init() was called.
    """
    data = input_data
    traces = data["late-start-tracing"]["traces"]

    # Count pre-init operations that should NOT be traced
    pre_init_operations = [
        "hipGetDeviceCount",
        "hipSetDevice",
        "hipGetDeviceProperties",
    ]

    # Also check for pre-init ROCTx ranges
    pre_init_markers = ["pre-init-phase"]

    traced_pre_init_ops = []
    traced_pre_init_markers = []

    for trace in traces:
        name = trace.get("name", "")

        # Check if any pre-init HIP operations were traced
        for op in pre_init_operations:
            if op in name:
                traced_pre_init_ops.append(name)

        # Check if pre-init ROCTx markers were traced
        for marker in pre_init_markers:
            if marker in name:
                traced_pre_init_markers.append(name)

    # These operations should NOT be in the traces
    assert (
        len(traced_pre_init_ops) == 0
    ), f"Pre-init HIP operations should not be traced, but found: {traced_pre_init_ops}"

    assert (
        len(traced_pre_init_markers) == 0
    ), f"Pre-init ROCTx markers should not be traced, but found: {traced_pre_init_markers}"


def test_post_init_traced(input_data):
    """Verify that post-init HIP calls WERE traced

    The test calls these after init():
    - hipMalloc
    - hipMemset
    - hipDeviceSynchronize
    - hipFree
    - hipGetLastError

    These SHOULD all appear in the traces.
    """
    data = input_data
    traces = data["late-start-tracing"]["traces"]

    # Post-init operations that SHOULD be traced
    post_init_operations = [
        "hipMalloc",
        "hipMemset",
        "hipDeviceSynchronize",
        "hipFree",
        "hipGetLastError",
    ]

    found_operations = set()
    found_roctx_calls = 0

    for trace in traces:
        name = trace.get("name", "")

        # Check for post-init HIP operations
        for op in post_init_operations:
            if op in name:
                found_operations.add(op)

        # Check for ROCTx API calls (roctxRangeStart/Stop)
        if "roctx" in name.lower():
            found_roctx_calls += 1

    # Verify all post-init operations were traced
    for op in post_init_operations:
        assert op in found_operations, (
            f"Post-init operation '{op}' should be traced but was not found. "
            f"Found operations: {found_operations}"
        )

    # Verify ROCTx calls were traced (at least the roctxRangeStart/Stop calls)
    assert found_roctx_calls >= 4, (
        f"Expected at least 4 ROCTx API calls (2 ranges with start/stop), "
        f"but found {found_roctx_calls}"
    )


def test_minimum_traces(input_data):
    """Verify we got a reasonable number of traces"""
    data = input_data
    traces = data["late-start-tracing"]["traces"]

    # We should have at least some traces from post-init operations
    # Minimum: 5 HIP API calls (hipMalloc, hipMemset, hipDeviceSynchronize, hipFree, hipGetLastError)
    # Plus their enter/exit phases = 10 traces minimum
    # Plus ROCTx ranges (start/stop for 2 ranges) = 4 more traces
    # Total minimum: 14 traces
    assert len(traces) >= 14, (
        f"Expected at least 14 traces (5 HIP APIs x 2 phases + 2 ROCTx ranges x 2 phases), "
        f"but got {len(traces)}"
    )


def test_timestamps_valid(input_data):
    """Verify all timestamps are valid (callback tracing may not provide timestamps)"""
    data = input_data
    traces = data["late-start-tracing"]["traces"]

    for trace in traces:
        assert "timestamp" in trace
        assert isinstance(trace["timestamp"], int)
        # Callback tracing doesn't provide timestamps, so 0 is acceptable
        assert trace["timestamp"] >= 0


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
