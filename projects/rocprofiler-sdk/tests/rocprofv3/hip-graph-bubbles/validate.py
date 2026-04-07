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


def test_kernel_trace_row_count(
    kernel_input_data,
    expected_dispatch_count,
    expected_kernels,
    expected_iterations,
):
    assert expected_dispatch_count == expected_kernels * expected_iterations
    assert expected_dispatch_count > 0, (
        "expected_dispatch_count must be positive — "
        "a test that expects zero dispatches validates nothing"
    )
    assert len(kernel_input_data) == expected_dispatch_count


def test_kernel_trace_dispatch_ids(kernel_input_data, expected_dispatch_count):
    dispatch_ids = []
    for idx, row in enumerate(kernel_input_data):
        assert row["Kind"] == "KERNEL_DISPATCH", f"Row {idx}: expected Kind=KERNEL_DISPATCH, got '{row['Kind']}'"
        try:
            dispatch_ids.append(int(row["Dispatch_Id"]))
        except ValueError as e:
            raise ValueError(f"Row {idx}: failed to parse Dispatch_Id='{row['Dispatch_Id']}'") from e

    assert len(dispatch_ids) == expected_dispatch_count
    assert len(set(dispatch_ids)) == expected_dispatch_count


def test_kernel_trace_fields(kernel_input_data, expected_dispatch_count):
    assert len(kernel_input_data) == expected_dispatch_count

    for idx, row in enumerate(kernel_input_data):
        assert row["Kind"] == "KERNEL_DISPATCH", f"Row {idx}: expected Kind=KERNEL_DISPATCH, got '{row['Kind']}'"
        try:
            assert int(row["Agent_Id"].split(" ")[-1]) >= 0, f"Row {idx}: Agent_Id must be >= 0"
            assert int(row["Queue_Id"]) > 0, f"Row {idx}: Queue_Id must be > 0"
            assert int(row["Kernel_Id"]) > 0, f"Row {idx}: Kernel_Id must be > 0"
            assert int(row["Correlation_Id"]) > 0, f"Row {idx}: Correlation_Id must be > 0"
            assert int(row["Workgroup_Size_X"]) == 256, f"Row {idx}: expected Workgroup_Size_X=256, got {row['Workgroup_Size_X']}"
            assert int(row["Workgroup_Size_Y"]) == 1, f"Row {idx}: expected Workgroup_Size_Y=1, got {row['Workgroup_Size_Y']}"
            assert int(row["Workgroup_Size_Z"]) == 1, f"Row {idx}: expected Workgroup_Size_Z=1, got {row['Workgroup_Size_Z']}"
            assert int(row["Grid_Size_X"]) == 256, f"Row {idx}: expected Grid_Size_X=256, got {row['Grid_Size_X']}"
            assert int(row["Grid_Size_Y"]) == 1, f"Row {idx}: expected Grid_Size_Y=1, got {row['Grid_Size_Y']}"
            assert int(row["Grid_Size_Z"]) == 1, f"Row {idx}: expected Grid_Size_Z=1, got {row['Grid_Size_Z']}"
            assert int(row["End_Timestamp"]) >= int(row["Start_Timestamp"]), f"Row {idx}: End_Timestamp must be >= Start_Timestamp"
        except ValueError as e:
            raise ValueError(f"Row {idx}: failed to parse integer field in {row}") from e


def test_kernel_trace_no_bubbles(
    kernel_input_data,
    expected_kernels,
    expected_iterations,
):
    """
    Validate that there are no excessive gaps ("bubbles") between consecutive
    kernel dispatches within a single hipGraphLaunch execution.

    This test groups dispatches by Correlation_Id (each unique ID represents
    one hipGraphLaunch call), then checks that kernels within the same graph
    execution launch back-to-back without large scheduling gaps.
    """
    from collections import defaultdict

    # Filter to only simpleKernel dispatches (exclude BLIT kernels, etc.)
    simple_kernel_data = [
        row for row in kernel_input_data
        if "simpleKernel" in row.get("Kernel_Name", "")
    ]

    assert len(simple_kernel_data) > 0, (
        "No simpleKernel dispatches found in kernel trace data. "
        "Check that the test binary is executing the expected kernels."
    )

    # Group dispatches by Correlation_Id (each ID = one hipGraphLaunch)
    iterations = defaultdict(list)

    for row in simple_kernel_data:
        try:
            corr_id = int(row["Correlation_Id"])
            iterations[corr_id].append(row)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse Correlation_Id from row: {row}") from e

    # Verify we have the expected number of graph launches
    assert len(iterations) == expected_iterations, (
        f"Expected {expected_iterations} unique graph launches (correlation IDs), "
        f"but found {len(iterations)}. "
        f"Correlation IDs: {sorted(iterations.keys())}"
    )

    # For each graph launch, check for bubbles within that execution
    all_gaps = []

    for corr_id, dispatches in iterations.items():
        # Sort by start timestamp within this graph launch
        sorted_dispatches = sorted(dispatches, key=lambda r: int(r["Start_Timestamp"]))

        # Verify correct number of kernels per graph launch
        assert len(sorted_dispatches) == expected_kernels, (
            f"Correlation ID {corr_id}: expected {expected_kernels} simpleKernel "
            f"dispatches per graph launch, got {len(sorted_dispatches)}"
        )

        # Calculate gaps between consecutive dispatches within this graph launch
        for i in range(len(sorted_dispatches) - 1):
            try:
                curr_end = int(sorted_dispatches[i]["End_Timestamp"])
                next_start = int(sorted_dispatches[i+1]["Start_Timestamp"])
                gap = next_start - curr_end
                all_gaps.append(gap)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Correlation ID {corr_id}, dispatch pair {i},{i+1}: "
                    f"failed to parse timestamps from {sorted_dispatches[i]} and {sorted_dispatches[i+1]}"
                ) from e

    # Analyze gaps
    if not all_gaps:
        # No gaps means each iteration had only 1 kernel, which is fine
        return

    max_gap = max(all_gaps)
    p95_gap = sorted(all_gaps)[int(len(all_gaps) * 0.95)]

    # Conservative threshold: 500µs to minimize spurious failures (Type I errors)
    # This can be tuned based on empirical results
    BUBBLE_THRESHOLD_NS = 500_000  # 500 microseconds

    assert p95_gap < BUBBLE_THRESHOLD_NS, (
        f"Bubble detected: 95th percentile gap between consecutive kernel dispatches "
        f"within the same graph launch is {p95_gap}ns ({p95_gap/1000:.1f}µs), "
        f"which exceeds the threshold of {BUBBLE_THRESHOLD_NS}ns ({BUBBLE_THRESHOLD_NS/1000}µs). "
        f"Max gap: {max_gap}ns ({max_gap/1000:.1f}µs). "
        f"Total gaps analyzed: {len(all_gaps)}. "
        f"This indicates the profiler may be introducing scheduling stalls."
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
