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


def test_perfetto_data(pftrace_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_data(
        pftrace_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy"),
    )


def test_otf2_data(otf2_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_data(
        otf2_data,
        json_data,
        ("hip", "marker", "kernel", "memory_copy", "memory_allocation"),
    )


def test_otf2_system_tree_node_data(otf2_system_tree_node_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_otf2_system_tree_node(
        otf2_system_tree_node_data,
    )


def test_csv_data(csv_data, json_data):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_csv_data(
        csv_data,
        json_data,
        (
            "agent",
            "counter_collection",
            "kernel",
            "memory_allocation",
            "memory_copy",
            "regions",
        ),
    )


def test_arg_annotations_exist(pftrace_reader):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_arg_annotations(pftrace_reader)


def test_event_id_annotations(pftrace_reader):
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_perfetto_event_id_annotations(pftrace_reader)


def test_summary_region_category_kernel(summary_kernel_dir):
    """Test that --region-categories KERNEL only produces kernel summaries."""
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_summary_region_category_filtering(
        summary_kernel_dir,
        expected_categories=["kernel"],
    )


def test_summary_region_category_hip(summary_hip_dir):
    """Test that --region-categories HIP only produces HIP summaries."""
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_summary_region_category_filtering(
        summary_hip_dir,
        expected_categories=["hip"],
    )


def test_summary_region_category_multiple(summary_multiple_dir):
    """Test that --region-categories HIP KERNEL produces those summaries."""
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_summary_region_category_filtering(
        summary_multiple_dir,
        expected_categories=["hip", "kernel"],
    )


def test_summary_region_category_none(summary_none_dir):
    """Test that --region-categories NONE includes views but no regions."""
    import rocprofiler_sdk.tests.rocprofv3 as rocprofv3

    rocprofv3.test_summary_region_category_filtering(
        summary_none_dir,
        expected_categories=["kernel", "memory"],
        allow_none=True,
    )


#########################################################################################
#
# ROCPD Summary Validation
#
#########################################################################################


def test_summary_truncate_kernels(csv_kernels_truncated, csv_kernels_full, json_data):
    """
    Test that --truncate-kernels flag only affects kernel names, not statistics.

    This test verifies:
    1. Full kernel names contain template parameters
    2. Truncated kernel names don't contain template parameters
    3. Both summaries have the same number of kernel entries
    4. All statistics (calls, duration) are preserved between truncated and full summaries
    """
    truncated_kernels = csv_kernels_truncated
    full_kernels = csv_kernels_full

    # Verify we have data
    assert len(truncated_kernels) > 0, "No kernels found in truncated summary"
    assert len(full_kernels) > 0, "No kernels found in full summary"

    # Verify we have the same number of kernel entries
    assert len(truncated_kernels) == len(
        full_kernels
    ), f"Mismatch in number of kernels: truncated={len(truncated_kernels)}, full={len(full_kernels)}"

    # Verify full kernel names have template parameters
    assert any(
        "(" in str(row["Name"]) or "<" in str(row["Name"])
        for idx, row in full_kernels.iterrows()
    ), (
        "No kernels with template parameters found. "
        "Expected full kernel names to contain template parameters like '(unsigned int)' or '<T>'"
    )

    # Verify truncated kernel names don't have template parameters
    for idx, row in truncated_kernels.iterrows():
        kernel_name = row["Name"]
        assert (
            "(" not in kernel_name and "<" not in kernel_name
        ), f"Truncated kernel name still contains template parameters: {kernel_name}"

    # Verify statistics are preserved (calls and duration)
    total_calls_truncated = truncated_kernels["Calls"].sum()
    total_calls_full = full_kernels["Calls"].sum()
    assert total_calls_truncated == total_calls_full, (
        f"Total call count mismatch: "
        f"truncated={total_calls_truncated}, full={total_calls_full}"
    )

    total_duration_truncated = truncated_kernels["Duration (Nsec)"].sum()
    total_duration_full = full_kernels["Duration (Nsec)"].sum()
    assert total_duration_truncated == total_duration_full, (
        f"Total duration mismatch: "
        f"truncated={total_duration_truncated}, full={total_duration_full}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
