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


#########################################################################################
#
# ROCPD Summary Validation
#
#########################################################################################


def _validate_summary_region_category_filtering(
    csv_files, expected_categories=None, allow_none=False
):
    """
    Test that summary output contains ONLY the expected categories.

    Args:
        csv_files: List of CSV file paths to validate
        expected_categories: List of category names that should be present (e.g., ['kernel', 'hip'])
        allow_none: If True, allows no region summaries (for --region-categories NONE test)
    """
    import os

    if not csv_files:
        raise ValueError("No CSV files provided for validation")

    basenames = [os.path.basename(f) for f in csv_files]

    assert len(basenames) > 0, "No summary files provided for validation"

    print(f"\nValidating {len(basenames)} summary files:")
    for name in sorted(basenames):
        print(f"  - {name}")

    # For NONE test: ensure no region-based summaries are generated
    if allow_none:
        # Region summaries have filenames like "rocm_hip_*.csv", "rocm_hsa_*.csv"
        region_files = [f for f in basenames if f.lower().startswith("rocm_")]
        assert len(region_files) == 0, (
            f"--region-categories NONE should not generate region summaries, "
            f"but found: {region_files}"
        )

    # Check that expected categories are present and ONLY those categories exist
    if expected_categories:
        # 1. Check all expected categories are present
        for category in expected_categories:
            category_lower = category.lower()
            matching_files = [f for f in basenames if category_lower in f.lower()]
            assert len(matching_files) > 0, (
                f"Expected category '{category}' not found. "
                f"No files matching '{category_lower}' in {basenames}"
            )

        # 2. Check no unexpected categories exist
        for filename in basenames:
            filename_lower = filename.lower()
            matches_expected = any(
                cat.lower() in filename_lower for cat in expected_categories
            )
            assert matches_expected, (
                f"Unexpected file '{filename}' found. "
                f"Does not match any expected category: {expected_categories}"
            )


def test_summary_region_category_kernel(summary_kernel_csv_files):
    """Test that --region-categories KERNEL only produces kernel summaries."""
    _validate_summary_region_category_filtering(
        summary_kernel_csv_files,
        expected_categories=["kernel"],
    )


def test_summary_region_category_hip(summary_hip_csv_files):
    """Test that --region-categories HIP only produces HIP summaries."""
    _validate_summary_region_category_filtering(
        summary_hip_csv_files,
        expected_categories=["hip"],
    )


def test_summary_region_category_multiple(summary_multiple_csv_files):
    """Test that --region-categories HIP KERNEL produces those summaries."""
    _validate_summary_region_category_filtering(
        summary_multiple_csv_files,
        expected_categories=["hip", "kernel"],
    )


def test_summary_region_category_none(summary_none_csv_files):
    """Test that --region-categories NONE includes views but no regions."""
    _validate_summary_region_category_filtering(
        summary_none_csv_files,
        expected_categories=["kernel", "memory"],
        allow_none=True,
    )


def _kernel_names(df):
    """Extract kernel names from dataframe as list of strings."""
    return [str(name) for name in df["Name"].tolist()]


def _looks_demangled(name: str) -> bool:
    """Check if kernel name appears to be demangled (has C++ syntax)."""
    return "(" in name or "<" in name


def _looks_mangled_cpp(name: str) -> bool:
    """Check if kernel name appears to be C++ mangled (starts with _Z)."""
    return name.startswith("_Z")


def _assert_demangled_names(df):
    """Verify that at least one kernel name is demangled."""
    names = _kernel_names(df)

    assert any(
        _looks_demangled(name) for name in names
    ), "Expected at least one demangled kernel name, but none contained '(' or '<'."


def _assert_truncated_names(df):
    """Verify that all kernel names are truncated."""
    names = _kernel_names(df)

    invalid_names = [name for name in names if _looks_demangled(name)]
    assert not invalid_names, (
        "Expected all kernel names to be truncated, "
        f"but found non-truncated name: {invalid_names[0]}"
    )


def _assert_mangled_names(df):
    """Verify that kernel names remain mangled/raw."""
    names = _kernel_names(df)

    demangled_names = [name for name in names if _looks_demangled(name)]
    assert not demangled_names, (
        "Expected no demangled kernel names when mangled names are requested, "
        f"but found: {demangled_names[0]}"
    )

    mangled_names = [name for name in names if _looks_mangled_cpp(name)]
    assert mangled_names, (
        "Expected at least one C++ mangled kernel name starting with '_Z', "
        "but none were found."
    )


def _assert_statistics_match(df1, df2, name1, name2):
    """
    Helper function to verify that statistics match between two kernel summaries.

    Args:
        df1: First dataframe
        df2: Second dataframe (reference)
        name1: Label for first dataframe (e.g., "truncated")
        name2: Label for second dataframe (e.g., "full")
    """
    # Verify we have data
    assert len(df1) > 0, f"No kernels found in {name1} summary"
    assert len(df2) > 0, f"No kernels found in {name2} summary"

    # Verify same number of entries
    assert len(df1) == len(
        df2
    ), f"Mismatch in number of kernels: {name1}={len(df1)}, {name2}={len(df2)}"

    # Verify statistics are preserved
    total_calls_1 = df1["Calls"].sum()
    total_calls_2 = df2["Calls"].sum()
    assert (
        total_calls_1 == total_calls_2
    ), f"Total call count mismatch: {name1}={total_calls_1}, {name2}={total_calls_2}"

    total_duration_1 = df1["Duration (Nsec)"].sum()
    total_duration_2 = df2["Duration (Nsec)"].sum()
    assert (
        total_duration_1 == total_duration_2
    ), f"Total duration mismatch: {name1}={total_duration_1}, {name2}={total_duration_2}"


def test_summary_truncate_kernels(csv_kernels_truncated, csv_kernels_full):
    """
    Test that --truncate-kernels flag only affects kernel names, not statistics.

    This test verifies:
    1. Full kernel names and truncated kernel names are valid
    2. Statistics (calls, duration) are preserved between truncated and full summaries
    """

    _assert_demangled_names(csv_kernels_full)
    _assert_truncated_names(csv_kernels_truncated)
    _assert_statistics_match(csv_kernels_truncated, csv_kernels_full, "truncated", "full")


def test_summary_mangled_kernels(csv_kernels_mangled, csv_kernels_full):
    """
    Test that --mangled-kernels flag only affects kernel names, not statistics.

    This test verifies:
    1. Full kernel names and mangled kernel names are valid
    2. Statistics (calls, duration) are preserved between mangled and demangled summaries
    """

    _assert_demangled_names(csv_kernels_full)
    _assert_mangled_names(csv_kernels_mangled)
    _assert_statistics_match(csv_kernels_mangled, csv_kernels_full, "mangled", "full")


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
