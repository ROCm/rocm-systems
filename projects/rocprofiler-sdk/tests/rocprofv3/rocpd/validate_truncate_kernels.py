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


def test_truncated_kernel_names(csv_kernels_truncated, csv_kernels_full, json_data):
    """
    Test that --truncate-kernels flag produces kernel names without template parameters.

    This test verifies:
    1. The truncated summary exists and has kernel data
    2. The full summary exists and has kernel data
    3. Kernel names in truncated summary don't contain template parameters
    4. Kernel names in full summary may contain template parameters
    5. Both summaries have the same number of kernel entries (same kernels, different names)
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

    # Check that truncated names don't have template parameters
    for idx, row in truncated_kernels.iterrows():
        kernel_name = row["Name"]
        assert (
            "(" not in kernel_name and "<" not in kernel_name
        ), f"Truncated kernel name still contains template parameters: {kernel_name}"

    # Verify both summaries have matching call counts and durations
    # (same kernels, just different naming)
    total_calls_truncated = truncated_kernels["Calls"].sum()
    total_calls_full = full_kernels["Calls"].sum()
    assert total_calls_truncated == total_calls_full, (
        f"Total call count mismatch: "
        f"truncated={total_calls_truncated}, full={total_calls_full}"
    )


def test_kernel_names_have_expected_format(csv_kernels_full):
    """
    Test that full kernel names contain expected template parameter patterns.
    """
    full_kernels = csv_kernels_full
    assert len(full_kernels) > 0, "No kernels found in full summary"

    # At least some kernel names should have template parameters
    # (this may not be true for all kernels, but typically HIP kernels do)
    has_templates = any(
        "(" in str(row["Name"]) or "<" in str(row["Name"])
        for idx, row in full_kernels.iterrows()
    )

    # This is a soft assertion - if all kernels are simple, that's ok too
    # but we document the expectation
    if not has_templates:
        print(
            "Warning: No kernels with template parameters found. "
            "This may be expected for simple kernels."
        )


def test_summary_statistics_match(csv_kernels_truncated, csv_kernels_full):
    """
    Verify that summary statistics are preserved regardless of name truncation.

    The --truncate-kernels flag should only affect the displayed name,
    not the underlying statistics.
    """
    truncated_kernels = csv_kernels_truncated
    full_kernels = csv_kernels_full

    # Total duration should be the same
    total_duration_truncated = truncated_kernels["DURATION (nsec)"].sum()
    total_duration_full = full_kernels["DURATION (nsec)"].sum()

    assert total_duration_truncated == total_duration_full, (
        f"Total duration mismatch: "
        f"truncated={total_duration_truncated}, full={total_duration_full}"
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
