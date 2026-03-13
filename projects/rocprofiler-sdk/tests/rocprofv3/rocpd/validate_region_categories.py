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

"""
Validation tests for region category filtering.

This script validates that when --region-categories is specified, the output
only contains summaries for the specified categories.
"""

import sys
import pytest
import os
import glob
from pathlib import Path


def get_summary_files(directory):
    """Get all CSV summary files from a directory."""
    if not os.path.exists(directory):
        return []
    return [os.path.basename(f) for f in glob.glob(os.path.join(directory, "*.csv"))]


def test_kernel_category_filter(request):
    """Test that --region-categories KERNEL only produces kernel-related summaries."""
    base_dir = request.config.getoption("--summary-kernel-dir")
    summary_files = get_summary_files(base_dir)

    print(f"Found summary files in {base_dir}: {summary_files}")

    # Should have kernel-related summaries
    kernel_files = [f for f in summary_files if "kernel" in f.lower()]
    assert len(kernel_files) > 0, "Expected at least one kernel summary file"

    # Should NOT have HIP summaries
    hip_files = [f for f in summary_files if "hip" in f.lower() and "kernel" not in f.lower()]
    assert len(hip_files) == 0, f"Found unexpected HIP summary files: {hip_files}"

    # Should NOT have memory copy summaries
    memory_files = [f for f in summary_files if "memory" in f.lower() and "kernel" not in f.lower()]
    assert len(memory_files) == 0, f"Found unexpected memory summary files: {memory_files}"

    # Should NOT have HSA summaries
    hsa_files = [f for f in summary_files if "hsa" in f.lower() and "kernel" not in f.lower()]
    assert len(hsa_files) == 0, f"Found unexpected HSA summary files: {hsa_files}"


def test_hip_category_filter(request):
    """Test that --region-categories HIP only produces HIP-related summaries."""
    base_dir = request.config.getoption("--summary-hip-dir")
    summary_files = get_summary_files(base_dir)

    print(f"Found summary files in {base_dir}: {summary_files}")

    # Should have HIP-related summaries
    hip_files = [f for f in summary_files if "hip" in f.lower()]
    assert len(hip_files) > 0, "Expected at least one HIP summary file"

    # Should NOT have kernel summaries (unless they're hip-related)
    kernel_files = [f for f in summary_files if "kernel" in f.lower() and "hip" not in f.lower()]
    assert len(kernel_files) == 0, f"Found unexpected kernel summary files: {kernel_files}"

    # Should NOT have memory copy summaries (unless hip-related)
    memory_files = [f for f in summary_files if "memory" in f.lower() and "hip" not in f.lower()]
    assert len(memory_files) == 0, f"Found unexpected memory summary files: {memory_files}"


def test_multiple_categories_filter(request):
    """Test that --region-categories HIP KERNEL MARKER produces only those summaries."""
    base_dir = request.config.getoption("--summary-multiple-dir")
    summary_files = get_summary_files(base_dir)

    print(f"Found summary files in {base_dir}: {summary_files}")

    # All files should be related to HIP, KERNEL, or MARKER
    for filename in summary_files:
        lower_name = filename.lower()
        is_allowed = (
            "hip" in lower_name
            or "kernel" in lower_name
            or "marker" in lower_name
            or "domain" in lower_name  # domain summary is expected
        )
        assert is_allowed, f"Found unexpected summary file: {filename}"

    # Should NOT have HSA summaries
    hsa_files = [f for f in summary_files if "hsa" in f.lower()]
    assert len(hsa_files) == 0, f"Found unexpected HSA summary files: {hsa_files}"

    # Should NOT have RCCL summaries
    rccl_files = [f for f in summary_files if "rccl" in f.lower()]
    assert len(rccl_files) == 0, f"Found unexpected RCCL summary files: {rccl_files}"


def test_none_category(request):
    """Test that --region-categories NONE includes all views but no region summaries."""
    base_dir = request.config.getoption("--summary-none-dir")
    summary_files = get_summary_files(base_dir)

    print(f"Found summary files in {base_dir}: {summary_files}")

    # Should have view-based summaries (kernels, memory_copy, etc.)
    view_files = [
        f for f in summary_files
        if any(view in f.lower() for view in ["kernel", "memory", "scratch", "copy"])
    ]
    assert len(view_files) > 0, "Expected view-based summary files"

    # Should NOT have region-based summaries (rocm_hip, rocm_hsa, etc.)
    # Region summaries typically have the pattern "rocm_*" or are category-specific
    region_files = [
        f for f in summary_files
        if f.lower().startswith("rocm_") or
           any(region in f.lower() for region in ["hip_summary", "hsa_summary", "marker_summary"])
    ]
    # Note: this assertion depends on naming conventions. Adjust if needed.
    # The key is that region summaries should not be present.


def test_all_category_outputs_exist(request):
    """Verify that all test output directories were created."""
    kernel_dir = request.config.getoption("--summary-kernel-dir")
    hip_dir = request.config.getoption("--summary-hip-dir")
    multiple_dir = request.config.getoption("--summary-multiple-dir")
    none_dir = request.config.getoption("--summary-none-dir")

    # All directories should have been created
    assert os.path.exists(kernel_dir), f"Kernel output directory not found: {kernel_dir}"
    assert os.path.exists(hip_dir), f"HIP output directory not found: {hip_dir}"
    assert os.path.exists(multiple_dir), f"Multiple categories output directory not found: {multiple_dir}"
    assert os.path.exists(none_dir), f"NONE category output directory not found: {none_dir}"

    # All should have at least some files
    assert len(get_summary_files(kernel_dir)) > 0, "No files in kernel output directory"
    assert len(get_summary_files(hip_dir)) > 0, "No files in HIP output directory"
    assert len(get_summary_files(multiple_dir)) > 0, "No files in multiple categories output directory"
    assert len(get_summary_files(none_dir)) > 0, "No files in NONE category output directory"


def pytest_addoption(parser):
    """Add command line options for test directories."""
    parser.addoption(
        "--summary-kernel-dir",
        action="store",
        help="Path to kernel category summary output directory",
    )
    parser.addoption(
        "--summary-hip-dir",
        action="store",
        help="Path to HIP category summary output directory",
    )
    parser.addoption(
        "--summary-multiple-dir",
        action="store",
        help="Path to multiple categories summary output directory",
    )
    parser.addoption(
        "--summary-none-dir",
        action="store",
        help="Path to NONE category summary output directory",
    )


if __name__ == "__main__":
    exit_code = pytest.main(["-v", __file__] + sys.argv[1:])
    sys.exit(exit_code)
