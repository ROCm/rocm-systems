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


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
