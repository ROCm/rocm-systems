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

EXPECTED_KERNELS = {
    "normal": (
        "roctx_attach_before_pause_kernel",
        "roctx_attach_after_resume_kernel",
    ),
    "selected": ("roctx_attach_inside_region_kernel",),
}

FORBIDDEN_KERNELS = {
    "normal": ("roctx_attach_during_pause_kernel",),
    "selected": (
        "roctx_attach_outside_before_kernel",
        "roctx_attach_outside_after_kernel",
    ),
}


def _kernel_names(json_data):
    data = json_data["rocprofiler-sdk-tool"]

    def get_kernel_name(kernel_id):
        return data["kernel_symbols"][kernel_id]["formatted_kernel_name"]

    kernel_dispatch_data = data["buffer_records"]["kernel_dispatch"]
    assert (
        len(kernel_dispatch_data) > 0
    ), "No kernel dispatches captured during attachment"

    names = []
    for dispatch in kernel_dispatch_data:
        dispatch_info = dispatch["dispatch_info"]
        assert dispatch_info["kernel_id"] > 0
        assert dispatch["end_timestamp"] >= dispatch["start_timestamp"]
        names.append(get_kernel_name(dispatch_info["kernel_id"]))

    return names


def _has_kernel(kernel_names, expected):
    return any(expected in name for name in kernel_names)


def test_roctx_pause_resume_kernel_gating(json_data, test_mode):
    kernel_names = _kernel_names(json_data)

    for expected in EXPECTED_KERNELS[test_mode]:
        assert _has_kernel(kernel_names, expected), (
            f"Expected kernel '{expected}' was not captured. "
            f"Captured kernels: {kernel_names}"
        )

    for forbidden in FORBIDDEN_KERNELS[test_mode]:
        assert not _has_kernel(kernel_names, forbidden), (
            f"Paused/outside-region kernel '{forbidden}' was unexpectedly captured. "
            f"Captured kernels: {kernel_names}"
        )


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
