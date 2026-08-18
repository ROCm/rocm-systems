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

import argparse
import importlib.util
import pathlib
import sys


def _load_rocprofv3(path):
    spec = importlib.util.spec_from_file_location("rocprofv3_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_reattach_option_rejected(rocprofv3, option, previous_value, current_value):
    previous_args = rocprofv3.dotdict(
        {
            "kernel_trace": True,
            "selected_regions": False,
            "selected_regions_ref_count": False,
        }
    )
    current_args = argparse.Namespace(
        kernel_trace=True,
        selected_regions=False,
        selected_regions_ref_count=False,
    )

    previous_args[option] = previous_value
    setattr(current_args, option, current_value)

    try:
        rocprofv3.get_args(
            current_args,
            previous_args,
            filter=rocprofv3.REATTACH_INVARIANT_FILTER,
            require_in_both=True,
            ignore_prev_inp=r"attach_duration_msec",
        )
    except RuntimeError as exc:
        if f"Option '{option}'" not in str(exc):
            raise AssertionError(f"Unexpected error for {option}: {exc}") from exc
    else:
        raise AssertionError(f"Expected {option} reattach mismatch to be rejected")


def test_selected_regions_options_are_reattach_invariants(rocprofv3_path):
    rocprofv3 = _load_rocprofv3(rocprofv3_path)
    for option in ("selected_regions", "selected_regions_ref_count"):
        _assert_reattach_option_rejected(rocprofv3, option, False, True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_reattach_config.py <path-to-rocprofv3.py>")

    rocprofv3_path = pathlib.Path(sys.argv[1]).resolve()
    test_selected_regions_options_are_reattach_invariants(rocprofv3_path)
