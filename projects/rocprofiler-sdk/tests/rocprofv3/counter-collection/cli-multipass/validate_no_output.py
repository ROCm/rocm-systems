#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""Verify the rejected multi-pass runs produced no output whatsoever."""

import glob
import os
import sys

import pytest


def _matching_dirs(binary_dir, patterns):
    """Directories under binary_dir matching any of the given glob patterns"""
    found = []
    for pattern in patterns:
        found.extend(
            path
            for path in glob.glob(os.path.join(binary_dir, pattern))
            if os.path.isdir(path)
        )
    return sorted(set(found))


def test_expected_directory_is_being_checked(binary_dir, present_globs):
    """Guard against the absence checks below passing because we looked in the
    wrong place: the fixture files copied next to the output directories must be
    visible from binary_dir."""
    assert present_globs, "--present-globs must list at least one pattern"

    for pattern in present_globs:
        assert glob.glob(
            os.path.join(binary_dir, pattern)
        ), f"expected '{pattern}' under {binary_dir}; --binary-dir is likely wrong"


def test_no_output_directories(binary_dir, absent_globs):
    """A run rejected before launch must not create its output directory.

    The patterns are globs because rocprofv3 only substitutes %argt% in the
    output path once the tool library runs, so a directory leaked by a rejected
    run could appear under either the substituted or the literal name.
    """
    assert absent_globs, "--absent-globs must list at least one pattern"

    leaked = _matching_dirs(binary_dir, absent_globs)
    assert not leaked, f"rejected runs must not create output directories: {leaked}"

    residue = []
    for pattern in absent_globs:
        residue.extend(
            glob.glob(os.path.join(binary_dir, pattern, "**", "*.csv"), recursive=True)
        )
    assert not residue, f"rejected runs must not emit any data: {sorted(residue)}"


if __name__ == "__main__":
    sys.exit(pytest.main(["-x", __file__] + sys.argv[1:]))
