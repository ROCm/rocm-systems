#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################
"""
Test driver: verify export_sqlite_query produces correct output for the five
stdlib-only formats (console, csv, json, html, md) when pandas is unavailable.

Pandas is blocked by inserting a None sentinel into sys.modules before any
rocpd code is imported. Any lazy 'import pandas' inside the library will then
raise ImportError, exercising the fallback path in export_sqlite_query.
"""

import sys
import os
import argparse

# Block pandas before any rocpd submodule is imported so every lazy
# 'import pandas' inside the library raises ImportError.
sys.modules["pandas"] = None  # type: ignore[assignment]

import rocpd
import rocpd.__main__


def main():
    # Pass thru all args to the rocpd module
    rocpd.__main__.main(sys.argv[1:])
    sys.exit(0)


if __name__ == "__main__":
    main()
