#!@PYTHON_EXECUTABLE@

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

import rocprofsys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Input code coverage",
        default=None,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output code coverage",
        default=None,
        required=True,
    )

    args = parser.parse_args()

    data = None
    for itr in args.input:
        _summary, _details = rocprofsys.coverage.load(itr)
        if data is None:
            data = _details
        else:
            data = rocprofsys.coverage.concat(data, _details)

    summary = rocprofsys.coverage.get_summary(data)
    top = rocprofsys.coverage.get_top(data)
    bottom = rocprofsys.coverage.get_bottom(data)

    print("Top code coverage:")
    for itr in top:
        print(
            f"    {itr.count} | {itr.function} | {itr.module}:{itr.line} | {itr.source}"
        )

    print("Bottom code coverage:")
    for itr in bottom:
        print(
            f"    {itr.count} | {itr.function} | {itr.module}:{itr.line} | {itr.source}"
        )

    print("\nSaving code coverage")
    rocprofsys.coverage.save(summary, data, args.output)
