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

import os
import sys
import time
import rocprofsys
from rocprofsys.user import region as omni_user_region
from rocprofsys.profiler import config as omni_config

_prefix = ""


def loop(n):
    pass


@rocprofsys.profile()
def run(i, n, v):
    for l in range(n * n):
        loop(v + l)
    return v + (n * n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-iterations", help="Number", type=int, default=100)
    parser.add_argument("-v", "--value", help="Starting value", type=int, default=10)
    args = parser.parse_args()

    omni_config.include_args = True
    _prefix = os.path.basename(__file__)
    print(f"[{_prefix}] Executing {args.num_iterations} iterations...\n")
    ans = 0
    for i in range(args.num_iterations):
        beg = ans
        ans = run(i, args.value, beg)
        print(f"[{_prefix}] [{i}] result of run({args.value}, {beg}) = {ans}")
