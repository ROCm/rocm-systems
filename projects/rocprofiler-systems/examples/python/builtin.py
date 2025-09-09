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
import random

_prefix = ""


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def inefficient(n):
    print(f"[{_prefix}] ... running inefficient({n})")
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    _len = a * n * n
    _arr = [random.random() for _ in range(_len)]
    _sum = sum(_arr)
    print(f"[{_prefix}] ... sum of {_len} random elements: {_sum}")
    return _sum


@profile
def run(n):
    _ret = 0
    _ret += fib(n)
    _ret += inefficient(n)
    return _ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-iterations", help="Number", type=int, default=3)
    parser.add_argument("-v", "--value", help="Starting value", type=int, default=20)
    args = parser.parse_args()

    _prefix = os.path.basename(__file__)
    print(f"[{_prefix}] Executing {args.num_iterations} iterations...\n")
    for i in range(args.num_iterations):
        ans = run(args.value)
        print(f"[{_prefix}] [{i}] result of run({args.value}) = {ans}\n")
