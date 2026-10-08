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
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Exercises multiprocessing's forkserver start method under the profiler.

The forkserver saves every signal disposition with signal.signal() and restores
it in each forked child. A disposition the interpreter did not install itself is
recorded as None, and None cannot be restored -- so a profiler holding a C
handler when the interpreter starts breaks the forkserver, and with it every
worker pool built on it. PyTorch's DataLoader with num_workers > 0 is the common
case, and forkserver is the default start method on Linux from Python 3.14.

The dispositions are printed before the pool runs: under a profiler that
installs too early they read None, which identifies the mechanism when the pool
itself fails.

No device work -- the processes under test are the forkserver and its children,
which never touch a GPU.
"""

import multiprocessing
import signal
import sys

WATCHED = ("SIGINT", "SIGQUIT", "SIGABRT", "SIGTERM")
POOL_TIMEOUT_SEC = 60
INPUTS = [1, 2, 3, 4]
EXPECTED = [1, 4, 9, 16]


def square(value):
    return value * value


def main():
    for name in WATCHED:
        print(f"{name} disposition: {signal.getsignal(getattr(signal, name))!r}")

    context = multiprocessing.get_context("forkserver")
    with context.Pool(2) as pool:
        result = pool.map_async(square, INPUTS).get(timeout=POOL_TIMEOUT_SEC)

    if result != EXPECTED:
        print(f"forkserver pool returned {result}, expected {EXPECTED}")
        return 1

    print("forkserver pool completed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - any failure here is a test failure
        print(f"forkserver pool failed: {type(exc).__name__}: {exc}")
        sys.exit(1)
