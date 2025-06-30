#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import ctypes
import os
import sys
import time

ROCPROFV3_AVAIL_DIR = os.path.dirname(os.path.realpath(__file__))
ROCM_DIR = os.path.dirname(ROCPROFV3_AVAIL_DIR)
ROCPROF_ATTACH_TOOL_LIBRARY = f"{ROCM_DIR}/libexec/rocprofiler-sdk/librocprofv3-attach.so"

MAX_STR = 256
libname = os.environ.get("ROCPROF_ATTACH_TOOL_LIBRARY", ROCPROF_ATTACH_TOOL_LIBRARY)
c_lib = ctypes.CDLL(libname)

if c_lib is None:
    fatal_error(f"Error opening {libname}")

c_lib.attach.argtypes = [ctypes.c_uint]

if __name__ == "__main__":
    # Load the shared library into ctypes

    pid = os.environ.get("ROCPROF_ATTACH_PID", None)

    if pid is None:
        raise RuntimeError(
            "rocprofv3_attach called without PID environment variable set (ROCPROF_ATTACH_PID)"
        )

    c_lib.attach(int(pid))

    duration = os.environ.get("ROCPROF_ATTACH_DURATION", None)

    if duration is None:
        input("Press Enter to detach...")
    else:
        time.sleep(int(duration) / 1000)
    
    c_lib.detach()
