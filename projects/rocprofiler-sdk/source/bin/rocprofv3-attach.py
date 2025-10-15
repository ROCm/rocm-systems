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
import signal
import sys
import time

ROCPROFV3_ATTACH_DIR = os.path.dirname(os.path.realpath(__file__))
ROCM_DIR = os.path.dirname(ROCPROFV3_ATTACH_DIR)
ROCPROF_ATTACH_TOOL_LIBRARY = f"{ROCM_DIR}/lib/rocprofiler-sdk/librocprofv3-attach.so"


def main(
    pid=os.environ.get("ROCPROF_ATTACH_PID", None),
    attach_library=os.environ.get(
        "ROCPROF_ATTACH_TOOL_LIBRARY", ROCPROF_ATTACH_TOOL_LIBRARY
    ),
    duration=os.environ.get("ROCPROF_ATTACH_DURATION", None),
):
    if pid is None:
        raise RuntimeError("rocprofv3_attach called with no PID specified")

    print(f"Attaching to PID {pid} using library {attach_library}")

    # Load the shared library into ctypes and attach
    try:
        c_lib = ctypes.CDLL(attach_library)
        c_lib.attach.restype = ctypes.c_int
        c_lib.attach.argtypes = [ctypes.c_uint]
        attach_status = c_lib.attach(int(pid))
    except Exception as e:
        raise RuntimeError(f"Exception during library load and attachment: {e}")

    if attach_status != 0:
        raise RuntimeError(
            f"Calling attach in {attach_library} returned non-zero status {attach_status}"
        )

    print(f"Attaching to PID {pid} using library {attach_library} :: success")

    def detach():
        try:
            c_lib.detach()
        except Exception as e:
            print(f"Exception during detachment: {e}")

    def signal_handler(sig, frame):
        print("\nCaught signal SIGINT, detaching")
        detach()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if duration is None:
        sys.stdout.write("Press Enter to detach...")
        sys.stdout.flush()  # Force the prompt to appear immediately
        input()  # Now wait for input
    else:
        time.sleep(int(duration) / 1000)

    detach()


if __name__ == "__main__":
    main()
