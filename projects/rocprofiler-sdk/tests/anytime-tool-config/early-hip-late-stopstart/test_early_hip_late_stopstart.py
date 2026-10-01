#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

"""
test_early_hip_late_stopstart.py -- context stop/start on a late-registered tool.

A tool registers late, then its contexts are stopped and restarted mid-process while GPU
work continues. Work performed while the contexts are stopped must NOT be captured by that
tool -- but a DIFFERENT tool tracing the same service, which was never stopped, must still
capture it. This proves per-client context independence: stopping one tool's context pauses
only that tool's collection, not the underlying GPU work or other tools.

  Tool A (early): LD_PRELOAD'd by CMake, traces KERNEL_DISPATCH_BUFFERED, NEVER stopped.
  Tool B (late):  dlopen'd, traces KERNEL_DISPATCH_BUFFERED, driven through stop()/start()
                  via the json_tool_stop()/json_tool_start() hooks.

Both tools trace the same service (kernel dispatch) over the same streams:
    warm-up (before B registers; only A sees it)
    register Tool B
    Phase A: A + B active             -> both capture
    json_tool_stop()  (Tool B only)
    Phase B: A active, B stopped      -> A captures, B drops
    json_tool_start() (Tool B only)
    Phase C: A + B active             -> both capture

validate.py asserts: Tool B captured strictly fewer dispatches than Tool A; and the
dispatches A has that B lacks all fall within the stop/start window (the work B dropped was
still captured by A).
"""

import ctypes
import os
import sys

NUM_STREAMS = 4
PHASE_ITERS = 5  # 5 * 4 = 20 kernels per phase


def _lib_dir():
    lib_dir = os.environ.get("ROCPROFILER_SDK_TEST_LIB_DIR")
    if not lib_dir or not os.path.isdir(lib_dir):
        raise RuntimeError("ROCPROFILER_SDK_TEST_LIB_DIR is not set to a valid directory")
    return lib_dir


def _load(lib_dir, name):
    path = os.path.join(lib_dir, name)
    if not os.path.isfile(path):
        raise RuntimeError(f"Could not find {name} in {lib_dir}")
    return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _launch_all(kernels):
    for _ in range(PHASE_ITERS):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize() failed")


def main():
    print("=== Early HIP Tool + Late Tool with stop/start cycles ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")

    print(f"--- Warm-up: {NUM_STREAMS * PHASE_ITERS} kernels (before Tool B) ---")
    _launch_all(kernels)

    print("--- Registering Tool B via json_tool_force_configure() ---")
    os.environ["ROCPROFILER_TOOL_CONTEXTS"] = "KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = late_output_file
    tool_b = _load(lib_dir, "librocprofiler-sdk-json-tool-late.so")
    for fn in ("json_tool_force_configure", "json_tool_stop", "json_tool_start"):
        getattr(tool_b, fn).restype = ctypes.c_int
        getattr(tool_b, fn).argtypes = []
    if tool_b.json_tool_force_configure() != 0:
        raise RuntimeError("json_tool_force_configure() failed")
    print("  Tool B registered")

    print(f"\n--- Phase A: {NUM_STREAMS * PHASE_ITERS} kernels (Tool B active) ---")
    _launch_all(kernels)

    print("\n--- json_tool_stop(): stopping Tool B contexts ---")
    if tool_b.json_tool_stop() != 0:
        raise RuntimeError("json_tool_stop() failed")

    print(f"\n--- Phase B: {NUM_STREAMS * PHASE_ITERS} kernels (Tool B stopped) ---")
    _launch_all(kernels)

    print("\n--- json_tool_start(): restarting Tool B contexts ---")
    if tool_b.json_tool_start() != 0:
        raise RuntimeError("json_tool_start() failed")

    print(f"\n--- Phase C: {NUM_STREAMS * PHASE_ITERS} kernels (Tool B active again) ---")
    _launch_all(kernels)

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
