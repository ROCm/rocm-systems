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
test_early_hip_late_dispatch.py -- Execute driver for the baseline anytime test.

This is the in-scope "anytime initialization" scenario: rocprofiler-sdk is already
present in the process (an early tool is LD_PRELOAD'd by CMake), and a SECOND tool
configures itself as a client mid-process via rocprofiler_force_configure().

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED only. Output file and
                  contexts are set in the CMake environment.
  Tool B (late):  dlopen'd by this script after some work has already run, traces
                  KERNEL_DISPATCH_BUFFERED (+ CODE_OBJECT). Gets its own contexts and
                  output file via os.environ, snapshotted at force_configure time.

Flow (all in one process; no subprocess, so the in-process tools observe the GPU work):

    create NUM_STREAMS HIP streams      <- queues exist before Tool B registers
    Phase 1: NUM_STREAMS x PHASE1_ITERS kernels   (only Tool A active)
    --- register Tool B via json_tool_force_configure() ---
    Phase 2: NUM_STREAMS x PHASE2_ITERS kernels   (Tool A + Tool B active)

Expectations (checked by validate.py):
  - Tool A captured HIP API records and ZERO kernel dispatches (it did not enable them).
  - Tool B captured exactly the Phase 2 kernel dispatches (Phase 1 was missed), even
    though the streams were created before Tool B registered.
"""

import ctypes
import os
import sys

NUM_STREAMS = 8
PHASE1_ITERS = 3  # 3 * 8 = 24 kernels before Tool B registers (B must miss these)
PHASE2_ITERS = 5  # 5 * 8 = 40 kernels after Tool B registers  (B must capture these)

# Tool B (late) configuration, applied to os.environ right before force_configure so the
# json-tool instance snapshots these at configure time.
TOOL_B_CONTEXTS = "KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"


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


def _launch_all(kernels, iters):
    for _ in range(iters):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize() failed")


def main():
    print("=== Early Tool + Late Kernel-Dispatch Tool Test ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    # Load the HIP kernel workload engine and create the streams. These queues exist
    # before Tool B registers -- the key condition under test.
    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    print(f"Created {NUM_STREAMS} HIP streams (before Tool B registers)")

    # Phase 1: work BEFORE Tool B registers. Tool B must not see these dispatches.
    print(f"\n--- Phase 1: {NUM_STREAMS * PHASE1_ITERS} kernels (Tool A only) ---")
    _launch_all(kernels, PHASE1_ITERS)

    # Register Tool B late, in this same process. Set its contexts + output file first so
    # the json-tool instance snapshots them at configure time (distinct from Tool A).
    print("\n--- Registering Tool B via json_tool_force_configure() ---")
    os.environ["ROCPROFILER_TOOL_CONTEXTS"] = TOOL_B_CONTEXTS
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = late_output_file
    tool_b = _load(lib_dir, "librocprofiler-sdk-json-tool-late.so")
    tool_b.json_tool_force_configure.restype = ctypes.c_int
    tool_b.json_tool_force_configure.argtypes = []
    status = tool_b.json_tool_force_configure()
    if status != 0:
        raise RuntimeError(f"json_tool_force_configure() failed with {status}")
    print("  Tool B registered")

    # Phase 2: work AFTER Tool B registers, on the SAME pre-existing streams. Tool B must
    # capture exactly these dispatches.
    print(f"\n--- Phase 2: {NUM_STREAMS * PHASE2_ITERS} kernels (Tool A + Tool B) ---")
    _launch_all(kernels, PHASE2_ITERS)

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
