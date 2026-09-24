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
test_early_multi_late_overlap.py -- Two tools with large, OVERLAPPING service subsets.

Proves that two independent clients can enable overlapping sets of services and each
receives its own complete stream of records for the shared services, while exclusive
services are seen only by the tool that enabled them.

  Tool A (early): HIP, HSA, kernel dispatch, memory copy, marker          (5 services)
  Tool B (late):  HSA, kernel dispatch, memory copy, marker, mem-alloc    (5 services)

  overlap (4): HSA, kernel dispatch, memory copy, marker  -> both capture
  A only:      HIP                                        -> absent from B
  B only:      memory allocation                          -> absent from A

The workload (kernels + H2D/D2H copies + alloc/free + roctx markers) is driven in both
phases; Tool B registers between them and must capture exactly the Phase 2 activity for
its services.
"""

import ctypes
import os
import sys

NUM_STREAMS = 4
PHASE1_ITERS = 3  # before Tool B registers (must be missed by B)
PHASE2_ITERS = 5  # after Tool B registers (must be captured by B)

# Tool B enables HSA + dispatch + memcpy + marker + alloc (overlaps A on the first four).
TOOL_B_CONTEXTS = (
    "HSA_API_BUFFERED,KERNEL_DISPATCH_BUFFERED,MEMORY_COPY_BUFFERED,"
    "MARKER_API_BUFFERED,MEMORY_ALLOCATION_BUFFERED"
)


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


def _run_phase(kernels, iters):
    for _ in range(iters):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
            if kernels.hip_kernels_memcpy(stream) != 0:
                raise RuntimeError(f"hip_kernels_memcpy({stream}) failed")
        if kernels.hip_kernels_alloc_free() != 0:
            raise RuntimeError("hip_kernels_alloc_free() failed")
        if kernels.hip_kernels_marker() != 0:
            raise RuntimeError("hip_kernels_marker() failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize() failed")


def main():
    print("=== Early + Late Tools, Overlapping Service Subsets ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    print(f"Created {NUM_STREAMS} HIP streams (before Tool B registers)")

    print(f"\n--- Phase 1: {PHASE1_ITERS} iters (Tool A only) ---")
    _run_phase(kernels, PHASE1_ITERS)

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

    print(f"\n--- Phase 2: {PHASE2_ITERS} iters (Tool A + Tool B) ---")
    _run_phase(kernels, PHASE2_ITERS)

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
