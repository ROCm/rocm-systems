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
test_early_hip_late_memalloc.py -- Execute driver for the memory-allocation variation.

Same anytime-init shape as the other variations; the late tool enables the
MEMORY_ALLOCATION service:

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED only.
  Tool B (late):  dlopen'd mid-process, traces MEMORY_ALLOCATION_BUFFERED only.

Each phase issues NUM_ALLOCS hipMalloc+hipFree pairs. Tool B must capture exactly the
Phase 2 allocate and free records (Phase 1 missed). The one-time HIP startup allocation
happens during Phase 1, before Tool B registers, so it does not perturb Tool B's counts.
"""

import ctypes
import os
import sys

PHASE1_ALLOCS = 6  # alloc/free pairs before Tool B registers (must be missed)
PHASE2_ALLOCS = 10  # alloc/free pairs after Tool B registers (must be captured)

TOOL_B_CONTEXTS = "MEMORY_ALLOCATION_BUFFERED"


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


def _run_phase(kernels, count):
    for _ in range(count):
        if kernels.hip_kernels_alloc_free() != 0:
            raise RuntimeError("hip_kernels_alloc_free() failed")


def main():
    print("=== Early HIP Tool + Late Memory-Allocation Tool Test ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    kernels = _load(lib_dir, "libpython-hip-kernels.so")

    # Phase 1: allocations BEFORE Tool B registers (includes the one-time HIP startup
    # allocation). Tool B must miss all of these.
    print(f"\n--- Phase 1: {PHASE1_ALLOCS} alloc/free pairs (Tool A only) ---")
    _run_phase(kernels, PHASE1_ALLOCS)

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

    print(f"\n--- Phase 2: {PHASE2_ALLOCS} alloc/free pairs (Tool A + Tool B) ---")
    _run_phase(kernels, PHASE2_ALLOCS)

    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
