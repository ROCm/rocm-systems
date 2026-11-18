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
test_early_hiprccl_late_hipmemcpy.py -- RCCL variation, disjoint subsets.

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED + RCCL_API_BUFFERED.
  Tool B (late):  dlopen'd mid-process, traces MEMORY_COPY_BUFFERED + KERNEL_DISPATCH_BUFFERED.

The workload runs HIP kernels + H2D/D2H copies AND RCCL all-reduce collectives each phase.
Tool A must capture RCCL (B-excluded) and HIP API; Tool B must capture kernel dispatch and
memory copies (A-excluded) from Phase 2 only, and no RCCL.
"""

import ctypes
import os
import sys

NUM_STREAMS = 4
PHASE1_ITERS = 3
PHASE2_ITERS = 5

TOOL_B_CONTEXTS = "MEMORY_COPY_BUFFERED,KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"


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


def _run_phase(kernels, rccl, iters):
    for _ in range(iters):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
            if kernels.hip_kernels_memcpy(stream) != 0:
                raise RuntimeError(f"hip_kernels_memcpy({stream}) failed")
        if rccl.rccl_allreduce(1) != 0:
            raise RuntimeError("rccl_allreduce() failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize() failed")


def main():
    print("=== Early HIP+RCCL Tool + Late HIP-memcpy/dispatch Tool ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    rccl = _load(lib_dir, "libpython-rccl.so")

    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    ndev = rccl.rccl_init()
    if ndev < 1:
        raise RuntimeError("rccl_init failed")
    print(f"Created {NUM_STREAMS} HIP streams and RCCL across {ndev} devices")

    print(f"\n--- Phase 1: {PHASE1_ITERS} iters (Tool A only) ---")
    _run_phase(kernels, rccl, PHASE1_ITERS)

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
    _run_phase(kernels, rccl, PHASE2_ITERS)

    rccl.rccl_fini()
    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
