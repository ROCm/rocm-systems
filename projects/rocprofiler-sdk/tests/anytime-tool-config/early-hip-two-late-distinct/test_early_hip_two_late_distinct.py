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
test_early_hip_two_late_distinct.py -- TWO tools register late in one process.

Exercises multiple `rocprofiler_force_configure()` calls in a single process, each from a
separate json-tool library (each json-tool .so keeps all state in globals, so two
concurrent instances need two distinct libraries):

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED.
  Tool B1 (late): json-tool-late,  traces KERNEL_DISPATCH_BUFFERED.
  Tool B2 (late): json-tool-late2, traces MEMORY_COPY_BUFFERED.

Both late tools register between phase 1 and phase 2. Each must capture exactly the Phase 2
activity for its own service and nothing from the other tools' services.
"""

import ctypes
import os
import sys

NUM_STREAMS = 4
PHASE1_ITERS = 3
PHASE2_ITERS = 5

COPIES_PER_CALL = 2  # hip_kernels_memcpy issues H2D + D2H


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


def _register_late(lib_dir, lib_name, contexts, output_file):
    os.environ["ROCPROFILER_TOOL_CONTEXTS"] = contexts
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = output_file
    tool = _load(lib_dir, lib_name)
    tool.json_tool_force_configure.restype = ctypes.c_int
    tool.json_tool_force_configure.argtypes = []
    if tool.json_tool_force_configure() != 0:
        raise RuntimeError(f"json_tool_force_configure() failed for {lib_name}")


def _run_phase(kernels, iters):
    for _ in range(iters):
        for stream in range(NUM_STREAMS):
            if kernels.hip_kernels_launch(stream) != 0:
                raise RuntimeError(f"hip_kernels_launch({stream}) failed")
            if kernels.hip_kernels_memcpy(stream) != 0:
                raise RuntimeError(f"hip_kernels_memcpy({stream}) failed")
    if kernels.hip_kernels_synchronize() != 0:
        raise RuntimeError("hip_kernels_synchronize() failed")


def main():
    print("=== Early HIP Tool + Two Distinct Late Tools ===\n")
    lib_dir = _lib_dir()

    b1_output = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    b2_output = os.environ.get("ROCPROFILER_TOOL_LATE2_OUTPUT_FILE")
    if not b1_output or not b2_output:
        raise RuntimeError(
            "ROCPROFILER_TOOL_LATE_OUTPUT_FILE and ROCPROFILER_TOOL_LATE2_OUTPUT_FILE "
            "must both be set"
        )

    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    print(f"Created {NUM_STREAMS} HIP streams (before late tools register)")

    print(f"\n--- Phase 1: {PHASE1_ITERS} iters (Tool A only) ---")
    _run_phase(kernels, PHASE1_ITERS)

    print("\n--- Registering Tool B1 (kernel dispatch) ---")
    _register_late(
        lib_dir,
        "librocprofiler-sdk-json-tool-late.so",
        "KERNEL_DISPATCH_BUFFERED,CODE_OBJECT",
        b1_output,
    )

    print("--- Registering Tool B2 (memory copy) ---")
    _register_late(
        lib_dir,
        "librocprofiler-sdk-json-tool-late2.so",
        "MEMORY_COPY_BUFFERED",
        b2_output,
    )

    print(f"\n--- Phase 2: {PHASE2_ITERS} iters (Tool A + B1 + B2) ---")
    _run_phase(kernels, PHASE2_ITERS)

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
