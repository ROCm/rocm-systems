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
test_early_hsa_late_hsa.py -- HSA-API subset variation, driven by DIRECT HSA calls.

Unlike the other variations (which drive HSA indirectly through HIP), this one uses the
python-hsa module, which calls the HSA core API directly (no HIP). It proves the HSA
dispatch table specifically is captured by a tool that registers late.

  Tool A (early): LD_PRELOAD'd by CMake, traces HSA_API_BUFFERED.
  Tool B (late):  dlopen'd mid-process, traces HSA_API_BUFFERED.

Both tools trace the same service here; the proof is timing-based: Tool B captures only
the HSA activity issued after it registered (Phase 2), not Phase 1.
"""

import ctypes
import os
import sys

PHASE1_REPS = 3  # HSA workload iterations before Tool B registers (must be missed)
PHASE2_REPS = 5  # HSA workload iterations after Tool B registers (must be captured)

TOOL_B_CONTEXTS = "HSA_API_BUFFERED"


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


def main():
    print("=== Early HSA Tool + Late HSA Tool (direct HSA workload) ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    hsa = _load(lib_dir, "libpython-hsa.so")
    if hsa.hsa_workload_init() != 0:
        raise RuntimeError("hsa_workload_init failed")
    print("HSA initialized")

    print(f"\n--- Phase 1: {PHASE1_REPS} HSA workload reps (Tool A only) ---")
    if hsa.hsa_workload_run(PHASE1_REPS) != 0:
        raise RuntimeError("hsa_workload_run(phase1) failed")

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

    print(f"\n--- Phase 2: {PHASE2_REPS} HSA workload reps (Tool A + Tool B) ---")
    if hsa.hsa_workload_run(PHASE2_REPS) != 0:
        raise RuntimeError("hsa_workload_run(phase2) failed")

    hsa.hsa_workload_fini()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
