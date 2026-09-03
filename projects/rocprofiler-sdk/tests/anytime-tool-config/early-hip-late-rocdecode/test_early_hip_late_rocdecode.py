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
test_early_hip_late_rocdecode.py -- rocDecode subset variation.

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED.
  Tool B (late):  dlopen'd mid-process, traces ROCDECODE_API_BUFFERED.

The workload drives the rocDecode bitstream-reader API each phase. We are testing the SDK's
ROCDECODE_API *tracing*, not video decoding: rocprofiler intercepts at the dispatch table,
so these (CPU-side) rocDecode calls are traced and work even when the AMD VCN VA-API video
decoder is unavailable. Tool B must capture rocDecode API records from Phase 2; Tool A must
capture HIP and no rocDecode.
"""

import ctypes
import os
import sys


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
    print("=== Early HIP Tool + Late rocDecode Tool ===\n")
    lib_dir = _lib_dir()

    video = os.environ.get("ROCDECODE_VIDEO_FILE")
    if not video or not os.path.isfile(video):
        print("test skipped: rocDecode video file not available")
        return 0

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    dec = _load(lib_dir, "libpython-rocdecode.so")
    dec.rocdecode_decode_file.argtypes = [ctypes.c_char_p]
    dec.rocdecode_decode_file.restype = ctypes.c_int

    # HIP kernel workload so the early HIP tool has activity to capture (the rocDecode
    # bitstream API is CPU-side and does not drive HIP).
    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(4) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")

    def _hip_work():
        for _ in range(3):
            for s in range(4):
                kernels.hip_kernels_launch(s)
        kernels.hip_kernels_synchronize()

    # Phase 1: HIP kernels + rocDecode API before Tool B registers.
    print("\n--- Phase 1: HIP kernels + rocDecode bitstream API (Tool A only) ---")
    _hip_work()
    n1 = dec.rocdecode_decode_file(video.encode())
    if n1 < 0:
        raise RuntimeError("rocdecode_decode_file(phase1) failed")
    print(f"  read {n1} packets")

    print("\n--- Registering Tool B via json_tool_force_configure() ---")
    os.environ["ROCPROFILER_TOOL_CONTEXTS"] = "ROCDECODE_API_BUFFERED"
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = late_output_file
    tool_b = _load(lib_dir, "librocprofiler-sdk-json-tool-late.so")
    tool_b.json_tool_force_configure.restype = ctypes.c_int
    tool_b.json_tool_force_configure.argtypes = []
    status = tool_b.json_tool_force_configure()
    if status != 0:
        raise RuntimeError(f"json_tool_force_configure() failed with {status}")
    print("  Tool B registered")

    # Phase 2: HIP kernels + rocDecode API after Tool B registers -- Tool B must capture
    # the rocDecode calls.
    print("\n--- Phase 2: HIP kernels + rocDecode bitstream API (Tool A + Tool B) ---")
    _hip_work()
    n2 = dec.rocdecode_decode_file(video.encode())
    if n2 < 0:
        raise RuntimeError("rocdecode_decode_file(phase2) failed")
    print(f"  read {n2} packets")

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
