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
test_early_hip_late_rocjpeg.py -- rocJPEG subset variation.

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED.
  Tool B (late):  dlopen'd mid-process, traces ROCJPEG_API_BUFFERED.

The workload issues rocJPEG API calls each phase. We are testing the SDK's ROCJPEG_API
*tracing*, not JPEG decoding: rocprofiler intercepts at the dispatch table, so the calls
are traced even if they fail (e.g. when the AMD VCN VA-API driver is absent). The workload
module therefore issues only crash-safe calls and ignores their return codes. Tool B must
capture rocJPEG API records from Phase 2; Tool A must capture HIP and no rocJPEG.
"""

import ctypes
import os
import sys

PHASE1_DECODES = 3
PHASE2_DECODES = 5

TOOL_B_CONTEXTS = "ROCJPEG_API_BUFFERED"


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
    print("=== Early HIP Tool + Late rocJPEG Tool ===\n")
    lib_dir = _lib_dir()

    image = os.environ.get("ROCJPEG_IMAGE_FILE")
    if not image or not os.path.isfile(image):
        print("test skipped: rocJPEG image file not available")
        return 0

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    jpeg = _load(lib_dir, "libpython-rocjpeg.so")
    jpeg.rocjpeg_decode_file.argtypes = [ctypes.c_char_p, ctypes.c_int]
    jpeg.rocjpeg_decode_file.restype = ctypes.c_int

    if jpeg.rocjpeg_init() != 0:
        raise RuntimeError("rocjpeg_init failed")

    print(f"\n--- Phase 1: {PHASE1_DECODES} rocJPEG API reps (Tool A only) ---")
    if jpeg.rocjpeg_decode_file(image.encode(), PHASE1_DECODES) != 0:
        raise RuntimeError("rocjpeg_decode_file(phase1) failed")

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

    print(f"\n--- Phase 2: {PHASE2_DECODES} rocJPEG API reps (Tool A + Tool B) ---")
    if jpeg.rocjpeg_decode_file(image.encode(), PHASE2_DECODES) != 0:
        raise RuntimeError("rocjpeg_decode_file(phase2) failed")

    jpeg.rocjpeg_fini()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
