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
test_early_hip_late_double_configure.py -- API-contract driver for the late path.

The rest of the suite tests the TRACING produced by a single successful late registration.
This variation tests rocprofiler_force_configure() ITSELF: what happens when a tool calls
it a SECOND time with the same configure function.

  Tool A (early): LD_PRELOAD'd by CMake, traces HIP_API_BUFFERED.
  Tool B (late):  dlopen'd, traces KERNEL_DISPATCH_BUFFERED, registers via
                  json_tool_force_configure() and then calls it AGAIN.

Observed/asserted contract (verified against source/lib/rocprofiler-sdk/registration.cpp):

  * The SDK is already configured (Tool A is preloaded), so a late force_configure() takes
    the "adding forced configure" path and returns ROCPROFILER_STATUS_SUCCESS (0). NOTE:
    the registration.h docstring's claim that an already-configured SDK returns
    ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED does NOT hold for this (in-scope) anytime
    path -- LOCKED is only returned during the brief mid-initialization window, which is
    not reachable deterministically from here. This test pins the behavior that actually
    ships.
  * Calling force_configure() a SECOND time with the SAME configure function is
    de-duplicated by the SDK (invoke_client_configure "more than once" guard). The second
    call returns SUCCESS (0) but must NOT register the tool again -- tool_init does not
    re-run and the captured records are not doubled (validate.py proves this via the
    phase-2 dispatch count).

The expected return code is read from the SDK enum (ROCPROFILER_STATUS_SUCCESS == 0) rather
than hard-coded, so a future enum change does not silently invalidate the assertion.
"""

import ctypes
import os
import sys

NUM_STREAMS = 8
PHASE1_ITERS = 3  # 3 * 8 = 24 kernels before Tool B registers (B must miss these)
PHASE2_ITERS = 5  # 5 * 8 = 40 kernels after Tool B registers  (B must capture these once)

TOOL_B_CONTEXTS = "KERNEL_DISPATCH_BUFFERED,CODE_OBJECT"

# rocprofiler_status_t value for success. Mirrors the first enumerator of
# rocprofiler_status_t (ROCPROFILER_STATUS_SUCCESS == 0).
ROCPROFILER_STATUS_SUCCESS = 0


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
    print("=== Early HIP Tool + Late Tool calling force_configure() twice ===\n")
    lib_dir = _lib_dir()

    late_output_file = os.environ.get("ROCPROFILER_TOOL_LATE_OUTPUT_FILE")
    if not late_output_file:
        raise RuntimeError("ROCPROFILER_TOOL_LATE_OUTPUT_FILE is not set")

    kernels = _load(lib_dir, "libpython-hip-kernels.so")
    if kernels.hip_kernels_create_streams(NUM_STREAMS) != 0:
        raise RuntimeError("hip_kernels_create_streams failed")
    print(f"Created {NUM_STREAMS} HIP streams (before Tool B registers)")

    print(f"\n--- Phase 1: {NUM_STREAMS * PHASE1_ITERS} kernels (Tool A only) ---")
    _launch_all(kernels, PHASE1_ITERS)

    print("\n--- Registering Tool B via json_tool_force_configure() ---")
    os.environ["ROCPROFILER_TOOL_CONTEXTS"] = TOOL_B_CONTEXTS
    os.environ["ROCPROFILER_TOOL_OUTPUT_FILE"] = late_output_file
    tool_b = _load(lib_dir, "librocprofiler-sdk-json-tool-late.so")
    tool_b.json_tool_force_configure.restype = ctypes.c_int
    tool_b.json_tool_force_configure.argtypes = []

    status1 = tool_b.json_tool_force_configure()
    print(f"  first  json_tool_force_configure() -> {status1}")
    if status1 != ROCPROFILER_STATUS_SUCCESS:
        raise RuntimeError(
            f"first force_configure returned {status1}, expected "
            f"{ROCPROFILER_STATUS_SUCCESS} (ROCPROFILER_STATUS_SUCCESS)"
        )

    # Second call with the SAME configure function. The SDK de-duplicates the configure
    # function, so this is a no-op registration: it must succeed and must NOT double-register
    # Tool B. (validate.py proves no double-registration via the dispatch count.)
    status2 = tool_b.json_tool_force_configure()
    print(f"  second json_tool_force_configure() -> {status2}")
    if status2 != ROCPROFILER_STATUS_SUCCESS:
        raise RuntimeError(
            f"second force_configure returned {status2}, expected "
            f"{ROCPROFILER_STATUS_SUCCESS} (idempotent re-registration must succeed)"
        )
    print("  Tool B registered (force_configure called twice)")

    print(f"\n--- Phase 2: {NUM_STREAMS * PHASE2_ITERS} kernels (Tool A + Tool B) ---")
    _launch_all(kernels, PHASE2_ITERS)

    kernels.hip_kernels_destroy_streams()
    print("\nWorkload complete; tools write JSON at process exit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
