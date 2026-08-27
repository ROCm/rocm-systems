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
hip_kernels -- HIP kernel launcher for torchesque.

Uses ctypes to load the python-hip-kernels shared library and
launch trivial HIP kernels on multiple streams.
"""

import ctypes
import os
import sys

_lib = None


def _find_library():
    """Locate the python-hip-kernels shared library."""
    lib_path = os.environ.get("PYTHON_HIP_KERNELS_LIBRARY_PATH")
    if lib_path and os.path.isfile(lib_path):
        return lib_path

    search_dirs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.append(script_dir)
    search_dirs.append(
        os.path.join(script_dir, "..", "..", "..", "..", "rocprofiler-sdk")
    )
    search_dirs.append(os.path.join(script_dir, "..", "..", "..", ".."))

    if "ROCPROFILER_SDK_TEST_LIB_DIR" in os.environ:
        search_dirs.append(os.environ["ROCPROFILER_SDK_TEST_LIB_DIR"])

    if "LD_LIBRARY_PATH" in os.environ:
        search_dirs.extend(os.environ["LD_LIBRARY_PATH"].split(":"))

    for d in search_dirs:
        candidate = os.path.join(d, "libpython-hip-kernels.so")
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)

    return None


def _load_library():
    """Load the python-hip-kernels shared library via ctypes."""
    global _lib

    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if lib_path is None:
        raise RuntimeError(
            "Could not find libpython-hip-kernels.so. "
            "Set PYTHON_HIP_KERNELS_LIBRARY_PATH or ROCPROFILER_SDK_TEST_LIB_DIR."
        )

    print(f"[torchesque.hip_kernels] Loading library: {lib_path}", file=sys.stderr)
    _lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)

    _lib.hip_kernels_create_streams.restype = ctypes.c_int
    _lib.hip_kernels_create_streams.argtypes = [ctypes.c_int]

    _lib.hip_kernels_launch.restype = ctypes.c_int
    _lib.hip_kernels_launch.argtypes = [ctypes.c_int]

    _lib.hip_kernels_synchronize.restype = ctypes.c_int
    _lib.hip_kernels_synchronize.argtypes = []

    _lib.hip_kernels_destroy_streams.restype = ctypes.c_int
    _lib.hip_kernels_destroy_streams.argtypes = []

    return _lib


class HipKernelRunner:
    """Manages HIP stream creation, kernel launches, and cleanup."""

    def __init__(self, num_streams):
        self._lib = _load_library()
        self._num_streams = num_streams
        self._created = False

    def create_streams(self):
        """Create HIP streams."""
        ret = self._lib.hip_kernels_create_streams(self._num_streams)
        if ret != 0:
            raise RuntimeError(
                f"hip_kernels_create_streams({self._num_streams}) failed with code {ret}"
            )
        self._created = True

    def launch_kernels(self):
        """Launch a trivial kernel on each stream."""
        if not self._created:
            self.create_streams()
        for i in range(self._num_streams):
            ret = self._lib.hip_kernels_launch(i)
            if ret != 0:
                raise RuntimeError(f"hip_kernels_launch({i}) failed with code {ret}")

    def synchronize(self):
        """Synchronize all streams."""
        ret = self._lib.hip_kernels_synchronize()
        if ret != 0:
            raise RuntimeError(f"hip_kernels_synchronize() failed with code {ret}")

    def destroy_streams(self):
        """Destroy all streams and free device memory."""
        if self._created:
            ret = self._lib.hip_kernels_destroy_streams()
            if ret != 0:
                raise RuntimeError(
                    f"hip_kernels_destroy_streams() failed with code {ret}"
                )
            self._created = False

    def __enter__(self):
        self.create_streams()
        return self

    def __exit__(self, *args):
        self.synchronize()
        self.destroy_streams()
