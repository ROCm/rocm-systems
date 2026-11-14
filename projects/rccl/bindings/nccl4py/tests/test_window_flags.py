# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``WindowFlag.GIN_ONLY`` (NCCL 2.32) reaches RCCL's window registration.

RCCL's ``ncclCommWindowRegister`` returns ``ncclInvalidArgument`` when
``NCCL_WIN_COLL_SYMMETRIC`` is combined with a window capability-restriction
flag, and ``NCCL_WIN_GIN_ONLY`` is the only one. That check comes before any
registration work, and ``NCCL_WIN_ENABLE=0`` makes RCCL skip the registration
of a window that gets past it, so the window address is never dereferenced.

The flags come from this tree's ``nccl/core/constants.py``. Their ``value`` goes
to a one-rank librccl communicator through ctypes, as ``RegisteredWindowHandle``
passes it to ``ncclCommWindowRegister``. Neither the Cython extensions nor
hip-python is needed, so this runs under pytest, or as a script that exits 1 on
a failure and 77 when skipped::

    RCCL_BUILD=/path/to/build/release python3 test_window_flags.py -v
"""

import ctypes
import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ADDR = 0x1000  # never dereferenced
_SUCCESS, _INVALID_ARGUMENT = 0, 4  # ncclResult_t


def _window_flags():
    path = os.path.join(_HERE, os.pardir, "nccl", "core", "constants.py")
    spec = importlib.util.spec_from_file_location("_nccl4py_constants", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WindowFlag


def _librccl():
    build = os.environ.get("RCCL_BUILD") or "build/release"
    rccl = os.path.join(_HERE, os.pardir, os.pardir, os.pardir)
    for path in (
        os.environ.get("NCCL_LIBRARY"),
        os.path.join(build, "librccl.so"),
        os.path.join(rccl, build, "librccl.so"),
    ):
        if path and os.path.exists(path):
            return ctypes.CDLL(path)
    raise unittest.SkipTest("librccl.so not found: set NCCL_LIBRARY or RCCL_BUILD")


class WindowFlagTest(unittest.TestCase):
    def setUp(self):
        self.flags = _window_flags()
        os.environ["NCCL_WIN_ENABLE"] = "0"  # before RCCL first reads (and caches) it
        self.lib = _librccl()
        count = ctypes.c_int(0)
        if self.lib.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value < 1:
            self.skipTest("no GPU")
        self.comm = ctypes.c_void_p()
        self.assertEqual(self.lib.ncclCommInitAll(ctypes.byref(self.comm), 1, None), _SUCCESS)
        self.addCleanup(self.lib.ncclCommDestroy, self.comm)

    def register(self, flags, size=4096):
        register = self.lib.ncclCommWindowRegister
        register.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
        ]
        win = ctypes.c_void_p()
        return register(self.comm, _ADDR, size, ctypes.byref(win), flags.value)

    def test_control_window_register_error(self):
        self.assertEqual(self.register(self.flags.COLL_SYMMETRIC), _SUCCESS)
        self.assertEqual(self.register(self.flags.COLL_SYMMETRIC, size=0), _INVALID_ARGUMENT)

    def test_gin_only_window_flag_reaches_rccl(self):
        gin_only = getattr(self.flags, "GIN_ONLY", None)
        self.assertIsNotNone(gin_only, "nccl.core.WindowFlag has no GIN_ONLY")
        self.assertEqual(self.register(gin_only), _SUCCESS)
        self.assertEqual(self.register(self.flags.COLL_SYMMETRIC | gin_only), _INVALID_ARGUMENT)


if __name__ == "__main__":
    result = unittest.main(exit=False).result
    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(77 if result.skipped or not result.testsRun else 0)
