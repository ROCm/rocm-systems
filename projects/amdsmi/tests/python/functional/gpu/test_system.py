#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""GPU system operations: init, shutdown, reset, status codes, socket handles."""

import os
import sys
import unittest

# Allow direct execution: walk up to find the 'common/' package root
import os as _os

_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, "common")):
    _d = _os.path.dirname(_d)
sys.path.insert(0, _d)
del _d, _os

import common.helpers as common

# common.helpers owns path resolution, sys.path setup, and amdsmi loading — borrow the
# reference so AMDSMI_PATH/ROCM_HOME/ROCM_PATH resolution and the stale-package check
# (see ROCM-1552 / PR #6359) are not duplicated or bypassed here.
from common.helpers import amdsmi

verbose = common.VERBOSITY_NORMAL
if "-q" in sys.argv or "--quiet" in sys.argv:
    verbose = common.VERBOSITY_QUIET
elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
    verbose = common.VERBOSITY_VERBOSE


class TestGpuSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)

    @classmethod
    def tearDownClass(cls):
        try:
            amdsmi.amdsmi_shut_down()
        except amdsmi.AmdSmiLibraryException:
            pass

    def setUp(self):
        self.raise_exception = None
        self.common.amdsmi_smart_init()
        self.common.processors = amdsmi.amdsmi_get_processor_handles()

    def tearDown(self):
        amdsmi.amdsmi_shut_down()

    def test_get_socket_handles(self):
        self.common.print_func_name("")
        msg = "\t### amdsmi_get_socket_handles():"
        try:
            ret = amdsmi.amdsmi_get_socket_handles()
            self.common.print(msg, [id(addr) for addr in ret])
            self.common.check_ret("", "", self.common.PASS)
        except amdsmi.AmdSmiLibraryException as e:
            if self.common.check_ret(msg, e, self.common.PASS):
                self.raise_exception = e
        self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return

    def test_init(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_init=amdsmi.amdsmi_init)
        return

    def test_reset_gpu(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_reset_gpu as it fails (MI350X, Hang)."
            self.common.print(msg)
            self.skipTest(msg)

        self.common.Test_API_Per_GPU(amdsmi_reset_gpu=amdsmi.amdsmi_reset_gpu)
        return

    def test_reset_gpu_fan(self):
        self.common.print_func_name("")
        self.common.Test_API_Per_GPU(amdsmi_reset_gpu_fan=amdsmi.amdsmi_reset_gpu_fan, index=0)
        return

    def test_shut_down(self):
        self.common.print_func_name("")
        self.common.Test_API(amdsmi_shut_down=amdsmi.amdsmi_shut_down)
        return

    def test_status_code_to_string(self):
        self.common.print_func_name("")

        if self.common.TODO_SKIP_FAIL:
            msg = "\tSkipping test_status_code_to_string as it fails (Unhashable type)."
            self.common.print(msg)
            self.skipTest(msg)

        for error_num, _ in self.common.error_map.items():
            msg = f"\t### amdsmi_status_code_to_string(error_num={error_num}):"
            try:
                ret = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(int(error_num, 0)))
                self.common.print(msg, ret)
            except amdsmi.AmdSmiLibraryException as e:
                if self.common.check_ret(msg, e, self.common.PASS):
                    self.raise_exception = e
            self.common.print("")
        if self.raise_exception:
            raise self.raise_exception
        return


if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Warning: Some tests may require elevated privileges (sudo/root).\n", file=sys.stderr)
    verbose = common.VERBOSITY_NORMAL
    if "-q" in sys.argv or "--quiet" in sys.argv:
        verbose = common.VERBOSITY_QUIET
    elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
        verbose = common.VERBOSITY_VERBOSE
    if not ("-k" in sys.argv or "--keyword" in sys.argv):
        if verbose > common.VERBOSITY_QUIET:
            common.print_tests(__name__)
    if "-h" in sys.argv or "--help" in sys.argv:
        unittest.main()
    if verbose < common.VERBOSITY_VERBOSE:
        common.print_legend()
    runner = unittest.TextTestRunner(
        stream=sys.stderr, verbosity=common.make_runner_verbosity(verbose)
    )
    common.expand_glob_k_arg(globals())
    unittest.main(testRunner=runner)
