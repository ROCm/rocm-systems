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
"""System initialization and shutdown functional tests."""

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

amdsmi_path = os.environ.get("AMDSMI_PATH", "/opt/rocm/share/amd_smi")
if not os.path.exists(amdsmi_path):
    raise FileNotFoundError(
        f'AMDSMI_PATH "{amdsmi_path}" does not exist. '
        "Please set the correct path in your environment."
    )
sys.path.append(amdsmi_path)
try:
    import amdsmi
except ImportError as exc:
    raise ImportError(f"Could not import {amdsmi_path}") from exc

verbose = common.VERBOSITY_NORMAL
if "-q" in sys.argv or "--quiet" in sys.argv:
    verbose = common.VERBOSITY_QUIET
elif any(a in ("-v", "-vv", "--verbose") for a in sys.argv):
    verbose = common.VERBOSITY_VERBOSE


class TestAmdSmiInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(verbose)
        return

    def test_init_shutdown(self):
        self.common.print(f"## test_init_shutdown()")

        msg = f"\t### amdsmi_init():"
        try:
            ret = self.common.amdsmi_smart_init()[0]
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e

        msg = f"\t### amdsmi_shut_down():"
        try:
            ret = amdsmi.amdsmi_shut_down()
            self.common.print(msg, ret)
        except amdsmi.AmdSmiLibraryException as e:
            self.common.print(msg, e)
            raise e
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
