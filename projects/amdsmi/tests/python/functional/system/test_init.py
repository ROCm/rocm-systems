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

import unittest

import common.common as common

# common.common owns path resolution, sys.path setup, and amdsmi loading — borrow the
# reference so AMDSMI_PATH/ROCM_HOME/ROCM_PATH resolution and the stale-package check
# (see ROCM-1552 / PR #6359) are not duplicated or bypassed here.
from common.common import amdsmi


class TestAmdSmiInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = common.Common(common.verbose)
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
