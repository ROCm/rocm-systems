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
"""Status-code to exception mapping unit tests for _check_res."""

from __future__ import annotations

import unittest

from common.common import amdsmi


class TestAmdSmiCheckRes(unittest.TestCase):
    def test_check_res(self):
        # Each status code maps to its dedicated exception, and that exception
        # exposes the originating code via get_error_code().
        cases = [
            (amdsmi.amdsmi_wrapper.AMDSMI_STATUS_RETRY, amdsmi.AmdSmiRetryException),
            (amdsmi.amdsmi_wrapper.AMDSMI_STATUS_TIMEOUT, amdsmi.AmdSmiTimeoutException),
            (amdsmi.amdsmi_wrapper.AMDSMI_STATUS_INVAL, amdsmi.AmdSmiLibraryException),
        ]
        for status, expected_exception in cases:
            with self.assertRaises(expected_exception) as ctx:
                amdsmi.amdsmi_interface._check_res(status)
            self.assertEqual(ctx.exception.get_error_code(), status)
        return
