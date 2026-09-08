#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
