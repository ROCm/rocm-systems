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

    def test_every_status_has_its_own_message(self):
        # A copy-pasted entry once gave CORRUPTED_EEPROM the SETTING_UNAVAILABLE text.
        owners = {}
        for code, name in amdsmi.amdsmi_wrapper.amdsmi_status_t__enumvalues.items():
            if code == amdsmi.amdsmi_wrapper.AMDSMI_STATUS_SUCCESS:
                continue
            info = amdsmi.AmdSmiLibraryException(code).get_error_info()
            self.assertTrue(info.startswith(f"{name} - "), info)
            text = info.split(" - ", 1)[1]
            self.assertNotIn(text, owners, f"{name} reuses the message of {owners.get(text)}")
            owners[text] = name
