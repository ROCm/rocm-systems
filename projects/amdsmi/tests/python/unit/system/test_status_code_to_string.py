#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hardware-free tests for amdsmi_status_code_to_string().

The call only translates a number into text, so it needs no GPU and no amdsmi_init().
The functional suite (functional/gpu/test_system.py) runs the same checks on a device.
"""

import ctypes
import unittest

from common.common import Common, amdsmi


class TestStatusCodeToString(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = Common.for_unit_tests()

    def test_status_code_to_string(self):
        self.common.print_func_name("")
        # Every status code, including the two sentinels, must describe itself with its own
        # enum name. UNKNOWN_ERROR means a `case` is missing from
        # amdsmi_status_code_to_string(); MAP_ERROR means a lower-level status has no mapping.
        accept = [amdsmi.AmdSmiStatus.SUCCESS]
        wrong_names = []
        with self.common.status_sweep():
            for status in amdsmi.AmdSmiStatus:
                error_name = f"AMDSMI_STATUS_{status.name}"
                msg = f"\t### amdsmi_status_code_to_string({error_name}={status.value}):"
                data = None
                with self.common.expect_status(msg, accept):
                    data = amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(status.value))
                if data is None:
                    continue  # the sweep already recorded the failed status

                description = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                self.common.print(f"\t\t{error_name}: {description}")
                self.common.print("")
                if not description.startswith(error_name):
                    wrong_names.append(f"{error_name}: '{description}'")

        self.assertEqual(
            wrong_names,
            [],
            "Descriptions that do not start with their own AMDSMI_STATUS_* name:\n  "
            + "\n  ".join(wrong_names),
        )

    def test_undefined_code(self):
        self.common.print_func_name("")
        # 100 sits in an enum gap: no case and no lower-level mapping, so it must fall
        # through to UNKNOWN_ERROR. SUCCESS is not accepted, since that would mean an
        # undefined code resolved to a string.
        msg = "\t### amdsmi_status_code_to_string(status=100):"
        with self.common.expect_status(msg, [amdsmi.AmdSmiStatus.UNKNOWN_ERROR]):
            amdsmi.amdsmi_status_code_to_string(ctypes.c_uint32(100))

    def test_rejects_non_status(self):
        self.common.print_func_name("")
        # The Python wrapper rejects a non-status before any library call, so there is no
        # status to judge, only the exception.
        msg = '\t### amdsmi_status_code_to_string(status="not_a_status"):'
        with self.common.expect_status(msg, raises=amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_status_code_to_string("not_a_status")

        # Confirms it is this isinstance check that fired, not another
        # AmdSmiParameterException raised for a different reason.
        with self.assertRaises(amdsmi.AmdSmiParameterException) as cm:
            amdsmi.amdsmi_status_code_to_string("not_a_status")
        self.assertEqual(cm.exception.expectedType, amdsmi.amdsmi_wrapper.amdsmi_status_t)
