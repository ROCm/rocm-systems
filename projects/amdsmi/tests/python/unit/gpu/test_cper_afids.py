#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amdsmi_get_afids_from_cper input-normalization unit tests (hardware-free)."""

from __future__ import annotations

import unittest

from common.common import amdsmi


class TestAmdSmiGetAfidsFromCper(unittest.TestCase):
    """Pins the list branch, unreachable while it was guarded by the invalid
    ``isinstance(data, List[Dict[str, Any]])``."""

    def test_empty_list_short_circuits_without_library_call(self):
        afids, count = amdsmi.amdsmi_interface.amdsmi_get_afids_from_cper([])
        self.assertEqual(afids, [])
        self.assertEqual(count, 0)

    def test_invalid_type_raises_parameter_exception(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_afids_from_cper(42)

    def test_malformed_record_raises_parameter_exception(self):
        with self.assertRaises(amdsmi.AmdSmiParameterException):
            amdsmi.amdsmi_interface.amdsmi_get_afids_from_cper([{"missing": "keys"}])


if __name__ == "__main__":
    unittest.main()
