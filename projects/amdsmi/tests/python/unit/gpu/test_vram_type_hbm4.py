#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HBM VRAM type enum values (hardware-free)."""

import unittest

from common.common import amdsmi


class TestVramTypeHbm(unittest.TestCase):
    """Pins the HBM3E/HBM4 values across the wrapper and the Python enum."""

    def test_hbm3e(self):
        self.assertEqual(amdsmi.amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM3E, 5)
        self.assertEqual(
            amdsmi.amdsmi_wrapper.amdsmi_vram_type_t__enumvalues[5], "AMDSMI_VRAM_TYPE_HBM3E"
        )
        self.assertEqual(amdsmi.amdsmi_interface.AmdSmiVramType.HBM3E, 5)

    def test_hbm4(self):
        self.assertEqual(amdsmi.amdsmi_wrapper.AMDSMI_VRAM_TYPE_HBM4, 6)
        self.assertEqual(
            amdsmi.amdsmi_wrapper.amdsmi_vram_type_t__enumvalues[6], "AMDSMI_VRAM_TYPE_HBM4"
        )
        self.assertEqual(amdsmi.amdsmi_interface.AmdSmiVramType.HBM4, 6)


if __name__ == "__main__":
    unittest.main()
