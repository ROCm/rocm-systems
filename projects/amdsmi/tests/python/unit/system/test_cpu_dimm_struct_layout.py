#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU DIMM power and thermal struct layout (hardware-free).

The generated ctypes structs must keep the C layout on every supported Python;
Python 3.14 lays out bit-fields with MSVC rules.
"""

import ctypes
import struct
import unittest

from common.common import amdsmi


class TestDimmStructLayout(unittest.TestCase):
    """Decode byte images laid out as the C compiler lays out these structs."""

    def test_power(self):
        power_t = amdsmi.amdsmi_wrapper.struct_amdsmi_dimm_power_t
        self.assertEqual(ctypes.sizeof(power_t), 6)
        # power:15 in the first 16-bit unit, update_rate:9 in the second, dimm_addr at byte 4.
        image = struct.pack("<HHBx", 0x1234, 0x155, 0xAB)
        record = power_t.from_buffer_copy(image)
        self.assertEqual(
            (record.power, record.update_rate, record.dimm_addr), (0x1234, 0x155, 0xAB)
        )

    def test_thermal(self):
        thermal_t = amdsmi.amdsmi_wrapper.struct_amdsmi_dimm_thermal_t
        self.assertEqual(ctypes.sizeof(thermal_t), 12)
        self.assertEqual(thermal_t.temp.offset, 8)
        image = struct.pack("<HHB3xf", 0x3FF, 0x155, 0xAB, 42.5)
        record = thermal_t.from_buffer_copy(image)
        self.assertEqual(
            (record.sensor, record.update_rate, record.dimm_addr, record.temp),
            (0x3FF, 0x155, 0xAB, 42.5),
        )
