#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name/value pair decoding unit tests (hardware-free).

Covers the buffer walk shared by amdsmi_get_gpu_pm_metrics_info() and
amdsmi_get_gpu_reg_table_info().
"""

from __future__ import annotations

import ctypes
import unittest

from common.common import amdsmi


def _build_buffer(pairs):
    """Return a POINTER(amdsmi_name_value_t) over *pairs*, as the C library hands back."""
    name_value_t = amdsmi.amdsmi_wrapper.struct_amdsmi_name_value_t
    buffer = (name_value_t * len(pairs))()
    for i, (name, value) in enumerate(pairs):
        buffer[i].name = name.encode()
        buffer[i].value = value
    return ctypes.cast(buffer, ctypes.POINTER(name_value_t))


class TestAmdSmiNameValuePairs(unittest.TestCase):
    """Hardware-free unit tests for _get_name_value."""

    def test_round_trip_preserves_all_pairs(self):
        # Every entry must survive, not just the first: a stride that does not
        # match sizeof(amdsmi_name_value_t) silently blanks later entries.
        pairs = [("clk_gfxclk", 1500), ("clk_socclk", 1100), ("temp_hotspot", 62)]
        result = amdsmi.amdsmi_interface._get_name_value(
            ctypes.c_uint32(len(pairs)), _build_buffer(pairs)
        )
        self.assertEqual(result, [{"name": name, "value": value} for name, value in pairs])

    def test_max_length_name_round_trips(self):
        # Longest name the fixed-size char array can hold alongside its terminator.
        name = "a" * (amdsmi.amdsmi_interface.AMDSMI_MAX_STRING_LENGTH - 1)
        result = amdsmi.amdsmi_interface._get_name_value(
            ctypes.c_uint32(1), _build_buffer([(name, 0xFFFFFFFFFFFFFFFF)])
        )
        self.assertEqual(result, [{"name": name, "value": 0xFFFFFFFFFFFFFFFF}])

    def test_zero_count_returns_empty(self):
        result = amdsmi.amdsmi_interface._get_name_value(ctypes.c_uint32(0), _build_buffer([]))
        self.assertEqual(result, [])
