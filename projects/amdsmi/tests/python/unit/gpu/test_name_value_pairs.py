#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name/value pair decoding unit tests (hardware-free).

Covers the buffer walk shared by amdsmi_get_gpu_pm_metrics_info() and
amdsmi_get_gpu_reg_table_info(). The library hands back rocm_smi's
rsmi_name_value_t records, allocated in blocks of 64, so the buffers here are
built the same way.
"""

import ctypes
import unittest

from common.common import amdsmi


class _RsmiNameValue(ctypes.Structure):
    # Mirrors rsmi_name_value_t in rocm_smi.h (MAX_RSMI_NAME_LENGTH is 64).
    _fields_ = [("name", ctypes.c_char * 64), ("value", ctypes.c_uint64)]


def _decode(pairs):
    """Decode *pairs* from a buffer laid out like the one the library returns."""
    capacity = max(64, -(-len(pairs) // 64) * 64)
    buffer = (_RsmiNameValue * capacity)()
    for i, (name, value) in enumerate(pairs):
        buffer[i].name = name.encode()
        buffer[i].value = value
    pointer = ctypes.cast(buffer, ctypes.POINTER(amdsmi.amdsmi_wrapper.amdsmi_name_value_t))
    return amdsmi.amdsmi_interface._get_name_value(ctypes.c_uint32(len(pairs)), pointer)


class TestAmdSmiNameValuePairs(unittest.TestCase):
    """Hardware-free unit tests for _get_name_value."""

    def test_round_trip_preserves_all_pairs(self):
        pairs = [("clk_gfxclk", 1500), ("clk_socclk", 1100), ("temp_hotspot", 62)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_full_pm_metrics_table_round_trips(self):
        # MI300 PM metrics return 259 records, more than the first 64-record block.
        pairs = [("metric[%d]" % i, (i + 1) * 0x0101010101) for i in range(259)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_max_length_name_round_trips(self):
        # Longest name the 64-byte field holds alongside its terminator.
        pairs = [("a" * 63, 0xFFFFFFFFFFFFFFFF), ("next", 7)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_zero_count_returns_empty(self):
        self.assertEqual(_decode([]), [])
