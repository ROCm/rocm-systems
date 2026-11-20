#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name/value pair decoding unit tests (hardware-free).

Covers the buffer walk shared by amdsmi_get_gpu_pm_metrics_info() and
amdsmi_get_gpu_reg_table_info(). The library returns an array of the
amdsmi_name_value_t records declared in amdsmi.h, so the buffers here are
built the same way. What the library itself returns is checked by
tests/amd_smi_test/unit/gpu/name_value_info_test.cc.
"""

import ctypes
import unittest

from common.common import amdsmi

_NameValue = amdsmi.amdsmi_wrapper.amdsmi_name_value_t


def _decode(pairs):
    """Decode *pairs* from a buffer laid out like the one the library returns."""
    buffer = (_NameValue * max(1, len(pairs)))()
    for i, (name, value) in enumerate(pairs):
        buffer[i].name = name.encode()
        buffer[i].value = value
    pointer = ctypes.cast(buffer, ctypes.POINTER(_NameValue))
    return amdsmi.amdsmi_interface._get_name_value(ctypes.c_uint32(len(pairs)), pointer)


class TestAmdSmiNameValuePairs(unittest.TestCase):
    """Hardware-free unit tests for _get_name_value."""

    def test_record_layout_matches_amdsmi_h(self):
        # char name[AMDSMI_MAX_STRING_LENGTH] (256) followed by a uint64_t value.
        self.assertEqual(ctypes.sizeof(_NameValue), 264)
        self.assertEqual(_NameValue.value.offset, 256)

    def test_round_trip_preserves_all_pairs(self):
        pairs = [("clk_gfxclk", 1500), ("clk_socclk", 1100), ("temp_hotspot", 62)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_full_pm_metrics_table_round_trips(self):
        # A table of 259 records.
        pairs = [("metric[%d]" % i, (i + 1) * 0x0101010101) for i in range(259)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_max_length_name_round_trips(self):
        # rocm_smi names are at most 63 characters; the public field holds 255.
        pairs = [("a" * 63, 0xFFFFFFFFFFFFFFFF), ("b" * 255, 7), ("next", 8)]
        self.assertEqual(_decode(pairs), [{"name": n, "value": v} for n, v in pairs])

    def test_zero_count_returns_empty(self):
        self.assertEqual(_decode([]), [])
