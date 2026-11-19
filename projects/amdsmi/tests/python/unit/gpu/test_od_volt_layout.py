#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overdrive voltage/frequency decoding (hardware-free).

The library fills the public amdsmi_od_volt_freq_data_t and
amdsmi_freq_volt_region_t layouts, so the stubs here write the wrapper's own
classes for those structs.
"""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_MHZ = 1000000
_wrapper = amdsmi.amdsmi_wrapper


class TestOdVoltLayout(unittest.TestCase):
    """The wrapper's classes have the layout amdsmi.h gives these structs."""

    def test_sizes_and_offsets(self):
        # Six 32-byte ranges (two reserved words each), the 48-byte curve, then
        # num_regions and 4 bytes of tail padding.
        freq_data = _wrapper.amdsmi_od_volt_freq_data_t
        self.assertEqual(ctypes.sizeof(freq_data), 248)
        self.assertEqual(freq_data.curr_mclk_range.offset, 32)
        self.assertEqual(freq_data.sclk_freq_limits.offset, 96)
        self.assertEqual(freq_data.curve.offset, 192)
        self.assertEqual((freq_data.num_regions.offset, freq_data.num_regions.size), (240, 4))
        self.assertEqual(ctypes.sizeof(_wrapper.amdsmi_freq_volt_region_t), 64)


class TestOdVoltInfo(unittest.TestCase):
    """Each key reports its own field."""

    def test_sclk_and_fclk_only(self):
        # An MI350X reports OD_SCLK and OD_FCLK but no OD_MCLK.
        handle = _wrapper.amdsmi_processor_handle()

        def _stub(_handle, data_ptr):
            data = data_ptr._obj
            for name in ("curr_sclk_range", "curr_mclk_range", "curr_fclk_range"):
                getattr(data, name).lower_bound = getattr(data, name).upper_bound = _UINT64_MAX
            data.curr_sclk_range.lower_bound = 500 * _MHZ
            data.curr_sclk_range.upper_bound = 2200 * _MHZ
            data.curr_fclk_range.lower_bound = 1250 * _MHZ
            data.curr_fclk_range.upper_bound = 1500 * _MHZ
            return 0

        with mock.patch.object(_wrapper, "amdsmi_get_gpu_od_volt_info", _stub):
            info = amdsmi.amdsmi_interface.amdsmi_get_gpu_od_volt_info(handle)
        self.assertEqual(
            info["curr_sclk_range"], {"lower_bound": 500 * _MHZ, "upper_bound": 2200 * _MHZ}
        )
        self.assertEqual(info["curr_mclk_range"], {"lower_bound": "N/A", "upper_bound": "N/A"})
        self.assertEqual(info["sclk_freq_limits"], {"lower_bound": 0, "upper_bound": 0})
        self.assertEqual(info["num_regions"], 0)


class TestOdVoltCurveRegions(unittest.TestCase):
    """Records are read in order."""

    _WRITTEN = [((100, 200), (300, 400)), ((500, 600), (700, 800))]

    def _regions(self, reported_count):
        handle = _wrapper.amdsmi_processor_handle()

        def _stub(_handle, count_ptr, buffer):
            for region, (freq, volt) in zip(buffer, self._WRITTEN):
                region.freq_range.lower_bound, region.freq_range.upper_bound = freq
                region.volt_range.lower_bound, region.volt_range.upper_bound = volt
            count_ptr._obj.value = reported_count
            return 0

        with mock.patch.object(_wrapper, "amdsmi_get_gpu_od_volt_curve_regions", _stub):
            return amdsmi.amdsmi_interface.amdsmi_get_gpu_od_volt_curve_regions(
                handle, len(self._WRITTEN)
            )

    _EXPECTED = [
        {
            "freq_range": {"lower_bound": freq[0], "upper_bound": freq[1]},
            "volt_range": {"lower_bound": volt[0], "upper_bound": volt[1]},
        }
        for freq, volt in _WRITTEN
    ]

    def test_two_regions(self):
        self.assertEqual(self._regions(2), self._EXPECTED)
