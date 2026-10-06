#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Overdrive voltage/frequency decoding (hardware-free).

amdsmi_get_gpu_od_volt_info() and amdsmi_get_gpu_od_volt_curve_regions() let
rocm_smi fill the caller's buffer in its own layout (16-byte ranges), so the
stubs here write that layout, as the library does.
"""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_MHZ = 1000000


# Mirrors rsmi_range_t, rsmi_freq_volt_region_t and rsmi_od_volt_freq_data_t in rocm_smi.h.
class _Range(ctypes.Structure):
    _fields_ = [("lower_bound", ctypes.c_uint64), ("upper_bound", ctypes.c_uint64)]


class _Region(ctypes.Structure):
    _fields_ = [("freq_range", _Range), ("volt_range", _Range)]


class _FreqData(ctypes.Structure):
    _fields_ = [
        ("curr_sclk_range", _Range),
        ("curr_mclk_range", _Range),
        ("curr_fclk_range", _Range),
        ("sclk_freq_limits", _Range),
        ("mclk_freq_limits", _Range),
        ("fclk_freq_limits", _Range),
        ("vc_points", ctypes.c_uint64 * 6),
        ("num_regions", ctypes.c_uint32),
    ]


def _od_volt_info_with(**ranges) -> dict:
    """Return the public dict with a stub library that writes *ranges* (lower, upper)."""
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _stub(_handle, data_ptr):
        data = _FreqData.from_buffer(data_ptr._obj)
        for name in ("curr_sclk_range", "curr_mclk_range", "curr_fclk_range"):
            getattr(data, name).lower_bound = getattr(data, name).upper_bound = _UINT64_MAX
        for name, (lower, upper) in ranges.items():
            getattr(data, name).lower_bound = lower
            getattr(data, name).upper_bound = upper
        return 0

    with mock.patch.object(amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_od_volt_info", _stub):
        return amdsmi.amdsmi_interface.amdsmi_get_gpu_od_volt_info(handle)


class TestOdVoltLayout(unittest.TestCase):
    """Pins the rocm_smi layout the decoder relies on."""

    def test_sizes_and_offsets(self):
        self.assertEqual(ctypes.sizeof(_FreqData), 152)
        self.assertEqual(_FreqData.curr_mclk_range.offset, 16)
        self.assertEqual(_FreqData.sclk_freq_limits.offset, 48)
        self.assertEqual(_FreqData.num_regions.offset, 144)
        self.assertEqual(ctypes.sizeof(_Region), 32)


class TestOdVoltInfo(unittest.TestCase):
    """Each key reports its own clock domain."""

    def test_sclk_and_fclk_only(self):
        # An MI350X reports OD_SCLK and OD_FCLK but no OD_MCLK: FCLK must not show up as MCLK.
        info = _od_volt_info_with(
            curr_sclk_range=(500 * _MHZ, 2200 * _MHZ), curr_fclk_range=(1250 * _MHZ, 1500 * _MHZ)
        )
        self.assertEqual(
            info["curr_sclk_range"], {"lower_bound": 500 * _MHZ, "upper_bound": 2200 * _MHZ}
        )
        self.assertEqual(info["curr_mclk_range"], {"lower_bound": "N/A", "upper_bound": "N/A"})
        self.assertEqual(info["sclk_freq_limits"], {"lower_bound": 0, "upper_bound": 0})
        self.assertEqual(info["mclk_freq_limits"], {"lower_bound": 0, "upper_bound": 0})
        self.assertEqual(info["num_regions"], 0)

    def test_ranges_and_limits(self):
        info = _od_volt_info_with(
            curr_sclk_range=(500 * _MHZ, 2500 * _MHZ),
            curr_mclk_range=(97 * _MHZ, 1000 * _MHZ),
            sclk_freq_limits=(500 * _MHZ, 3150 * _MHZ),
            mclk_freq_limits=(674 * _MHZ, 1200 * _MHZ),
        )
        self.assertEqual(
            info["curr_mclk_range"], {"lower_bound": 97 * _MHZ, "upper_bound": 1000 * _MHZ}
        )
        self.assertEqual(
            info["sclk_freq_limits"], {"lower_bound": 500 * _MHZ, "upper_bound": 3150 * _MHZ}
        )
        self.assertEqual(
            info["mclk_freq_limits"], {"lower_bound": 674 * _MHZ, "upper_bound": 1200 * _MHZ}
        )


class TestOdVoltCurveRegions(unittest.TestCase):
    """Records are read at rocm_smi's 32-byte spacing."""

    def test_two_regions(self):
        handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()
        written = [((100, 200), (300, 400)), ((500, 600), (700, 800))]

        def _stub(_handle, count_ptr, buffer):
            regions = (_Region * len(written)).from_buffer(buffer)
            for region, (freq, volt) in zip(regions, written):
                region.freq_range.lower_bound, region.freq_range.upper_bound = freq
                region.volt_range.lower_bound, region.volt_range.upper_bound = volt
            count_ptr._obj.value = len(written)
            return 0

        with mock.patch.object(
            amdsmi.amdsmi_wrapper, "amdsmi_get_gpu_od_volt_curve_regions", _stub
        ):
            result = amdsmi.amdsmi_interface.amdsmi_get_gpu_od_volt_curve_regions(handle, 2)
        self.assertEqual(
            result,
            [
                {
                    "freq_range": {"lower_bound": freq[0], "upper_bound": freq[1]},
                    "volt_range": {"lower_bound": volt[0], "upper_bound": volt[1]},
                }
                for freq, volt in written
            ],
        )


if __name__ == "__main__":
    unittest.main()
