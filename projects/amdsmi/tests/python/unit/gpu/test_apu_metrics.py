#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""APU metrics helper unit tests (hardware-free)."""

from __future__ import annotations

import unittest

from common.common import amdsmi


class TestAmdSmiApuMetrics(unittest.TestCase):
    """Hardware-free unit tests for the APU metrics helpers."""

    def test_convert_apu_unit_na_passthrough(self):
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit("N/A", 100), "N/A")
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit("N/A", 1000), "N/A")

    def test_convert_apu_unit_scalar(self):
        # centidegrees -> C
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit(2500, 100), 25.0)
        # milliwatts -> W
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit(15000, 1000), 15.0)
        # result is rounded to 2 decimals
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit(2533, 100), 25.33)

    def test_convert_apu_unit_zero_is_valid(self):
        # 0 is a real value, not a sentinel
        self.assertEqual(amdsmi.amdsmi_interface._convert_apu_unit(0, 100), 0.0)

    def test_convert_apu_unit_list_mixed(self):
        self.assertEqual(
            amdsmi.amdsmi_interface._convert_apu_unit([2500, "N/A", 0], 100), [25.0, "N/A", 0.0]
        )

    def test_populate_apu_metrics_conversions_and_sentinel(self):
        apu = amdsmi.amdsmi_wrapper.struct_amdsmi_apu_metrics_t()
        apu.temperature_gfx = 0xFFFF  # UINT16 sentinel -> N/A
        apu.temperature_soc = 2500  # centidegrees -> 25.0 C
        apu.average_socket_power = 15000  # milliwatts -> 15.0 W
        apu.average_gfxclk_frequency = 0  # 0 is a valid MHz value

        result = amdsmi.amdsmi_interface._populate_apu_metrics(apu)

        self.assertEqual(result["apu_metrics.temperature_gfx"], "N/A")
        self.assertEqual(result["apu_metrics.temperature_soc"], 25.0)
        self.assertEqual(result["apu_metrics.average_socket_power"], 15.0)
        self.assertEqual(result["apu_metrics.average_gfxclk_frequency"], 0)

    def test_na_dict_symmetry(self):
        na_dict_keys = {
            k
            for k in amdsmi.amdsmi_interface._apu_metrics_na_dict()
            if k.startswith("apu_metrics.")
        }
        na_full_keys = {
            k
            for k in amdsmi.amdsmi_interface._NA_amdsmi_get_gpu_metrics_info()
            if k.startswith("apu_metrics.")
        }
        # The live is_apu=True path must expose the same apu_metrics.* keys.
        live_keys = {
            k
            for k in amdsmi.amdsmi_interface._populate_apu_metrics(
                amdsmi.amdsmi_wrapper.struct_amdsmi_apu_metrics_t()
            )
            if k.startswith("apu_metrics.")
        }
        self.assertEqual(na_dict_keys, na_full_keys)
        self.assertEqual(na_dict_keys, live_keys)


if __name__ == "__main__":
    unittest.main()
