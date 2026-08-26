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
"""CPU clock, pstate and link-width APIs."""

import unittest

import common.api_test as api
import common.common as common

# amdsmi_get_cpu_current_io_bandwidth()/amdsmi_get_cpu_current_xgmi_bw() take a
# link-ID string and a bandwidth-type int; IO_BW_ENCODINGS pairs them up.
_LINK_NAME, _BW_ENCODING = common.IO_BW_ENCODINGS[0][0], common.IO_BW_ENCODINGS[0][1]


class TestCpuClock(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_fclk_mclk(self):
        self.both("amdsmi_get_cpu_fclk_mclk", self.handle)

    def test_get_cpu_cclk_limit(self):
        self.both("amdsmi_get_cpu_cclk_limit", self.handle)

    def test_get_cpu_socket_current_active_freq_limit(self):
        self.both("amdsmi_get_cpu_socket_current_active_freq_limit", self.handle)

    def test_get_cpu_socket_freq_range(self):
        self.both("amdsmi_get_cpu_socket_freq_range", self.handle)

    def test_get_cpu_core_current_freq_limit(self):
        self.both("amdsmi_get_cpu_core_current_freq_limit", self.handle)

    def test_get_cpu_freq_range(self):
        self.expect_only("amdsmi_get_cpu_freq_range")

    def test_get_cpu_core_floor_freq_limit(self):
        self.both("amdsmi_get_cpu_core_floor_freq_limit", self.handle)

    def test_get_cpu_floor_freq_limit(self):
        self.both("amdsmi_get_cpu_floor_freq_limit", self.handle)

    def test_get_cpu_core_eff_floor_freq_limit(self):
        self.both("amdsmi_get_cpu_core_eff_floor_freq_limit", self.handle)

    def test_get_cpu_eff_floor_freq_limit(self):
        self.both("amdsmi_get_cpu_eff_floor_freq_limit", self.handle)

    def test_get_cpu_socket_lclk_dpm_level(self):
        self.both("amdsmi_get_cpu_socket_lclk_dpm_level", self.handle, api.integer("nbio_id", 0))

    def test_get_cpu_xgmi_pstate_range(self):
        self.both("amdsmi_get_cpu_xgmi_pstate_range", self.handle)

    def test_get_cpu_current_io_bandwidth(self):
        self.both(
            "amdsmi_get_cpu_current_io_bandwidth",
            self.handle,
            api.integer("encoding", _BW_ENCODING),
            api.text("link_name", _LINK_NAME),
        )

    def test_get_cpu_current_xgmi_bw(self):
        self.both(
            "amdsmi_get_cpu_current_xgmi_bw",
            self.handle,
            api.integer("encoding", _BW_ENCODING),
            api.text("link_name", _LINK_NAME),
        )

    def test_set_cpu_core_floor_freq_limit(self):
        self.reject_only(
            "amdsmi_set_cpu_core_floor_freq_limit", self.handle, api.integer("floorlimit", 0)
        )

    def test_set_cpu_floor_freq_limit(self):
        self.reject_only(
            "amdsmi_set_cpu_floor_freq_limit", self.handle, api.integer("floorlimit", 0)
        )

    def test_set_cpu_msr_floor_freq_limit(self):
        self.reject_only(
            "amdsmi_set_cpu_msr_floor_freq_limit", self.handle, api.integer("msrfloorlimit", 0)
        )

    def test_set_cpu_core_msr_floor_freq_limit(self):
        self.reject_only(
            "amdsmi_set_cpu_core_msr_floor_freq_limit", self.handle, api.integer("msrfloorlimit", 0)
        )

    def test_cpu_apb_enable(self):
        self.reject_only("amdsmi_cpu_apb_enable", self.handle)

    def test_cpu_apb_disable(self):
        self.reject_only("amdsmi_cpu_apb_disable", self.handle, api.integer("pstate", 0))

    def test_set_cpu_socket_lclk_dpm_level(self):
        self.reject_only(
            "amdsmi_set_cpu_socket_lclk_dpm_level",
            self.handle,
            api.integer("nbio_id", 0),
            api.integer("min_val", 0),
            api.integer("max_val", 0),
        )

    def test_set_cpu_pcie_link_rate(self):
        self.reject_only("amdsmi_set_cpu_pcie_link_rate", self.handle, api.integer("rate_ctrl", 0))

    def test_set_cpu_df_pstate_range(self):
        self.reject_only(
            "amdsmi_set_cpu_df_pstate_range",
            self.handle,
            api.integer("min_pstate", 0),
            api.integer("max_pstate", 0),
        )

    def test_set_cpu_xgmi_pstate_range(self):
        self.reject_only(
            "amdsmi_set_cpu_xgmi_pstate_range",
            self.handle,
            api.integer("min_pstate", 0),
            api.integer("max_pstate", 0),
        )

    def test_set_cpu_xgmi_width(self):
        self.reject_only(
            "amdsmi_set_cpu_xgmi_width",
            self.handle,
            api.integer("min_width", 0),
            api.integer("max_width", 0),
        )

    def test_set_cpu_gmi3_link_width_range(self):
        self.reject_only(
            "amdsmi_set_cpu_gmi3_link_width_range",
            self.handle,
            api.integer("min_link_width", 0),
            api.integer("max_link_width", 0),
        )


if __name__ == "__main__":
    unittest.main()
