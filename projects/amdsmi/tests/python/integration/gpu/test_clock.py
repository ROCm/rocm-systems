#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU clock and pstate APIs: bad arguments are rejected, good reads are valid."""

import unittest

import common.api_test as api
import common.common as common


class TestGpuClock(api.ApiTestCase):
    def test_get_clk_freq(self):
        self.both("amdsmi_get_clk_freq", self.handle, api.enum("clk_type", common.CLK_TYPES))

    def test_get_clock_info(self):
        self.both("amdsmi_get_clock_info", self.handle, api.enum("clock_type", common.CLK_TYPES))

    def test_get_soc_pstate(self):
        self.both("amdsmi_get_soc_pstate", self.handle)

    def test_set_clk_freq(self):
        # clk_type is a name string here, not an AmdSmiClkType.
        self.reject_only(
            "amdsmi_set_clk_freq",
            self.handle,
            api.text("clk_type", common.CLK_TYPES[0][0]),
            api.integer("freq_bitmask", 0),
        )

    def test_set_gpu_clk_limit(self):
        self.reject_only(
            "amdsmi_set_gpu_clk_limit",
            self.handle,
            api.text("clk_type", common.CLK_TYPES[0][0]),
            api.text("limit_type", common.CLK_LIMIT_TYPES[0][0]),
            api.integer("value", 0),
        )

    def test_set_soc_pstate(self):
        self.reject_only("amdsmi_set_soc_pstate", self.handle, api.integer("policy_id", 0))


if __name__ == "__main__":
    unittest.main()
