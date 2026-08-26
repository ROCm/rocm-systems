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
"""GPU clock and pstate APIs: bad arguments are rejected, good reads are valid."""

import unittest

import common.common as common


class TestGpuClock(common.ApiTestCase):
    def test_get_clk_freq(self):
        self.both("amdsmi_get_clk_freq", self.handle, common.enum("clk_type", common.CLK_TYPES))

    def test_get_clock_info(self):
        self.both("amdsmi_get_clock_info", self.handle, common.enum("clock_type", common.CLK_TYPES))

    def test_get_soc_pstate(self):
        self.both("amdsmi_get_soc_pstate", self.handle)

    def test_set_clk_freq(self):
        # clk_type is a name string here, not an AmdSmiClkType.
        self.reject_only(
            "amdsmi_set_clk_freq",
            self.handle,
            common.text("clk_type", common.CLK_TYPES[0][0]),
            common.integer("freq_bitmask", 0),
        )

    def test_set_gpu_clk_limit(self):
        self.reject_only(
            "amdsmi_set_gpu_clk_limit",
            self.handle,
            common.text("clk_type", common.CLK_TYPES[0][0]),
            common.text("limit_type", common.CLK_LIMIT_TYPES[0][0]),
            common.integer("value", 0),
        )

    def test_set_soc_pstate(self):
        self.reject_only("amdsmi_set_soc_pstate", self.handle, common.integer("policy_id", 0))


if __name__ == "__main__":
    unittest.main()
