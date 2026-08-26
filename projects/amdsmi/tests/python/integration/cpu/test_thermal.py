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
"""CPU thermal APIs."""

import unittest

import common.api_test as api


class TestCpuThermal(api.ApiTestCase):
    HANDLE_KIND = "cpu"

    def test_get_cpu_socket_temperature(self):
        self.both("amdsmi_get_cpu_socket_temperature", self.handle)

    def test_get_cpu_prochot_status(self):
        self.both("amdsmi_get_cpu_prochot_status", self.handle)

    def test_get_cpu_tdelta(self):
        self.both("amdsmi_get_cpu_tdelta", self.handle)

    def test_get_cpu_svi3_vr_controller_temp(self):
        self.both(
            "amdsmi_get_cpu_svi3_vr_controller_temp",
            self.handle,
            api.integer("rail_selection", 0),
            api.integer("rail_index", 0),
        )


if __name__ == "__main__":
    unittest.main()
