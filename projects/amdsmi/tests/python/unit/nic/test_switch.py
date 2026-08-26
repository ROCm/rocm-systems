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
"""Fabric switch APIs reached through a NIC."""

import unittest

import common.common as common


class TestNicSwitch(common.ApiTestCase):
    HANDLE_KIND = "nic"

    def test_get_root_switch(self):
        # The rejection path needs no live BDF, so run it before fetching one.
        bad = [("bad-type", common.BAD_STR), ("malformed", "not-a-bdf")]
        self._announce()
        self.api.reject(
            "amdsmi_get_root_switch",
            common.Param("amdsmi_bdf", ("0000:00:00.0", "0000:00:00.0"), bad),
        )
        self._require_device("amdsmi_get_root_switch")
        bdf = self.prerequisite("amdsmi_get_switch_device_bdf", self.handle.accepted[0][1])
        self.api.expect("amdsmi_get_root_switch", common.Param("amdsmi_bdf", (bdf, bdf), bad))

    def test_get_switch_device_bdf(self):
        self.both("amdsmi_get_switch_device_bdf", self.handle)

    def test_get_switch_device_uuid(self):
        self.both("amdsmi_get_switch_device_uuid", self.handle)

    def test_get_switch_link_info(self):
        self.both("amdsmi_get_switch_link_info", self.handle)

    def test_get_switch_metrics_info(self):
        self.both("amdsmi_get_switch_metrics_info", self.handle)

    def test_get_switch_topo_numa_affinity(self):
        self.both("amdsmi_get_switch_topo_numa_affinity", self.handle)

    def test_get_switch_topo_cpu_affinity(self):
        self.both("amdsmi_get_switch_topo_cpu_affinity", self.handle)


if __name__ == "__main__":
    unittest.main()
