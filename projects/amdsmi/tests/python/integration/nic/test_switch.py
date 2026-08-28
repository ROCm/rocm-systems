#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fabric switch APIs reached through a NIC."""

import unittest

import common.api_test as api


class TestNicSwitch(api.ApiTestCase):
    HANDLE_KIND = "nic"

    def test_get_root_switch(self):
        # The rejection path needs no live BDF, so run it before fetching one.
        bad = [("bad-type", api.BAD_STR), ("malformed", "not-a-bdf")]
        self._announce()
        self.api.reject(
            "amdsmi_get_root_switch", api.Param("amdsmi_bdf", ("0000:00:00.0", "0000:00:00.0"), bad)
        )
        self._require_device("amdsmi_get_root_switch")
        bdf = self.prerequisite("amdsmi_get_switch_device_bdf", self.handle.accepted[0][1])
        self.api.expect("amdsmi_get_root_switch", api.Param("amdsmi_bdf", (bdf, bdf), bad))

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
