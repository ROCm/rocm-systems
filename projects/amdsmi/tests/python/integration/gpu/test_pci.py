#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU PCIe APIs."""

import unittest

import common.api_test as api


class TestGpuPci(api.ApiTestCase):
    def test_get_gpu_pci_bandwidth(self):
        self.both("amdsmi_get_gpu_pci_bandwidth", self.handle)

    def test_get_gpu_pci_throughput(self):
        self.both("amdsmi_get_gpu_pci_throughput", self.handle)

    def test_get_gpu_pci_replay_counter(self):
        self.both("amdsmi_get_gpu_pci_replay_counter", self.handle)

    def test_get_pcie_info(self):
        self.both("amdsmi_get_pcie_info", self.handle)

    def test_set_gpu_pci_bandwidth(self):
        self.reject_only("amdsmi_set_gpu_pci_bandwidth", self.handle, api.integer("bitmask", 0))


if __name__ == "__main__":
    unittest.main()
