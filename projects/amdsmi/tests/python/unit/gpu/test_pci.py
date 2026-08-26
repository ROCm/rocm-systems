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
"""GPU PCIe APIs."""

import unittest

import common.common as common


class TestGpuPci(common.ApiTestCase):
    def test_get_gpu_pci_bandwidth(self):
        self.both("amdsmi_get_gpu_pci_bandwidth", self.handle)

    def test_get_gpu_pci_throughput(self):
        self.both("amdsmi_get_gpu_pci_throughput", self.handle)

    def test_get_gpu_pci_replay_counter(self):
        self.both("amdsmi_get_gpu_pci_replay_counter", self.handle)

    def test_get_pcie_info(self):
        self.both("amdsmi_get_pcie_info", self.handle)

    def test_set_gpu_pci_bandwidth(self):
        self.reject_only("amdsmi_set_gpu_pci_bandwidth", self.handle, common.integer("bitmask", 0))


if __name__ == "__main__":
    unittest.main()
