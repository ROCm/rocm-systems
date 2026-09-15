#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""NIC discovery and information APIs."""

import unittest

import common.api_test as api
import common.common as common


class TestNicIdentity(api.ApiTestCase):
    HANDLE_KIND = "nic"

    def _sockets(self):
        return api.Handle("socket", common.amdsmi.amdsmi_get_socket_handles())

    def test_get_nic_processor_handles(self):
        self.both("amdsmi_get_nic_processor_handles", self._sockets())

    def test_get_nic_info(self):
        self.both("amdsmi_get_nic_info", self.handle)

    def test_get_ainic_info(self):
        self.both("amdsmi_get_ainic_info", self.handle)

    def test_get_nic_asic_info(self):
        self.both("amdsmi_get_nic_asic_info", self.handle)

    def test_get_nic_bus_info(self):
        self.both("amdsmi_get_nic_bus_info", self.handle)

    def test_get_nic_driver_info(self):
        self.both("amdsmi_get_nic_driver_info", self.handle)

    def test_get_nic_numa_info(self):
        self.both("amdsmi_get_nic_numa_info", self.handle)

    def test_get_nic_device_bdf(self):
        self.both("amdsmi_get_nic_device_bdf", self.handle)

    def test_get_nic_device_uuid(self):
        self.both("amdsmi_get_nic_device_uuid", self.handle)

    def test_get_nic_fw_info(self):
        self.both("amdsmi_get_nic_fw_info", self.handle)

    def test_get_nic_fw_version(self):
        self.both("amdsmi_get_nic_fw_version", self.handle)

    def test_get_nic_temp_info(self):
        self.both("amdsmi_get_nic_temp_info", self.handle)

    def test_get_nic_metrics_info(self):
        self.both("amdsmi_get_nic_metrics_info", self.handle)

    def test_get_nic_port_info(self):
        self.both("amdsmi_get_nic_port_info", self.handle)

    def test_get_nic_port_statistics(self):
        self.both("amdsmi_get_nic_port_statistics", self.handle, api.integer("port_index", 0))

    def test_get_nic_vendor_statistics(self):
        self.both("amdsmi_get_nic_vendor_statistics", self.handle)

    def test_get_nic_rdma_dev_info(self):
        self.both("amdsmi_get_nic_rdma_dev_info", self.handle)

    def test_get_nic_rdma_port_statistics(self):
        self.both(
            "amdsmi_get_nic_rdma_port_statistics", self.handle, api.integer("rdma_port_index", 0)
        )

    def test_get_nic_topo_numa_affinity(self):
        self.both("amdsmi_get_nic_topo_numa_affinity", self.handle)

    def test_get_nic_topo_cpu_affinity(self):
        self.both("amdsmi_get_nic_topo_cpu_affinity", self.handle)

    def test_get_nic_gpu_topo_info(self):
        self.both(
            "amdsmi_get_nic_gpu_topo_info",
            self.handle,
            api.Handle("gpu_dst", self.common.processors),
        )


if __name__ == "__main__":
    unittest.main()
