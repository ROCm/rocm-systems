#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AI NIC port ifindex: struct layout and reporting (hardware-free)."""

import ctypes
import unittest
from unittest import mock

from common.common import amdsmi

# Includes the trailing pad that sets the stride of amdsmi_nic_port_info_t.ports.
_PORT_SIZE = 2336
_INT_MAX = 0x7FFFFFFF
_UINT32_MAX = 0xFFFFFFFF


def _reported_ifindex(ifindex: int):
    """Return IFINDEX for one port whose stub library reports the given ifindex."""
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _empty(_handle, _info_ptr):
        return 0

    def _one_port(_handle, info_ptr):
        info_ptr._obj.num_ports = 1
        info_ptr._obj.ports[0].ifindex = ifindex
        return 0

    with mock.patch.multiple(
        amdsmi.amdsmi_wrapper,
        amdsmi_get_nic_asic_info=_empty,
        amdsmi_get_nic_bus_info=_empty,
        amdsmi_get_nic_driver_info=_empty,
        amdsmi_get_nic_numa_info=_empty,
        amdsmi_get_nic_port_info=_one_port,
        amdsmi_get_nic_rdma_dev_info=_empty,
    ):
        info = amdsmi.amdsmi_interface.amdsmi_get_ainic_info(handle, detail=True)
    return info["PORTS"]["PORT_0"]["IFINDEX"]


class TestNicPortLayout(unittest.TestCase):
    """Pins the 32-bit ifindex and the field it moved."""

    def test_port_struct_size(self):
        self.assertEqual(
            ctypes.sizeof(amdsmi.amdsmi_wrapper.struct_amdsmi_nic_port_t),
            _PORT_SIZE,
            "regenerate the Python wrapper",
        )

    def test_ifindex_is_32_bits_and_mac_address_follows(self):
        port_t = amdsmi.amdsmi_wrapper.struct_amdsmi_nic_port_t
        self.assertEqual(port_t.ifindex.size, 4)
        self.assertEqual(port_t.mac_address.offset, port_t.ifindex.offset + 4)


class TestNicPortIfindexReporting(unittest.TestCase):
    def test_largest_kernel_index_is_reported_whole(self):
        self.assertEqual(_reported_ifindex(_INT_MAX), _INT_MAX)

    def test_unknown_index_is_na(self):
        self.assertEqual(_reported_ifindex(_UINT32_MAX), "N/A")
