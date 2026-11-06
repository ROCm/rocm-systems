#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AI NIC values the library reports as unknown come out as N/A (hardware-free)."""

import unittest
from unittest import mock

from common.common import amdsmi

_U8_MAX = 0xFF
_U16_MAX = 0xFFFF
_U32_MAX = 0xFFFFFFFF


def _detail(bus, numa, port, driver=None):
    """Return amdsmi_get_ainic_info_detail() output for a NIC with one port."""
    handle = amdsmi.amdsmi_wrapper.amdsmi_processor_handle()

    def _filled(fields):
        def _stub(_handle, info_ptr):
            for name, value in fields.items():
                setattr(info_ptr._obj, name, value)
            return 0

        return _stub

    def _one_port(_handle, info_ptr):
        info_ptr._obj.num_ports = 1
        for name, value in port.items():
            setattr(info_ptr._obj.ports[0], name, value)
        return 0

    with mock.patch.multiple(
        amdsmi.amdsmi_wrapper,
        amdsmi_get_nic_asic_info=_filled({}),
        amdsmi_get_nic_bus_info=_filled(bus),
        amdsmi_get_nic_driver_info=_filled(driver or {}),
        amdsmi_get_nic_numa_info=_filled(numa),
        amdsmi_get_nic_port_info=_one_port,
        amdsmi_get_nic_rdma_dev_info=_filled({}),
    ):
        return amdsmi.amdsmi_interface.amdsmi_get_ainic_info(handle, detail=True)


def _fields(info):
    port = info["PORTS"]["PORT_0"]
    return {
        "MAX_PCIE_WIDTH": info["BUS"]["MAX_PCIE_WIDTH"],
        "MAX_PCIE_SPEED": info["BUS"]["MAX_PCIE_SPEED"],
        "NODE": info["NUMA"]["NODE"],
        "CARRIER": port["CARRIER"],
        "MTU": port["MTU"],
        "LINK_SPEED": port["LINK_SPEED"],
        "ACTIVE_FEC": port["ACTIVE_FEC"],
    }


class TestAinicUnknownValues(unittest.TestCase):
    def test_unknown_values_are_na(self):
        info = _detail(
            bus={"max_pcie_width": _U8_MAX, "max_pcie_speed": _U32_MAX},
            numa={"node": _U8_MAX},
            port={
                "carrier": _U8_MAX,
                "mtu": _U16_MAX,
                "link_speed": _U32_MAX,
                "active_fec": _U32_MAX,
            },
        )
        self.assertEqual(
            _fields(info),
            {
                "MAX_PCIE_WIDTH": "N/A",
                "MAX_PCIE_SPEED": "N/A",
                "NODE": "N/A",
                "CARRIER": "N/A",
                "MTU": "N/A",
                "LINK_SPEED": "N/A",
                "ACTIVE_FEC": "N/A",
            },
        )

    def test_known_values_keep_their_format(self):
        info = _detail(
            bus={"max_pcie_width": 16, "max_pcie_speed": 32},
            numa={"node": 0},
            port={"carrier": 0, "mtu": 1500, "link_speed": 100000, "active_fec": 8},
        )
        self.assertEqual(
            _fields(info),
            {
                "MAX_PCIE_WIDTH": 16,
                "MAX_PCIE_SPEED": 32,
                "NODE": 0,
                "CARRIER": 0,
                "MTU": "1500B",
                "LINK_SPEED": "100000 Mb/s",
                "ACTIVE_FEC": "8",
            },
        )

    def test_fw_fields_do_not_repeat_the_driver(self):
        info = _detail(
            bus={}, numa={}, port={}, driver={"name": b"ionic", "version": b"25.12.15.001"}
        )
        self.assertEqual(info["DRIVER"], {"NAME": "ionic", "VERSION": "25.12.15.001"})
        self.assertEqual(info["FW"], {"FW_0": "N/A", "FW_HEARTBEAT": "N/A"})
