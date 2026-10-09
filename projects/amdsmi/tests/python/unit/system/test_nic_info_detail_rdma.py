#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC detail: each RDMA device is listed once, whatever the number of NIC ports.

``metric --nic --rdma`` flattens ``RDMA_DEVICES`` into the port index it passes to the library,
so a device repeated per NIC port (or dropped on a NIC with no ports) addresses the wrong ports.
The shipped ``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the function is
compiled from the real source and run against a fake NIC.
"""

import ast
import types
import unittest
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERFACE_PATH = _REPO_ROOT / "py-interface" / "amdsmi_interface.py"
_FUNCTION_NAME = "amdsmi_get_ainic_info_detail"
_ARRAY_LEN = 4  # the library arrays are larger than the populated count


def _load_detail() -> Any:
    """Compile the shipped detail function with its helpers replaced by stubs."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == _FUNCTION_NAME
    ]
    if len(nodes) != 1:
        raise AssertionError(f"{_FUNCTION_NAME} not found in amdsmi_interface.py")
    namespace: Dict[str, Any] = {
        "_format_bdf": lambda bdf: "bdf",
        "_decode_nic_capabilities": lambda bitmask: [],
    }
    exec(compile(ast.Module(nodes, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace[_FUNCTION_NAME]


def _port() -> types.SimpleNamespace:
    text_fields = (
        "type flavour netdev mac_address link_state autoneg pause_autoneg pause_rx pause_tx"
    ).split()
    values = {name: b"" for name in text_fields}
    values.update(ifindex=0, carrier=0, mtu=0, link_speed=0, active_fec=0)
    return types.SimpleNamespace(**values)


def _rdma_port(port_num: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        netdev=b"eth0", rdma_port=port_num, state=b"ACTIVE", max_mtu=4096, active_mtu=4096
    )


def _rdma_dev(name: bytes, num_ports: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        rdma_dev=name,
        node_guid=b"",
        node_type=b"",
        sys_image_guid=b"",
        fw_ver=b"",
        num_rdma_ports=num_ports,
        rdma_port_info=[_rdma_port(i + 1) for i in range(_ARRAY_LEN)],
    )


def _nic(num_ports: int, rdma_devs: List[types.SimpleNamespace]) -> types.SimpleNamespace:
    asic = types.SimpleNamespace(
        vendor_id=0,
        subvendor_id=0,
        device_id=0,
        subsystem_id=0,
        revision=0,
        permanent_address=b"",
        product_name=b"",
        part_number=b"",
        serial_number=b"",
        vendor_name=b"",
        capability=0,
    )
    bus = types.SimpleNamespace(
        bdf=0, max_pcie_width=0, max_pcie_speed=0, pcie_interface_version=b"", slot_type=b""
    )
    padded = rdma_devs + [_rdma_dev(b"", 0)] * (_ARRAY_LEN - len(rdma_devs))
    return types.SimpleNamespace(
        asic=asic,
        bus=bus,
        driver=types.SimpleNamespace(name=b"", version=b""),
        numa=types.SimpleNamespace(node=0, affinity=b""),
        port=types.SimpleNamespace(num_ports=num_ports, ports=[_port() for _ in range(_ARRAY_LEN)]),
        rdma_dev=types.SimpleNamespace(num_rdma_dev=len(rdma_devs), rdma_dev_info=padded),
    )


class TestNicInfoDetailRdma(unittest.TestCase):
    def test_a_device_is_listed_once_on_a_two_port_nic(self) -> None:
        detail = _load_detail()(_nic(2, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(list(detail["PORTS"]), ["PORT_0", "PORT_1"])
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0"])

    def test_devices_are_numbered_in_order(self) -> None:
        detail = _load_detail()(_nic(2, [_rdma_dev(b"ionic_0", 1), _rdma_dev(b"ionic_1", 1)]))
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0", "RDMA_DEVICE_1"])
        self.assertEqual(detail["RDMA_DEVICES"]["RDMA_DEVICE_1"]["NAME"], "ionic_1")

    def test_a_device_is_listed_on_a_nic_with_no_ports(self) -> None:
        detail = _load_detail()(_nic(0, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0"])

    def test_a_device_keeps_only_its_populated_ports(self) -> None:
        detail = _load_detail()(_nic(1, [_rdma_dev(b"ionic_0", 2)]))
        device = detail["RDMA_DEVICES"]["RDMA_DEVICE_0"]
        self.assertEqual([key for key in device if key.startswith("PORT_")], ["PORT_0", "PORT_1"])
