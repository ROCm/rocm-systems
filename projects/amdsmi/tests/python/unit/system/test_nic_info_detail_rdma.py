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
from enum import IntEnum
from typing import Any, Callable, Dict, List, Union

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_UTILS_PATH = _INTERFACE_PATH.with_name("amdsmi_interface_utils.py")
_FUNCTION_NAME = "amdsmi_get_ainic_info_detail"
_UNIT_HELPER_NAME = "_nic_value_with_unit"
_ARRAY_LEN = 4  # the library arrays are larger than the populated count
_INVALID_UTF8 = b"ab\xffcd"
_REPLACEMENT_CHAR = "\N{REPLACEMENT CHARACTER}"
_IFINDEX_BASE = 300  # above a byte, which the port struct cannot hold
_CAPABILITY_BITMASK = 0x5
_NIC_HANDLE = object()


def _capability_of(handle: Any) -> int:
    if handle is not _NIC_HANDLE:
        raise AssertionError("the capability getter was not given the info object's handle")
    return _CAPABILITY_BITMASK


def _ifindex_of(handle: Any, port_index: int) -> int:
    if handle is not _NIC_HANDLE:
        raise AssertionError("the ifindex getter was not given the info object's handle")
    return _IFINDEX_BASE + port_index


def _load_detail(ifindex_of: Callable[[Any, int], int] = _ifindex_of) -> Any:
    """Compile the shipped detail function with its helpers replaced by stubs."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in (_FUNCTION_NAME, _UNIT_HELPER_NAME)
    ]
    if len(nodes) != 2:
        raise AssertionError(
            f"{_FUNCTION_NAME} or {_UNIT_HELPER_NAME} not found in amdsmi_interface.py"
        )
    namespace: Dict[str, Any] = {
        "IntEnum": IntEnum,
        "Union": Union,
        "_format_bdf": lambda bdf: "bdf",
        "_decode_nic_capabilities": lambda bitmask: [bitmask],
        "amdsmi_get_nic_capability": _capability_of,
        "amdsmi_get_nic_port_ifindex": ifindex_of,
        "AmdSmiLibraryException": _FakeLibraryException,
    }
    exec(compile(ast.Module(_load_unavailable_helpers(), []), str(_UTILS_PATH), "exec"), namespace)
    exec(compile(ast.Module(nodes, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace[_FUNCTION_NAME]


def _load_unavailable_helpers() -> List[ast.stmt]:
    """The real ``MaxUIntegerTypes`` and ``_validate_if_max_uint`` the detail function relies on."""
    tree = ast.parse(_UTILS_PATH.read_text(encoding="utf-8"))
    wanted = {"MaxUIntegerTypes", "_validate_if_max_uint"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
    ]
    missing = wanted - {node.name for node in nodes}
    if missing:
        raise AssertionError(f"not found in amdsmi_interface_utils.py: {sorted(missing)}")
    return nodes


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
    )
    bus = types.SimpleNamespace(
        bdf=0, max_pcie_width=0, max_pcie_speed=0, pcie_interface_version=b"", slot_type=b""
    )
    padded = rdma_devs + [_rdma_dev(b"", 0)] * (_ARRAY_LEN - len(rdma_devs))
    return types.SimpleNamespace(
        handle=_NIC_HANDLE,
        asic=asic,
        bus=bus,
        driver=types.SimpleNamespace(name=b"", version=b""),
        numa=types.SimpleNamespace(node=0, affinity=b""),
        port=types.SimpleNamespace(num_ports=num_ports, ports=[_port() for _ in range(_ARRAY_LEN)]),
        rdma_dev=types.SimpleNamespace(num_rdma_dev=len(rdma_devs), rdma_dev_info=padded),
    )


# Largest value of each field's C type, which the library reports for an attribute it could not read.
_UNAVAILABLE_CARRIER = 0xFF
_UNAVAILABLE_MTU = 0xFFFF
_UNAVAILABLE_UINT32 = 0xFFFFFFFF
# One below each marker: a real value that a ``>=`` check would wrongly hide.
_LARGEST_CARRIER = _UNAVAILABLE_CARRIER - 1
_LARGEST_MTU = _UNAVAILABLE_MTU - 1
_LARGEST_UINT32 = _UNAVAILABLE_UINT32 - 1
_FAILING_PORT = 1
_TOP_LEVEL_KEYS = {"ASIC", "BUS", "DRIVER", "NUMA", "PORTS", "RDMA_DEVICES"}


def _ports_values(*per_port: Dict[str, int]) -> List[Dict[str, Any]]:
    """Detail of a NIC with one port per entry of ``per_port``, each carrying those values."""
    nic = _nic(len(per_port), [])
    for port, values in zip(nic.port.ports, per_port):
        for name, value in values.items():
            setattr(port, name, value)
    ports = _load_detail()(nic)["PORTS"]
    return [ports[f"PORT_{index}"] for index in range(len(per_port))]


def _port_values(**values: int) -> Dict[str, Any]:
    """Detail of a one-port NIC whose port carries ``values``."""
    return _ports_values(values)[0]


class _FakeLibraryException(Exception):
    """Stands in for AmdSmiLibraryException, which the real getter raises on a library error."""


def _ifindex_failing_on_one_port(handle: Any, port_index: int) -> int:
    if port_index == _FAILING_PORT:
        raise _FakeLibraryException(port_index)
    return _ifindex_of(handle, port_index)


class TestNicInfoDetailIfindexFailure(unittest.TestCase):
    def _detail(self) -> Dict[str, Any]:
        nic = _nic(2, [])
        return _load_detail(_ifindex_failing_on_one_port)(nic)

    def test_a_failed_ifindex_read_shows_na_for_that_port_only(self) -> None:
        ports = self._detail()["PORTS"]
        self.assertEqual(ports["PORT_0"]["IFINDEX"], _IFINDEX_BASE)
        self.assertEqual(ports["PORT_1"]["IFINDEX"], "N/A")

    def test_a_failed_ifindex_read_keeps_the_rest_of_the_nic(self) -> None:
        detail = self._detail()
        self.assertEqual(set(detail), _TOP_LEVEL_KEYS)
        self.assertEqual(detail["PORTS"]["PORT_1"]["MTU"], "0B")


class TestNicInfoDetailPortValues(unittest.TestCase):
    def test_a_value_the_library_could_not_read_shows_na(self) -> None:
        port = _port_values(
            carrier=_UNAVAILABLE_CARRIER,
            mtu=_UNAVAILABLE_MTU,
            link_speed=_UNAVAILABLE_UINT32,
            active_fec=_UNAVAILABLE_UINT32,
        )
        self.assertEqual(port["CARRIER"], "N/A")
        self.assertEqual(port["MTU"], "N/A")
        self.assertEqual(port["LINK_SPEED"], "N/A")
        self.assertEqual(port["ACTIVE_FEC"], "N/A")

    def test_a_down_link_reads_na_speed_with_a_known_carrier(self) -> None:
        port = _port_values(carrier=0, link_speed=_UNAVAILABLE_UINT32)
        self.assertEqual(port["CARRIER"], 0)
        self.assertEqual(port["LINK_SPEED"], "N/A")

    def test_a_value_that_was_read_keeps_its_unit(self) -> None:
        port = _port_values(carrier=1, mtu=1500, link_speed=100000, active_fec=8)
        self.assertEqual(port["CARRIER"], 1)
        self.assertEqual(port["MTU"], "1500B")
        self.assertEqual(port["LINK_SPEED"], "100000 Mb/s")
        self.assertEqual(port["ACTIVE_FEC"], "8")

    def test_one_unavailable_value_does_not_hide_its_neighbours(self) -> None:
        port = _port_values(carrier=1, mtu=1500, link_speed=_UNAVAILABLE_UINT32, active_fec=8)
        self.assertEqual(port["CARRIER"], 1)
        self.assertEqual(port["MTU"], "1500B")
        self.assertEqual(port["LINK_SPEED"], "N/A")
        self.assertEqual(port["ACTIVE_FEC"], "8")

    def test_a_value_of_zero_is_a_real_value(self) -> None:
        port = _port_values(carrier=0, mtu=0, link_speed=0, active_fec=0)
        self.assertEqual(port["CARRIER"], 0)
        self.assertEqual(port["MTU"], "0B")
        self.assertEqual(port["LINK_SPEED"], "0 Mb/s")
        self.assertEqual(port["ACTIVE_FEC"], "0")

    def test_the_value_below_the_marker_is_a_real_value(self) -> None:
        port = _port_values(
            carrier=_LARGEST_CARRIER,
            mtu=_LARGEST_MTU,
            link_speed=_LARGEST_UINT32,
            active_fec=_LARGEST_UINT32,
        )
        self.assertEqual(port["CARRIER"], _LARGEST_CARRIER)
        self.assertEqual(port["MTU"], f"{_LARGEST_MTU}B")
        self.assertEqual(port["LINK_SPEED"], f"{_LARGEST_UINT32} Mb/s")
        self.assertEqual(port["ACTIVE_FEC"], f"{_LARGEST_UINT32}")

    def test_the_carrier_stays_an_integer(self) -> None:
        port = _port_values(carrier=1)
        self.assertIs(type(port["CARRIER"]), int)

    def test_each_port_reports_its_own_values(self) -> None:
        down, up = _ports_values(
            {"carrier": 0, "mtu": 1500, "link_speed": _UNAVAILABLE_UINT32, "active_fec": 4},
            {"carrier": 1, "mtu": 9000, "link_speed": 100000, "active_fec": 8},
        )
        self.assertEqual(
            (down["CARRIER"], down["MTU"], down["LINK_SPEED"], down["ACTIVE_FEC"]),
            (0, "1500B", "N/A", "4"),
        )
        self.assertEqual(
            (up["CARRIER"], up["MTU"], up["LINK_SPEED"], up["ACTIVE_FEC"]),
            (1, "9000B", "100000 Mb/s", "8"),
        )


class TestNicInfoDetailFirmware(unittest.TestCase):
    def test_static_does_not_report_the_driver_as_firmware(self) -> None:
        nic = _nic(1, [])
        nic.driver = types.SimpleNamespace(name=b"ionic", version=b"25.12")
        detail = _load_detail()(nic)
        self.assertEqual(set(detail), _TOP_LEVEL_KEYS)


class TestNicInfoDetailRdma(unittest.TestCase):
    def test_a_device_is_listed_once_on_a_two_port_nic(self) -> None:
        detail = _load_detail()(_nic(2, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(list(detail["PORTS"]), ["PORT_0", "PORT_1"])
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0"])

    def test_devices_are_numbered_in_order(self) -> None:
        detail = _load_detail()(_nic(2, [_rdma_dev(b"ionic_0", 1), _rdma_dev(b"ionic_1", 1)]))
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0", "RDMA_DEVICE_1"])
        self.assertEqual(detail["RDMA_DEVICES"]["RDMA_DEVICE_1"]["NAME"], "ionic_1")

    def test_each_port_reports_its_full_ifindex(self) -> None:
        detail = _load_detail()(_nic(2, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(detail["PORTS"]["PORT_0"]["IFINDEX"], _IFINDEX_BASE)
        self.assertEqual(detail["PORTS"]["PORT_1"]["IFINDEX"], _IFINDEX_BASE + 1)

    def test_the_capability_comes_from_the_capability_getter(self) -> None:
        detail = _load_detail()(_nic(1, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(detail["ASIC"]["CAPABILITY"], [_CAPABILITY_BITMASK])

    def test_a_device_is_listed_on_a_nic_with_no_ports(self) -> None:
        detail = _load_detail()(_nic(0, [_rdma_dev(b"ionic_0", 1)]))
        self.assertEqual(list(detail["RDMA_DEVICES"]), ["RDMA_DEVICE_0"])

    def test_a_device_keeps_only_its_populated_ports(self) -> None:
        detail = _load_detail()(_nic(1, [_rdma_dev(b"ionic_0", 2)]))
        device = detail["RDMA_DEVICES"]["RDMA_DEVICE_0"]
        self.assertEqual([key for key in device if key.startswith("PORT_")], ["PORT_0", "PORT_1"])


def _poison(node: Any) -> None:
    """Replace every bytes field under ``node`` with text that is not valid UTF-8."""
    for name, value in vars(node).items():
        if isinstance(value, bytes):
            setattr(node, name, _INVALID_UTF8)
        elif isinstance(value, types.SimpleNamespace):
            _poison(value)
        elif isinstance(value, list):
            for item in value:
                _poison(item)


class TestNicInfoDetailText(unittest.TestCase):
    """A device string with a stray byte must not abort the whole ``static --nic`` output."""

    def test_non_utf8_text_is_replaced_in_every_field(self) -> None:
        nic = _nic(1, [_rdma_dev(b"ionic_0", 1)])
        _poison(nic)
        detail = _load_detail()(nic)
        port = detail["PORTS"]["PORT_0"]
        device = detail["RDMA_DEVICES"]["RDMA_DEVICE_0"]
        for text in (
            detail["ASIC"]["SERIAL_NUMBER"],
            detail["DRIVER"]["NAME"],
            port["NETDEV"],
            device["FW_VER"],
            device["PORT_0"]["STATE"],
        ):
            self.assertIn(_REPLACEMENT_CHAR, text)
