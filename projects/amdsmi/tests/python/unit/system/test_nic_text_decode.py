#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC getters replace undecodable device text instead of raising.

Firmware names and versions, netdev names and counter names come from the device and the kernel.
A single stray byte made ``UnicodeDecodeError`` escape and ``amd-smi`` exit 1 with nothing printed.
The shipped functions are compiled from ``amdsmi_interface.py`` and run against a fake library,
because importing the module needs a built ``libamd_smi.so``.
"""

import ast
import ctypes
import types
import unittest
from typing import Any, Dict, List

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_STATUS_SUCCESS = 0
_NAME_BYTES = 64
_INVALID_UTF8 = b"ab\xffcd"
_REPLACEMENT_CHAR = "\N{REPLACEMENT CHARACTER}"
_FIXED_NAME = "ab" + _REPLACEMENT_CHAR + "cd"


class _FakeParameterException(Exception):
    """Stands in for AmdSmiParameterException."""


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _FwEntry(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * _NAME_BYTES), ("version", ctypes.c_char * _NAME_BYTES)]


class _FwInfo(ctypes.Structure):
    _fields_ = [("num_fw", ctypes.c_uint32), ("fw", _FwEntry * 2)]


class _Port(ctypes.Structure):
    _fields_ = [("netdev", ctypes.c_char * _NAME_BYTES)]


class _PortInfo(ctypes.Structure):
    _fields_ = [("num_ports", ctypes.c_uint32), ("ports", _Port * 2)]


class _Stat(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * _NAME_BYTES), ("value", ctypes.c_uint64)]


def _fake_check_res(status: int) -> None:
    if status != _STATUS_SUCCESS:
        raise AssertionError(f"unexpected status {status}")


def _load(names: List[str], wrapper: types.SimpleNamespace) -> Dict[str, Any]:
    """Compile the named shipped functions against a fake ``amdsmi_wrapper``."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in nodes}
    if missing:
        raise AssertionError(f"not found in amdsmi_interface.py: {sorted(missing)}")
    namespace: Dict[str, Any] = {
        "amdsmi_wrapper": wrapper,
        "ctypes": ctypes,
        "Dict": Dict,
        "Any": Any,
        "AmdSmiParameterException": _FakeParameterException,
        "_check_res": _fake_check_res,
    }
    exec(compile(ast.Module(nodes, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace


def _stats_getter(function_name: str) -> Any:
    """A statistics getter whose fake library reports one counter with a bad name."""

    def fake_get_stats(handle: object, index: int, *rest: Any) -> int:
        # The vendor getter takes a scope before the count and the buffer.
        num_stats, stats = rest[-2], rest[-1]
        num_stats._obj.value = 1
        if stats is not None:
            stats[0].name = _INVALID_UTF8
            stats[0].value = 7
        return _STATUS_SUCCESS

    wrapper = types.SimpleNamespace(
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_nic_stat_t=_Stat,
        AMDSMI_NIC_STAT_SCOPE_DEFAULT=0,
        **{function_name: fake_get_stats},
    )
    return _load([function_name], wrapper)[function_name]


class TestNicTextDecode(unittest.TestCase):
    def test_a_firmware_name_and_version_are_replaced(self) -> None:
        def fake_get_fw_info(handle: object, out: Any) -> int:
            fw_info = out._obj
            fw_info.num_fw = 1
            fw_info.fw[0].name = _INVALID_UTF8
            fw_info.fw[0].version = _INVALID_UTF8
            return _STATUS_SUCCESS

        wrapper = types.SimpleNamespace(
            amdsmi_processor_handle=_FakeHandle,
            amdsmi_nic_fw_info_t=_FwInfo,
            amdsmi_get_nic_fw_info=fake_get_fw_info,
        )
        getter = _load(["amdsmi_get_nic_fw_info"], wrapper)["amdsmi_get_nic_fw_info"]
        self.assertEqual(getter(_FakeHandle()), {_FIXED_NAME: _FIXED_NAME})

    def test_a_netdev_name_is_replaced(self) -> None:
        def fake_get_port_info(handle: object, out: Any) -> int:
            port_info = out._obj
            port_info.num_ports = 1
            port_info.ports[0].netdev = _INVALID_UTF8
            return _STATUS_SUCCESS

        wrapper = types.SimpleNamespace(
            amdsmi_processor_handle=_FakeHandle,
            amdsmi_nic_port_info_t=_PortInfo,
            amdsmi_get_nic_port_info=fake_get_port_info,
        )
        getter = _load(["amdsmi_get_nic_port_info"], wrapper)["amdsmi_get_nic_port_info"]
        self.assertEqual(
            getter(_FakeHandle()), {"num_ports": 1, "ports": [{"netdev": _FIXED_NAME}]}
        )

    def test_a_port_statistic_name_is_replaced(self) -> None:
        getter = _stats_getter("amdsmi_get_nic_port_statistics")
        self.assertEqual(getter(_FakeHandle(), 0), {_FIXED_NAME: 7})

    def test_a_vendor_statistic_name_is_replaced(self) -> None:
        getter = _stats_getter("amdsmi_get_nic_vendor_statistics")
        self.assertEqual(getter(_FakeHandle(), 0), {_FIXED_NAME: 7})

    def test_an_rdma_statistic_name_is_replaced(self) -> None:
        getter = _stats_getter("amdsmi_get_nic_rdma_port_statistics")
        self.assertEqual(getter(_FakeHandle(), 0), {_FIXED_NAME: 7})
