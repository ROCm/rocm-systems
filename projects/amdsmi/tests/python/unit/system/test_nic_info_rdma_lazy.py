#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The NIC summary does not read the RDMA info; only the detail does.

Reading the RDMA info makes the library probe the MTU of every RDMA port (a fork per port), and
every ``amd-smi`` run builds the NIC summary while it builds its parser. The shipped
``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so ``ainic_info_t`` and
``amdsmi_get_ainic_info`` are compiled from the real source and run against a fake wrapper.
"""

import ast
import types
import unittest
from typing import Any, Callable, Dict
from unittest import mock

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_DEFINITIONS = ("ainic_info_t", "amdsmi_get_ainic_info")
_STATUS_SUCCESS = 0
_INFO_STRUCTS = (
    "amdsmi_nic_asic_info_t",
    "amdsmi_nic_bus_info_t",
    "amdsmi_nic_driver_info_t",
    "amdsmi_nic_numa_info_t",
    "amdsmi_nic_port_info_t",
    "amdsmi_nic_rdma_devices_info_t",
)
_INFO_GETTERS = (
    "amdsmi_get_nic_asic_info",
    "amdsmi_get_nic_bus_info",
    "amdsmi_get_nic_driver_info",
    "amdsmi_get_nic_numa_info",
    "amdsmi_get_nic_port_info",
)
_SUMMARY = {"summary": True}
_DETAIL = {"detail": True}


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _EmptyStruct:
    """Stands in for each amdsmi_nic_*_info_t."""


def _get_info(handle: object, out: object) -> int:
    return _STATUS_SUCCESS


def _load_namespace() -> Dict[str, Any]:
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    wanted = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _DEFINITIONS
    ]
    if len(wanted) != len(_DEFINITIONS):
        raise AssertionError(f"{_DEFINITIONS} not all found in amdsmi_interface.py")

    rdma_getter = mock.create_autospec(_get_info, return_value=_STATUS_SUCCESS)
    wrapper = types.SimpleNamespace(
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_get_nic_rdma_dev_info=rdma_getter,
        **{name: _EmptyStruct for name in _INFO_STRUCTS},
        **{
            name: mock.create_autospec(_get_info, return_value=_STATUS_SUCCESS)
            for name in _INFO_GETTERS
        },
    )
    namespace: Dict[str, Any] = {
        "amdsmi_wrapper": wrapper,
        "ctypes": types.SimpleNamespace(byref=lambda obj: obj),
        "AmdSmiParameterException": Exception,
        "_check_res": lambda status: None,
        "amdsmi_get_ainic_info_summary": lambda info: _SUMMARY,
        "amdsmi_get_ainic_info_detail": lambda info: _DETAIL,
        "Any": Any,
        "Dict": Dict,
        "Callable": Callable,
    }
    exec(compile(ast.Module(wanted, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace


class TestAinicInfoRdmaRead(unittest.TestCase):
    def test_the_summary_does_not_read_the_rdma_info(self) -> None:
        namespace = _load_namespace()

        result = namespace["amdsmi_get_ainic_info"](_FakeHandle())

        self.assertEqual(result, _SUMMARY)
        namespace["amdsmi_wrapper"].amdsmi_get_nic_rdma_dev_info.assert_not_called()

    def test_the_detail_reads_the_rdma_info_once(self) -> None:
        namespace = _load_namespace()

        result = namespace["amdsmi_get_ainic_info"](_FakeHandle(), True)

        self.assertEqual(result, _DETAIL)
        namespace["amdsmi_wrapper"].amdsmi_get_nic_rdma_dev_info.assert_called_once()
