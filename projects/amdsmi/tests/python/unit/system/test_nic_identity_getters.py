#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC capability and port interface index getters.

``amdsmi_nic_asic_info_t`` and ``amdsmi_nic_port_t`` keep their layouts from the previous release, so the
capability bitmask and the full interface index come from their own C getters. The shipped
``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so each test compiles only the
definitions it exercises from the real source and runs them against a fake wrapper.
"""

import ast
import ctypes
import types
import unittest
from typing import Dict, List

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)

_STATUS_SUCCESS = 0
_STATUS_FAILURE = 1
_CAPABILITY_BITMASK = 0x5
_BEYOND_A_BYTE_IFINDEX = 300
_PORT_INDEX = 1


class _FakeParameterException(Exception):
    """Stands in for AmdSmiParameterException."""


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _FakeLibraryException(Exception):
    """Raised by the fake _check_res on a non-zero status."""


def _fake_check_res(status: int) -> None:
    if status != _STATUS_SUCCESS:
        raise _FakeLibraryException(status)


def _definition_name(node: ast.stmt) -> str:
    """Name bound by a top-level function, class or single-target assignment; "" otherwise."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


def _load_definitions(names: List[str], namespace: Dict[str, object]) -> Dict[str, object]:
    """Compile the named top-level functions and classes from the shipped interface source."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    wanted = [node for node in tree.body if _definition_name(node) in names]
    missing = set(names) - {_definition_name(node) for node in wanted}
    if missing:
        raise AssertionError(f"not found in amdsmi_interface.py: {sorted(missing)}")
    exec(compile(ast.Module(wanted, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace


def _fake_wrapper() -> types.SimpleNamespace:
    """A wrapper whose getters write fixed values and record how they were called."""
    capability_calls: List[object] = []
    ifindex_calls: List[tuple] = []

    def fake_get_nic_capability(handle: object, out: ctypes.pointer) -> int:
        capability_calls.append(handle)
        out._obj.value = _CAPABILITY_BITMASK
        return _STATUS_SUCCESS

    def fake_get_nic_port_ifindex(handle: object, port_index: int, out: ctypes.pointer) -> int:
        ifindex_calls.append((handle, port_index))
        out._obj.value = _BEYOND_A_BYTE_IFINDEX
        return _STATUS_SUCCESS

    return types.SimpleNamespace(
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_get_nic_capability=fake_get_nic_capability,
        amdsmi_get_nic_port_ifindex=fake_get_nic_port_ifindex,
        capability_calls=capability_calls,
        ifindex_calls=ifindex_calls,
    )


def _namespace() -> Dict[str, object]:
    return {
        "amdsmi_wrapper": _fake_wrapper(),
        "ctypes": ctypes,
        "AmdSmiParameterException": _FakeParameterException,
        "_check_res": _fake_check_res,
    }


class _EmptyStruct(ctypes.Structure):
    """Stands in for each amdsmi_nic_*_info_t that ainic_info_t fills."""

    _fields_: list = []


class TestAinicInfoHandle(unittest.TestCase):
    """The summary and detail readers fetch type, capability and ifindex with this handle."""

    def test_the_object_keeps_the_handle_it_was_built_from(self) -> None:
        wrapper = types.SimpleNamespace(
            **{
                name: _EmptyStruct
                for name in (
                    "amdsmi_nic_asic_info_t",
                    "amdsmi_nic_bus_info_t",
                    "amdsmi_nic_driver_info_t",
                    "amdsmi_nic_numa_info_t",
                    "amdsmi_nic_port_info_t",
                    "amdsmi_nic_rdma_devices_info_t",
                )
            },
            **{
                name: (lambda handle, out: _STATUS_SUCCESS)
                for name in (
                    "amdsmi_get_nic_asic_info",
                    "amdsmi_get_nic_bus_info",
                    "amdsmi_get_nic_driver_info",
                    "amdsmi_get_nic_numa_info",
                    "amdsmi_get_nic_port_info",
                    "amdsmi_get_nic_rdma_dev_info",
                )
            },
        )
        namespace = _namespace()
        namespace["amdsmi_wrapper"] = wrapper
        _load_definitions(["ainic_info_t"], namespace)
        handle = _FakeHandle()

        info = namespace["ainic_info_t"](handle)

        self.assertIs(info.handle, handle)


class TestGetNicCapability(unittest.TestCase):
    """amdsmi_get_nic_capability wraps the C getter."""

    def _getter(self) -> Dict[str, object]:
        return _load_definitions(["amdsmi_get_nic_capability"], _namespace())

    def test_returns_the_bitmask_the_library_reports(self) -> None:
        namespace = self._getter()
        result = namespace["amdsmi_get_nic_capability"](_FakeHandle())
        self.assertEqual(result, _CAPABILITY_BITMASK)

    def test_passes_the_handle_to_the_library(self) -> None:
        namespace = self._getter()
        handle = _FakeHandle()
        namespace["amdsmi_get_nic_capability"](handle)
        self.assertEqual(namespace["amdsmi_wrapper"].capability_calls, [handle])

    def test_rejects_an_object_that_is_not_a_handle(self) -> None:
        namespace = self._getter()
        with self.assertRaises(_FakeParameterException):
            namespace["amdsmi_get_nic_capability"]("not a handle")
        self.assertEqual(namespace["amdsmi_wrapper"].capability_calls, [])

    def test_a_library_failure_is_raised_not_swallowed(self) -> None:
        namespace = self._getter()
        namespace["amdsmi_wrapper"].amdsmi_get_nic_capability = lambda handle, out: _STATUS_FAILURE
        with self.assertRaises(_FakeLibraryException):
            namespace["amdsmi_get_nic_capability"](_FakeHandle())


class TestGetNicPortIfindex(unittest.TestCase):
    """amdsmi_get_nic_port_ifindex returns the whole index, not the struct's one byte."""

    def _getter(self) -> Dict[str, object]:
        return _load_definitions(["amdsmi_get_nic_port_ifindex"], _namespace())

    def test_returns_a_value_above_a_byte_whole(self) -> None:
        namespace = self._getter()
        result = namespace["amdsmi_get_nic_port_ifindex"](_FakeHandle(), _PORT_INDEX)
        self.assertEqual(result, _BEYOND_A_BYTE_IFINDEX)

    def test_passes_the_handle_and_port_index_to_the_library(self) -> None:
        namespace = self._getter()
        handle = _FakeHandle()
        namespace["amdsmi_get_nic_port_ifindex"](handle, _PORT_INDEX)
        self.assertEqual(namespace["amdsmi_wrapper"].ifindex_calls, [(handle, _PORT_INDEX)])

    def test_rejects_an_object_that_is_not_a_handle(self) -> None:
        namespace = self._getter()
        with self.assertRaises(_FakeParameterException):
            namespace["amdsmi_get_nic_port_ifindex"]("not a handle", _PORT_INDEX)
        self.assertEqual(namespace["amdsmi_wrapper"].ifindex_calls, [])

    def test_a_library_failure_is_raised_not_swallowed(self) -> None:
        namespace = self._getter()
        namespace["amdsmi_wrapper"].amdsmi_get_nic_port_ifindex = lambda handle, port_index, out: (
            _STATUS_FAILURE
        )
        with self.assertRaises(_FakeLibraryException):
            namespace["amdsmi_get_nic_port_ifindex"](_FakeHandle(), _PORT_INDEX)
