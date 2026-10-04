#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC type: the interface helpers that decode and fetch ``amdsmi_nic_type_t``.

The shipped ``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so each test
compiles only the definitions it exercises from the real source and runs them against a fake
wrapper. A change to the shipped code therefore still fails here.
"""

import ast
import ctypes
import types
import unittest
from enum import IntEnum
from pathlib import Path
from typing import Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERFACE_PATH = _REPO_ROOT / "py-interface" / "amdsmi_interface.py"

_NIC_TYPE_UNKNOWN = 0
_NIC_TYPE_AINIC = 1
_NIC_TYPE_UALOE = 2
_NIC_TYPE_OTHER = 3
_UNDEFINED_NIC_TYPE = 9
_STATUS_SUCCESS = 0
_STATUS_FAILURE = 1


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


_VENDOR_BUFFER_BYTES = 256


class _FakeAsicInfo(ctypes.Structure):
    """Stands in for amdsmi_nic_asic_info_t; only the field the label reads."""

    _fields_ = [("vendor_name", ctypes.c_char * _VENDOR_BUFFER_BYTES)]


def _fake_wrapper(reported_type: int, vendor: str = "") -> types.SimpleNamespace:
    """A wrapper whose amdsmi_get_nic_type reports ``reported_type``."""
    calls: List[object] = []
    asic_calls: List[object] = []

    def fake_get_nic_asic_info(handle: object, out: ctypes.pointer) -> int:
        asic_calls.append(handle)
        out._obj.vendor_name = vendor.encode("utf-8")
        return _STATUS_SUCCESS

    def fake_get_nic_type(handle: object, out: ctypes.pointer) -> int:
        calls.append(handle)
        out._obj.value = reported_type
        return _STATUS_SUCCESS

    return types.SimpleNamespace(
        AMDSMI_NIC_TYPE_UNKNOWN=_NIC_TYPE_UNKNOWN,
        AMDSMI_NIC_TYPE_AINIC=_NIC_TYPE_AINIC,
        AMDSMI_NIC_TYPE_UALOE=_NIC_TYPE_UALOE,
        AMDSMI_NIC_TYPE_OTHER=_NIC_TYPE_OTHER,
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_nic_type_t=ctypes.c_uint32,
        amdsmi_get_nic_type=fake_get_nic_type,
        amdsmi_nic_asic_info_t=_FakeAsicInfo,
        amdsmi_get_nic_asic_info=fake_get_nic_asic_info,
        calls=calls,
        asic_calls=asic_calls,
    )


def _namespace(reported_type: int, vendor: str = "") -> Dict[str, object]:
    wrapper = _fake_wrapper(reported_type, vendor)
    return {
        "amdsmi_wrapper": wrapper,
        "ctypes": ctypes,
        "IntEnum": IntEnum,
        "AmdSmiParameterException": _FakeParameterException,
        "_check_res": _fake_check_res,
    }


class TestNicTypeFromValue(unittest.TestCase):
    """The decode that turns the C byte into AmdSmiNicType."""

    def _decode(self, value: int) -> object:
        namespace = _load_definitions(["AmdSmiNicType", "_nic_type_from_value"], _namespace(0))
        return namespace["_nic_type_from_value"](value)

    def test_each_defined_value_maps_to_its_member(self) -> None:
        expected = {
            _NIC_TYPE_UNKNOWN: "UNKNOWN",
            _NIC_TYPE_AINIC: "AINIC",
            _NIC_TYPE_UALOE: "UALOE",
            _NIC_TYPE_OTHER: "OTHER",
        }
        for value, name in expected.items():
            with self.subTest(value=value):
                self.assertEqual(self._decode(value).name, name)

    def test_a_value_past_the_enum_degrades_to_unknown(self) -> None:
        # A newer library may report a type this interface does not know; the caller must get
        # UNKNOWN, not a bare ValueError that bypasses the AmdSmi exception types.
        self.assertEqual(self._decode(_UNDEFINED_NIC_TYPE).name, "UNKNOWN")


class TestGetNicType(unittest.TestCase):
    """amdsmi_get_nic_type wraps the C getter."""

    def _namespace_with_getter(self, reported_type: int) -> Dict[str, object]:
        return _load_definitions(
            ["AmdSmiNicType", "_nic_type_from_value", "amdsmi_get_nic_type"],
            _namespace(reported_type),
        )

    def test_returns_the_type_the_library_reports(self) -> None:
        for value, name in ((_NIC_TYPE_AINIC, "AINIC"), (_NIC_TYPE_UALOE, "UALOE")):
            with self.subTest(name=name):
                namespace = self._namespace_with_getter(value)
                result = namespace["amdsmi_get_nic_type"](_FakeHandle())
                self.assertEqual(result.name, name)

    def test_passes_the_handle_to_the_library(self) -> None:
        namespace = self._namespace_with_getter(_NIC_TYPE_OTHER)
        handle = _FakeHandle()
        namespace["amdsmi_get_nic_type"](handle)
        self.assertEqual(namespace["amdsmi_wrapper"].calls, [handle])

    def test_rejects_an_object_that_is_not_a_handle(self) -> None:
        namespace = self._namespace_with_getter(_NIC_TYPE_AINIC)
        with self.assertRaises(_FakeParameterException):
            namespace["amdsmi_get_nic_type"]("not a handle")
        self.assertEqual(namespace["amdsmi_wrapper"].calls, [])

    def test_a_library_failure_is_raised_not_swallowed(self) -> None:
        namespace = self._namespace_with_getter(_NIC_TYPE_AINIC)
        namespace["amdsmi_wrapper"].amdsmi_get_nic_type = lambda handle, out: _STATUS_FAILURE
        with self.assertRaises(_FakeLibraryException):
            namespace["amdsmi_get_nic_type"](_FakeHandle())


class TestNicTypeLabel(unittest.TestCase):
    """The name a NIC type is shown as; OTHER shows the vendor so unlike cards stay distinct."""

    _DEFINITIONS = ["AmdSmiNicType", "_NIC_TYPE_LABELS", "_nic_type_label"]

    def _label(self, nic_type: str, vendor_name: str) -> str:
        namespace = _load_definitions(self._DEFINITIONS, _namespace(0))
        return namespace["_nic_type_label"](namespace["AmdSmiNicType"][nic_type], vendor_name)

    def test_ainic_is_shown_as_ainic(self) -> None:
        self.assertEqual(self._label("AINIC", "AMD Pensando Systems, Inc."), "AINIC")

    def test_ualoe_is_shown_with_its_mixed_case_name(self) -> None:
        self.assertEqual(self._label("UALOE", "AMD"), "UALoE")

    def test_other_shows_the_vendor_name(self) -> None:
        self.assertEqual(self._label("OTHER", "Broadcom Inc."), "Broadcom Inc.")

    def test_other_without_a_vendor_name_is_na(self) -> None:
        self.assertEqual(self._label("OTHER", ""), "N/A")

    def test_unknown_is_na_whatever_the_vendor(self) -> None:
        self.assertEqual(self._label("UNKNOWN", "AMD"), "N/A")


class TestAinicInfoSummaryType(unittest.TestCase):
    """The summary dict that list --nic renders carries the display name under "Type"."""

    def _summary(self, reported_type: int, vendor_name: str) -> Dict[str, object]:
        namespace = _namespace(0)
        namespace.update(
            {
                "_format_bdf": lambda bdf: "0000:01:00.0",
                "_decode_nic_capabilities": lambda bitmask: [],
            }
        )
        _load_definitions(
            [
                "AmdSmiNicType",
                "_NIC_TYPE_LABELS",
                "_nic_type_from_value",
                "_nic_type_label",
                "amdsmi_get_ainic_info_summary",
            ],
            namespace,
        )
        asic = types.SimpleNamespace(
            type=reported_type,
            capability=0,
            permanent_address=b"aa:bb",
            product_name=b"product",
            part_number=b"part",
            serial_number=b"serial",
            vendor_name=vendor_name.encode("utf-8"),
        )
        nic_info = types.SimpleNamespace(bus=types.SimpleNamespace(bdf=0), asic=asic)
        return namespace["amdsmi_get_ainic_info_summary"](nic_info)

    def test_ainic_reports_ainic(self) -> None:
        self.assertEqual(
            self._summary(_NIC_TYPE_AINIC, "AMD Pensando Systems, Inc.")["Type"], "AINIC"
        )

    def test_ualoe_endpoint_reports_ualoe(self) -> None:
        self.assertEqual(self._summary(_NIC_TYPE_UALOE, "AMD")["Type"], "UALoE")

    def test_other_vendor_reports_its_name(self) -> None:
        self.assertEqual(self._summary(_NIC_TYPE_OTHER, "Broadcom Inc.")["Type"], "Broadcom Inc.")

    def test_an_undefined_type_reports_na(self) -> None:
        self.assertEqual(self._summary(_UNDEFINED_NIC_TYPE, "AMD")["Type"], "N/A")


class TestGetNicTypeLabel(unittest.TestCase):
    """The label metric --nic prints, read through the type getter and nothing else."""

    _DEFINITIONS = [
        "AmdSmiNicType",
        "_NIC_TYPE_LABELS",
        "_nic_type_from_value",
        "_nic_type_label",
        "amdsmi_get_nic_type",
        "amdsmi_get_nic_type_label",
    ]

    def _namespace_with_label(self, reported_type: int, vendor: str = "") -> Dict[str, object]:
        return _load_definitions(self._DEFINITIONS, _namespace(reported_type, vendor))

    def test_ainic_and_ualoe_are_labelled_without_reading_the_asic_info(self) -> None:
        for value, label in ((_NIC_TYPE_AINIC, "AINIC"), (_NIC_TYPE_UALOE, "UALoE")):
            with self.subTest(label=label):
                namespace = self._namespace_with_label(value)
                self.assertEqual(namespace["amdsmi_get_nic_type_label"](_FakeHandle()), label)
                # Only OTHER needs the vendor, so the asic getter must not run.
                self.assertEqual(namespace["amdsmi_wrapper"].asic_calls, [])

    def test_other_is_labelled_with_the_vendor_name(self) -> None:
        namespace = self._namespace_with_label(_NIC_TYPE_OTHER, "Broadcom Inc.")
        handle = _FakeHandle()
        self.assertEqual(namespace["amdsmi_get_nic_type_label"](handle), "Broadcom Inc.")
        self.assertEqual(namespace["amdsmi_wrapper"].asic_calls, [handle])

    def test_an_undefined_type_is_labelled_na(self) -> None:
        namespace = self._namespace_with_label(_UNDEFINED_NIC_TYPE)
        self.assertEqual(namespace["amdsmi_get_nic_type_label"](_FakeHandle()), "N/A")

    def test_rejects_an_object_that_is_not_a_handle(self) -> None:
        namespace = self._namespace_with_label(_NIC_TYPE_AINIC)
        with self.assertRaises(_FakeParameterException):
            namespace["amdsmi_get_nic_type_label"]("not a handle")
