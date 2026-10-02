#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC telemetry: the health reporter name is ``N/A`` when the NIC exposes no reporter.

The C layer documents ``reporter`` as ``""`` when none exists; the Python dict must show the
same ``N/A`` marker as every other unavailable telemetry field. The shipped
``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the decode is compiled
from the real source and run against a fake wrapper.
"""

import ast
import ctypes
import types
import unittest
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INTERFACE_PATH = _REPO_ROOT / "py-interface" / "amdsmi_interface.py"

_STATUS_SUCCESS = 0
_REPORTER_BYTES = 64
_HEALTH_UNKNOWN = 0
_HEALTH_HEALTHY = 1
_HEALTH_WARNING = 2
_HEALTH_ERROR = 3
_HEALTH_UNSUPPORTED = 4
_TEMP_UNSUPPORTED = 0xFFFF
_COUNT_UNSUPPORTED = 0xFFFFFFFF
_BYTE_UNSUPPORTED = 0xFF
_ASIC_TEMP_C = 32


class _FakeParameterException(Exception):
    """Stands in for AmdSmiParameterException."""


class _FakeHandle:
    """Stands in for the ctypes amdsmi_processor_handle type."""


class _Temperature(ctypes.Structure):
    _fields_ = [
        ("asic_temp_c", ctypes.c_uint16),
        ("transceiver_temp_c", ctypes.c_uint16),
        ("board_temp_c", ctypes.c_uint16),
    ]


class _Health(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_uint8),
        ("error_count", ctypes.c_uint32),
        ("reporter", ctypes.c_char * _REPORTER_BYTES),
    ]


class _PortSplit(ctypes.Structure):
    _fields_ = [("splittable", ctypes.c_uint8), ("split_count", ctypes.c_uint8)]


class _Telemetry(ctypes.Structure):
    _fields_ = [("temperature", _Temperature), ("health", _Health), ("port_split", _PortSplit)]


def _fake_check_res(status: int) -> None:
    if status != _STATUS_SUCCESS:
        raise AssertionError(f"unexpected status {status}")


def _definition_name(node: ast.stmt) -> str:
    """Name bound by a top-level function, class or single-target assignment; "" otherwise."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return ""


def _load_definitions(names: List[str], namespace: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the named top-level definitions from the shipped interface source."""
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    wanted = [node for node in tree.body if _definition_name(node) in names]
    missing = set(names) - {_definition_name(node) for node in wanted}
    if missing:
        raise AssertionError(f"not found in amdsmi_interface.py: {sorted(missing)}")
    exec(compile(ast.Module(wanted, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace


def _decode(state: int, error_count: int, reporter: bytes) -> Dict[str, Any]:
    """Run the shipped amdsmi_get_nic_telemetry against a fake library reporting these values."""

    def fake_get_nic_telemetry(handle: object, out: ctypes.pointer) -> int:
        telem = out._obj
        telem.temperature.asic_temp_c = _ASIC_TEMP_C
        telem.temperature.transceiver_temp_c = _TEMP_UNSUPPORTED
        telem.temperature.board_temp_c = _TEMP_UNSUPPORTED
        telem.health.state = state
        telem.health.error_count = error_count
        telem.health.reporter = reporter
        telem.port_split.splittable = _BYTE_UNSUPPORTED
        telem.port_split.split_count = _BYTE_UNSUPPORTED
        return _STATUS_SUCCESS

    wrapper = types.SimpleNamespace(
        AMDSMI_NIC_HEALTH_UNKNOWN=_HEALTH_UNKNOWN,
        AMDSMI_NIC_HEALTH_HEALTHY=_HEALTH_HEALTHY,
        AMDSMI_NIC_HEALTH_WARNING=_HEALTH_WARNING,
        AMDSMI_NIC_HEALTH_ERROR=_HEALTH_ERROR,
        AMDSMI_NIC_HEALTH_UNSUPPORTED=_HEALTH_UNSUPPORTED,
        amdsmi_processor_handle=_FakeHandle,
        amdsmi_nic_telemetry_t=_Telemetry,
        amdsmi_get_nic_telemetry=fake_get_nic_telemetry,
    )
    namespace = _load_definitions(
        ["AmdSmiNicHealthState", "amdsmi_get_nic_telemetry"],
        {
            "amdsmi_wrapper": wrapper,
            "ctypes": ctypes,
            "IntEnum": IntEnum,
            "Dict": Dict,
            "Any": Any,
            "AmdSmiParameterException": _FakeParameterException,
            "_check_res": _fake_check_res,
        },
    )
    return namespace["amdsmi_get_nic_telemetry"](_FakeHandle())


class TestNicTelemetryReporter(unittest.TestCase):
    def test_no_reporter_reports_na(self) -> None:
        # A UALoE endpoint registers no health reporter; the C layer leaves the name empty.
        health = _decode(_HEALTH_UNSUPPORTED, _COUNT_UNSUPPORTED, b"")["health"]
        self.assertEqual(health["reporter"], "N/A")

    def test_a_named_reporter_is_passed_through(self) -> None:
        health = _decode(_HEALTH_HEALTHY, 0, b"fw")["health"]
        self.assertEqual(health["reporter"], "fw")

    def test_the_other_unavailable_health_fields_still_report_na(self) -> None:
        health = _decode(_HEALTH_UNSUPPORTED, _COUNT_UNSUPPORTED, b"")["health"]
        self.assertEqual(health["state"], "UNSUPPORTED")
        self.assertEqual(health["error_count"], "N/A")
