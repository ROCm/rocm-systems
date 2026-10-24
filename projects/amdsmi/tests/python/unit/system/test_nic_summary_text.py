#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""NIC summary: device-supplied text that is not valid UTF-8 must not raise.

``AMDSMIParser`` builds its NIC choices from ``amdsmi_get_ainic_info`` on every run, so one NIC
reporting a non-UTF-8 byte in a VPD or devlink string would abort every ``amd-smi`` command. The
shipped ``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the function is
compiled from the real source and run against a fake NIC.
"""

import ast
import types
import unittest
from typing import Any, Dict

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_FUNCTION_NAME = "amdsmi_get_ainic_info_summary"
_REPLACEMENT_CHAR = "\N{REPLACEMENT CHARACTER}"


def _load_summary() -> Any:
    """Compile the shipped summary function with its helpers replaced by stubs."""
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
        "_nic_type_label": lambda nic_type, vendor_name: vendor_name,
        "_decode_nic_capabilities": lambda bitmask: [],
        "amdsmi_get_nic_type": lambda handle: 0,
        "amdsmi_get_nic_capability": lambda handle: 0,
    }
    exec(compile(ast.Module(nodes, []), str(_INTERFACE_PATH), "exec"), namespace)
    return namespace[_FUNCTION_NAME]


def _nic_info(serial: bytes) -> types.SimpleNamespace:
    asic = types.SimpleNamespace(
        vendor_name=b"AMD",
        permanent_address=b"00:11:22:33:44:55",
        product_name=b"Pollara",
        part_number=b"PN-1",
        serial_number=serial,
    )
    return types.SimpleNamespace(handle=object(), bus=types.SimpleNamespace(bdf=0), asic=asic)


class TestNicSummaryText(unittest.TestCase):
    def test_ascii_text_is_unchanged(self) -> None:
        summary = _load_summary()(_nic_info(b"SN12345"))
        self.assertEqual(summary["Serial Number"], "SN12345")

    def test_non_utf8_text_is_replaced_not_raised(self) -> None:
        summary = _load_summary()(_nic_info(b"AB\xe9\xff"))
        self.assertEqual(summary["Serial Number"], "AB" + _REPLACEMENT_CHAR * 2)
        self.assertEqual(summary["Product Name"], "Pollara")
