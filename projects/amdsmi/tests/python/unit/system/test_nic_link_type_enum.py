#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Link types a NIC pair can report: ``AmdSmiLinkType`` names them.

``amdsmi_topo_get_link_type`` returns PCIE, NUMA, XNUMA or UNKNOWN for a NIC pair. The shipped
``amdsmi_interface.py`` needs a built ``libamd_smi.so`` to import, so the enum class is read from
the source instead.
"""

import ast
import unittest
from typing import Dict, List

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_WRAPPER_PATH = _INTERFACE_PATH.with_name("amdsmi_wrapper.py")
_ENUM_NAME = "AmdSmiLinkType"
# AMDSMI_LINK_TYPE_NUMA and _XNUMA in amdsmi.h
_NUMA_VALUE = 5
_XNUMA_VALUE = 6


def _enum_assignments() -> Dict[str, ast.expr]:
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == _ENUM_NAME]
    if len(classes) != 1:
        raise AssertionError(f"{_ENUM_NAME} not found in amdsmi_interface.py")
    return {
        target.id: node.value
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def _enum_members() -> List[str]:
    return list(_enum_assignments())


def _wrapper_values() -> Dict[str, int]:
    """The integer constants of the generated wrapper, which is built from ``amdsmi.h``."""
    tree = ast.parse(_WRAPPER_PATH.read_text(encoding="utf-8"))
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and isinstance(node.value.value, int)
    }


class TestNicLinkTypeEnum(unittest.TestCase):
    def test_the_nic_link_types_are_named(self) -> None:
        members = _enum_members()

        for name in ("AMDSMI_LINK_TYPE_PCIE", "AMDSMI_LINK_TYPE_NUMA", "AMDSMI_LINK_TYPE_XNUMA"):
            self.assertIn(name, members)

    def test_the_unknown_link_type_is_still_named(self) -> None:
        self.assertIn("AMDSMI_LINK_TYPE_UNKNOWN", _enum_members())

    def test_each_member_takes_the_value_of_the_wrapper_constant_of_its_name(self) -> None:
        for name, value in _enum_assignments().items():
            self.assertIsInstance(value, ast.Attribute, name)
            self.assertEqual((value.value.id, value.attr), ("amdsmi_wrapper", name))

    def test_the_nic_link_types_have_the_values_of_the_header(self) -> None:
        values = _wrapper_values()

        self.assertEqual(values["AMDSMI_LINK_TYPE_NUMA"], _NUMA_VALUE)
        self.assertEqual(values["AMDSMI_LINK_TYPE_XNUMA"], _XNUMA_VALUE)
