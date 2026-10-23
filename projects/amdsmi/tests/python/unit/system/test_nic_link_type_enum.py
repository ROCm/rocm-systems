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
from typing import List

from common.common import amdsmi_interface_path

_INTERFACE_PATH = amdsmi_interface_path(__file__)
_ENUM_NAME = "AmdSmiLinkType"


def _enum_members() -> List[str]:
    tree = ast.parse(_INTERFACE_PATH.read_text(encoding="utf-8"))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == _ENUM_NAME]
    if len(classes) != 1:
        raise AssertionError(f"{_ENUM_NAME} not found in amdsmi_interface.py")
    return [
        target.id
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


class TestNicLinkTypeEnum(unittest.TestCase):
    def test_the_nic_link_types_are_named(self) -> None:
        members = _enum_members()

        for name in ("AMDSMI_LINK_TYPE_PCIE", "AMDSMI_LINK_TYPE_NUMA", "AMDSMI_LINK_TYPE_XNUMA"):
            self.assertIn(name, members)

    def test_the_unknown_link_type_is_still_named(self) -> None:
        self.assertIn("AMDSMI_LINK_TYPE_UNKNOWN", _enum_members())


if __name__ == "__main__":
    unittest.main()
