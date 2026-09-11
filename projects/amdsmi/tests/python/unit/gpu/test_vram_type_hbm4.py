#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the ``AMDSMI_VRAM_TYPE_HBM4`` / ``HBM3E`` enum values.

``py-interface`` is loaded directly from the source tree, without executing
its package ``__init__.py`` (which requires a CMake-generated ``_version.py``),
so the test exercises the regenerated wrapper and interface rather than a
possibly-stale installed copy.
"""

import importlib
import os
import sys
import types
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
PY_INTERFACE_DIR = os.path.join(_REPO_ROOT, "py-interface")


def _load_amdsmi_interface_module():
    amdsmi_pkg = types.ModuleType("amdsmi")
    amdsmi_pkg.__path__ = [PY_INTERFACE_DIR]
    sys.modules["amdsmi"] = amdsmi_pkg
    return importlib.import_module("amdsmi.amdsmi_interface")


class TestVramTypeHbm4(unittest.TestCase):
    def test_hbm4_value_and_name_mapping(self):
        interface = _load_amdsmi_interface_module()
        wrapper = interface.amdsmi_wrapper

        self.assertEqual(wrapper.AMDSMI_VRAM_TYPE_HBM4, 6)
        self.assertEqual(wrapper.amdsmi_vram_type_t__enumvalues[6], "AMDSMI_VRAM_TYPE_HBM4")
        self.assertEqual(interface.AmdSmiVramType.HBM4, 6)

    def test_hbm3e_value_and_name_mapping(self):
        interface = _load_amdsmi_interface_module()
        wrapper = interface.amdsmi_wrapper

        self.assertEqual(wrapper.AMDSMI_VRAM_TYPE_HBM3E, 5)
        self.assertEqual(wrapper.amdsmi_vram_type_t__enumvalues[5], "AMDSMI_VRAM_TYPE_HBM3E")
        self.assertEqual(interface.AmdSmiVramType.HBM3E, 5)


if __name__ == "__main__":
    unittest.main()
