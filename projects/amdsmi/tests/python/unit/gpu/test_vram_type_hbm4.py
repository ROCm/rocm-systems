#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the ``AMDSMI_VRAM_TYPE_HBM4`` enum value.

``amdsmi_wrapper.py`` is loaded directly from the source tree (it has no
relative imports and does not load the compiled library at import time), so
the test exercises the regenerated wrapper rather than a possibly-stale
installed copy.
"""

import importlib.util
import os
import unittest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
WRAPPER_PATH = os.path.join(_REPO_ROOT, "py-interface", "amdsmi_wrapper.py")


def _load_wrapper_module():
    spec = importlib.util.spec_from_file_location("amdsmi_wrapper_under_test", WRAPPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVramTypeHbm4(unittest.TestCase):
    def test_hbm4_value_and_name_mapping(self):
        wrapper = _load_wrapper_module()

        self.assertEqual(wrapper.AMDSMI_VRAM_TYPE_HBM4, 6)
        self.assertEqual(wrapper.amdsmi_vram_type_t__enumvalues[6], "AMDSMI_VRAM_TYPE_HBM4")


if __name__ == "__main__":
    unittest.main()
