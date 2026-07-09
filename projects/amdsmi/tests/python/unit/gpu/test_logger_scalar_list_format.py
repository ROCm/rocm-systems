#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Mock-based unit tests for ``AMDSMILogger`` human-readable scalar-list output.

Loads the installed ``amdsmi_logger`` with its ``amdsmi_helpers`` dependency
stubbed, so the human-readable formatter is exercised without GPU hardware.
Locks in the scalar-list rendering contract used by ``amd-smi static --profile``
(and any section that emits a list of plain strings):

* A scalar list item is indented strictly deeper than its own key.
* Scalar list items are not prefixed with a ``- `` bullet.
"""

import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path

_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
LOGGER_PATH = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli", "amdsmi_logger.py")


def _install_fake_helpers():
    """Register a stub ``amdsmi_helpers`` so ``amdsmi_logger`` imports cleanly."""
    module = types.ModuleType("amdsmi_helpers")
    module.AMDSMIHelpers = type("AMDSMIHelpers", (), {})
    sys.modules["amdsmi_helpers"] = module


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("amdsmi_logger_under_test", LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indent(line):
    return len(line) - len(line.lstrip(" "))


class TestHumanReadableScalarList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(LOGGER_PATH):
            raise unittest.SkipTest(f"amdsmi_logger not installed at {LOGGER_PATH}")
        _install_fake_helpers()
        cls.logger_mod = _load_logger_module()
        cls.logger = cls.logger_mod.AMDSMILogger(format="human_readable")

    def _render_profile(self, profiles):
        return self.logger._convert_json_to_human_readable(
            {"gpu": 0, "profile": {"available_profiles": profiles}}
        )

    def test_items_indented_deeper_than_key(self):
        """Each item must sit deeper than its AVAILABLE_PROFILES key."""
        profiles = ["CUSTOM", "VIDEO", "BOOTUP_DEFAULT"]
        lines = self._render_profile(profiles).splitlines()

        key_lines = [line for line in lines if line.strip() == "AVAILABLE_PROFILES:"]
        self.assertEqual(len(key_lines), 1, "expected exactly one AVAILABLE_PROFILES key line")
        key_indent = _indent(key_lines[0])

        for name in profiles:
            item_lines = [line for line in lines if line.strip() == name]
            self.assertEqual(len(item_lines), 1, f"expected one line for profile {name}")
            self.assertGreater(
                _indent(item_lines[0]),
                key_indent,
                f"profile {name} should be indented deeper than its key",
            )

    def test_no_dash_bullets(self):
        """Scalar list items must not render as ``- item`` bullets."""
        lines = self._render_profile(["CUSTOM", "VIDEO"]).splitlines()
        for line in lines:
            self.assertFalse(
                line.lstrip(" ").startswith("- "),
                f"unexpected dash bullet in human-readable output: {line!r}",
            )

    def test_items_do_not_outdent_between_sibling_fields(self):
        """Items nest under the list key while sibling scalars keep key-level indent."""
        out = self.logger._convert_json_to_human_readable(
            {
                "gpu": 0,
                "profile": {
                    "available_profiles": ["CUSTOM", "VIDEO"],
                    "current": "CUSTOM",
                    "num_profiles": 2,
                },
            }
        )
        lines = out.splitlines()
        key_indent = _indent(next(l for l in lines if l.strip() == "AVAILABLE_PROFILES:"))
        current_indent = _indent(next(l for l in lines if l.strip().startswith("CURRENT:")))
        self.assertEqual(
            current_indent, key_indent, "sibling field CURRENT should share the list key's indent"
        )
        for name in ("CUSTOM", "VIDEO"):
            item_indent = _indent(next(l for l in lines if l.strip() == name))
            self.assertGreater(item_indent, current_indent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
