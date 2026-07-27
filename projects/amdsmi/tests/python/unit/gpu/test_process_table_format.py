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

"""Unit tests for the ``amd-smi`` default-output process-table formatting.

Loads the installed ``amdsmi_logger`` with its only non-stdlib dependency
(``amdsmi_helpers``) stubbed, so the column-alignment logic is exercised
without GPU hardware or the compiled ``amdsmi`` package. Pins the fix that
aligned the ``CU %``/``SDMA`` columns and dropped the redundant ``%`` suffix.
"""

import importlib.util
import os
import sys
import types
import unittest

from common.common import amdsmi_path

_ROCM_ROOT = os.path.dirname(os.path.dirname(amdsmi_path))
LOGGER_PATH = os.path.join(_ROCM_ROOT, "libexec", "amdsmi_cli", "amdsmi_logger.py")

# Fixed inner width of the default-output box (between the two '|' borders).
_BOX_INNER_WIDTH = 78


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


def _process(name="python3", cu=None, sdma="0", gpu="0", pid="12345"):
    """Build a minimal process dict matching the default-output payload."""
    if cu is None:
        cu_occupancy = {"total_num_cu": "N/A", "current_cu": "N/A"}
    else:
        cu_occupancy = {"total_num_cu": 100, "current_cu": cu}
    return {
        "gpu": gpu,
        "pid": pid,
        "name": name,
        "gtt": "0",
        "vram": "0",
        "mem_usage": "0",
        "cu_occupancy": cu_occupancy,
        "sdma_usage": sdma,
    }


class TestProcessTableFormat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(LOGGER_PATH):
            raise unittest.SkipTest(f"amdsmi_logger not installed at {LOGGER_PATH}")
        _install_fake_helpers()
        cls.logger = _load_logger_module()

    def _assert_boxed(self, line):
        self.assertTrue(line.startswith("|") and line.endswith("|"), line)
        self.assertEqual(len(line), _BOX_INNER_WIDTH + 2, f"line width {len(line)}: {line!r}")

    def test_header_fits_box(self):
        self._assert_boxed(self.logger.AMDSMILogger.PROCESS_TABLE_HEADER)

    def test_row_fits_box(self):
        row = self.logger.AMDSMILogger._format_process_row(_process())
        self._assert_boxed(row)

    def test_na_row_fits_box(self):
        row = self.logger.AMDSMILogger._format_process_row(_process(name="N/A"))
        self._assert_boxed(row)

    def test_cu_value_has_no_percent_suffix(self):
        # 12.5% must render as "12.5", not "12.5 %" (which overflowed the column).
        row = self.logger.AMDSMILogger._format_process_row(_process(cu=12.5))
        self.assertIn("12.5", row)
        self.assertNotIn("%", row)
        self._assert_boxed(row)

    def test_sdma_and_cu_columns_align_with_header(self):
        # Header labels must sit over the value fields. Locate the value column
        # spans from the format widths and confirm the header keyword sits there.
        header = self.logger.AMDSMILogger.PROCESS_TABLE_HEADER
        # Sentinels chosen to not collide with the fixed pid ("40002") elsewhere.
        row = self.logger.AMDSMILogger._format_process_row(
            _process(cu=99.9, sdma="765", pid="40002")
        )
        # CU value 99.9 and SDMA value 765 must both be present and sit at or
        # right of their respective header labels (right-justified in-column).
        self.assertIn("99.9", row)
        self.assertIn("765", row)
        self.assertLessEqual(header.index("CU %"), row.index("99.9"))
        self.assertLessEqual(header.index("SDMA"), row.index("765"))
        self._assert_boxed(row)


if __name__ == "__main__":
    unittest.main()
