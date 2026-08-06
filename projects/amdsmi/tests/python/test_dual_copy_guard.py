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

"""Unit tests for the dual-copy drift guard's compare_trees logic.

Exercises the comparison used by run_amdsmi_dual_copy_test.py against synthetic
trees, so the guard is proven to catch drift without needing an installed
package or GPU.
"""

import importlib.util
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_guard():
    # In the source tree the guard lives at tests/run_amdsmi_dual_copy_test.py;
    # in the installed tests layout REPO_ROOT is share/amd_smi and it lives at
    # tests/run_amdsmi_dual_copy_test.py there too.
    for cand in (
        REPO_ROOT / "tests" / "run_amdsmi_dual_copy_test.py",
        REPO_ROOT / "run_amdsmi_dual_copy_test.py",
    ):
        if cand.is_file():
            spec = importlib.util.spec_from_file_location("amdsmi_dual_copy_guard", cand)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise unittest.SkipTest("run_amdsmi_dual_copy_test.py not found")


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CompareTreesTest(unittest.TestCase):
    def setUp(self):
        self.guard = _load_guard()
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.a = self.base / "a"
        self.b = self.base / "b"

    def tearDown(self):
        self._tmp.cleanup()

    def test_identical_trees_report_no_drift(self):
        _write(self.a, "amdsmi_interface.py", "x = 1\n")
        _write(self.b, "amdsmi_interface.py", "x = 1\n")
        missing, differing = self.guard.compare_trees(self.a, self.b)
        self.assertEqual(missing, [])
        self.assertEqual(differing, [])

    def test_differing_contents_are_flagged(self):
        _write(self.a, "amdsmi_interface.py", "x = 1\n")
        _write(self.b, "amdsmi_interface.py", "x = 2\n")
        missing, differing = self.guard.compare_trees(self.a, self.b)
        self.assertEqual(missing, [])
        self.assertIn("amdsmi_interface.py", differing)

    def test_missing_file_is_flagged(self):
        _write(self.a, "amdsmi_interface.py", "x = 1\n")
        _write(self.a, "extra.py", "y = 3\n")
        _write(self.b, "amdsmi_interface.py", "x = 1\n")
        missing, differing = self.guard.compare_trees(self.a, self.b)
        self.assertIn("extra.py", missing)
        self.assertEqual(differing, [])

    def test_non_python_files_are_ignored(self):
        _write(self.a, "amdsmi_interface.py", "x = 1\n")
        _write(self.b, "amdsmi_interface.py", "x = 1\n")
        # a bundled .so differing between trees must not count as drift
        (self.a / "libamd_smi_python.so").write_bytes(b"AAAA")
        missing, differing = self.guard.compare_trees(self.a, self.b)
        self.assertEqual(missing, [])
        self.assertEqual(differing, [])


if __name__ == "__main__":
    unittest.main()
