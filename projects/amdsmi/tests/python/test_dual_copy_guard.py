#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
