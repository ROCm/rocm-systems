#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_expected_cases.py.

subprocess.run is replaced so no Catch2 binary is needed.
"""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_expected_cases as cec  # noqa: E402


def listing(*names: str) -> str:
    body = "".join(f"<TestCase><Name>{n}</Name></TestCase>" for n in names)
    return f"<Catch2TestRun>{body}</Catch2TestRun>"


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class ListedCasesTest(unittest.TestCase):
    def test_sorted_names_and_the_star_selector(self):
        with mock.patch.object(cec.subprocess, "run", return_value=completed(listing("b", "a"))) as run:
            self.assertEqual(cec.listed_cases("./bin"), ["a", "b"])
        # Without "*" Catch2 omits hidden cases, which is most of the suite.
        self.assertEqual(run.call_args.args[0], ["./bin", "--list-tests", "*", "--reporter", "xml"])

    def test_failing_listing_exits(self):
        with mock.patch.object(cec.subprocess, "run", return_value=completed("", 2, "bad flag")):
            with self.assertRaises(SystemExit) as caught:
                cec.listed_cases("./bin")
        self.assertIn("failed (2)", str(caught.exception))
        self.assertIn("bad flag", str(caught.exception))

    def test_unparseable_listing_exits(self):
        with mock.patch.object(cec.subprocess, "run", return_value=completed("<oops")):
            with self.assertRaises(SystemExit) as caught:
                cec.listed_cases("./bin")
        self.assertIn("could not parse", str(caught.exception))

    def test_empty_listing_exits(self):
        with mock.patch.object(cec.subprocess, "run", return_value=completed(listing())):
            with self.assertRaises(SystemExit) as caught:
                cec.listed_cases("./bin")
        self.assertIn("listed no test cases", str(caught.exception))


class ScratchDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = Path(self.tmp.name) / "expected.txt"


class ReadManifestTest(ScratchDir):
    def test_skips_comments_and_blank_lines_and_sorts(self):
        self.manifest.write_text("# header\n\nzeta\n  alpha  \n# note\n")
        self.assertEqual(cec.read_manifest(str(self.manifest)), ["alpha", "zeta"])

    def test_missing_manifest_exits(self):
        with self.assertRaises(SystemExit) as caught:
            cec.read_manifest(str(self.manifest))
        self.assertIn("not found", str(caught.exception))


class MainTest(ScratchDir):
    def run_main(self, actual: list[str], *extra: str):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(cec, "listed_cases", return_value=sorted(actual)):
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = cec.main(["--binary", "./bin", "--manifest", str(self.manifest), *extra])
        return rc, out.getvalue(), err.getvalue()

    def test_matching_binary_passes(self):
        self.manifest.write_text("# h\na\nb\n")
        rc, out, _ = self.run_main(["b", "a"])
        self.assertEqual(rc, 0)
        self.assertIn("OK: 2 test cases present", out)

    def test_missing_case_fails_with_the_guard_hint(self):
        self.manifest.write_text("a\nb\n")
        rc, _, err = self.run_main(["a"])
        self.assertEqual(rc, 1)
        self.assertIn("MISSING (compiled out or deleted): b", err)
        self.assertIn("a guard that stopped being satisfied", err)

    def test_unannounced_case_fails(self):
        self.manifest.write_text("a\n")
        rc, _, err = self.run_main(["a", "c"])
        self.assertEqual(rc, 1)
        self.assertIn("UNEXPECTED (added without updating manifest): c", err)
        self.assertNotIn("guard that stopped", err)

    def test_update_rewrites_the_manifest_and_it_round_trips(self):
        self.manifest.write_text("stale\n")
        rc, out, _ = self.run_main(["b", "a"], "--update")
        self.assertEqual(rc, 0)
        self.assertIn("wrote 2 case names", out)
        text = self.manifest.read_text()
        self.assertTrue(text.startswith("# Test cases expected"))
        self.assertNotIn("stale", text)
        rc, _, _ = self.run_main(["a", "b"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
