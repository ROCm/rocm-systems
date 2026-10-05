#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How run_catch2.py and summarize_junit.py read Catch2's JUnit output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_catch2 import top_level_results  # noqa: E402
from summarize_junit import summarize_xml  # noqa: E402

# Shaped like Catch2 3.8's JUnit reporter: the case, then one entry per
# section, with a section's assertion reported only on the section's entry.
JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="hrr-integration-tests" errors="0" failures="1" skipped="1" tests="7">
    <testcase classname="global" name="Plain_Pass" time="0.1"/>
    <testcase classname="global" name="Sections" time="0.1"/>
    <testcase classname="global" name="Sections/first" time="0.1"/>
    <testcase classname="global" name="Sections/second" time="0.1">
      <failure message="x == 0" type="CHECK">FAILED: CHECK( x == 0 )</failure>
    </testcase>
    <testcase classname="global" name="Sections/third" time="0.1"/>
    <testcase classname="global" name="Skipped" time="0.1">
      <skipped type="SKIP">SKIPPED</skipped>
    </testcase>
    <testcase classname="global" name="Clean_Sections" time="0.1"/>
    <testcase classname="global" name="Clean_Sections/only" time="0.1"/>
  </testsuite>
</testsuites>
"""


class TopLevelResultsTest(unittest.TestCase):
    def results(self):
        return top_level_results(ET.fromstring(JUNIT).iterfind(".//testcase"))

    def test_a_failing_section_fails_its_case(self):
        result, failures = self.results()["Sections"]
        self.assertEqual(result, "FAIL")
        self.assertEqual([entry for entry, _ in failures], ["Sections/second"])

    def test_sections_are_not_counted_as_cases(self):
        self.assertEqual(
            {name: result for name, (result, _) in self.results().items()},
            {
                "Plain_Pass": "PASS",
                "Sections": "FAIL",
                "Skipped": "SKIP",
                "Clean_Sections": "PASS",
            },
        )

    def test_the_summary_table_counts_the_same_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "integration-gfx90a.xml"
            path.write_text(JUNIT)
            summary = summarize_xml(path)
        self.assertEqual(summary["counts"], {"PASS": 2, "FAIL": 1, "SKIP": 1})
        self.assertEqual(summary["failed"], ["Sections"])
        self.assertEqual(summary["total"], 4)


class RunCatch2Test(unittest.TestCase):
    def test_reports_a_suite_whose_output_is_not_utf8(self):
        # A workload can print bytes that are not UTF-8 (0x82 on gfx1151 and
        # gfx90a runners); the summary must still come from the JUnit file.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "results.xml.in").write_text(JUNIT)
            suite = tmp / "suite.py"
            suite.write_text(textwrap.dedent("""\
                import shutil, sys
                from pathlib import Path
                out = next(a[len("junit::out="):] for a in sys.argv
                           if a.startswith("junit::out="))
                shutil.copy(Path(__file__).with_name("results.xml.in"), out)
                sys.stdout.buffer.write(b"workload says \\x81\\x82\\n")
                sys.exit(42)
                """))
            run = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_catch2.py"),
                    "--xml",
                    str(tmp / "results.xml"),
                    "--",
                    sys.executable,
                    str(suite),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        output = run.stdout.decode("utf-8", "replace")
        self.assertEqual(run.returncode, 42, output)
        self.assertIn("FAIL: Sections", output)
        self.assertIn("in section 'second':", output)
        self.assertIn("2 passed, 1 failed, 1 skipped", output)


if __name__ == "__main__":
    unittest.main()
