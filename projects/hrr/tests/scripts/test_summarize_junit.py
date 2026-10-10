#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for summarize_junit.py."""

import contextlib
import io
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_catch2  # noqa: E402
import summarize_junit  # noqa: E402


def junit(*records: str) -> str:
    return "<testsuites><testsuite>" + "".join(records) + "</testsuite></testsuites>"


def record(name: str, kind: str | None = None, message: str = "") -> str:
    if kind is None:
        return f'<testcase name="{name}"/>'
    attr = f' message="{message}"' if message else ""
    return f'<testcase name="{name}"><{kind}{attr}>text</{kind}></testcase>'


class ScratchDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text)
        return path


class CaseTest(unittest.TestCase):
    def test_case_result(self):
        parse = ET.fromstring
        self.assertEqual(summarize_junit.case_result(parse(record("a"))), "PASS")
        self.assertEqual(summarize_junit.case_result(parse(record("a", "skipped"))), "SKIP")
        self.assertEqual(summarize_junit.case_result(parse(record("a", "failure"))), "FAIL")
        self.assertEqual(summarize_junit.case_result(parse(record("a", "error"))), "FAIL")

    def test_group_cases_and_result(self):
        root = ET.fromstring(junit(record("A"), record("B/x"), record("B/y", "failure"), '<testcase/>'))
        groups = summarize_junit.group_cases(root)
        self.assertEqual(list(groups), ["A", "B"])
        self.assertEqual(summarize_junit.group_result(groups["A"]), "PASS")
        self.assertEqual(summarize_junit.group_result(groups["B"]), "FAIL")

    def test_grouping_agrees_with_run_catch2(self):
        # The workflow checks the two scripts out separately, so they cannot
        # share code; they must not drift apart.
        samples = [
            junit(record("A"), record("A/s", "failure")),
            junit(record("A/s", "skipped"), record("A/t")),
            junit(record("A", "skipped"), record("B"), record("C/x", "error")),
            junit('<testcase name="D"><skipped message="hrr-config: linux/gfx1151: because"/></testcase>'),
        ]
        for sample in samples:
            root = ET.fromstring(sample)
            mine = {k: summarize_junit.group_result(v) for k, v in summarize_junit.group_cases(root).items()}
            theirs = {k: run_catch2.group_result(v) for k, v in run_catch2.group_cases(root).items()}
            self.assertEqual(mine, theirs)


class FormatTest(unittest.TestCase):
    def test_cell(self):
        self.assertEqual(summarize_junit.cell(None), "—")
        self.assertEqual(summarize_junit.cell(0), "0")

    def test_failed_cell(self):
        self.assertEqual(summarize_junit.failed_cell([]), "")
        self.assertEqual(summarize_junit.failed_cell(["a", "b"]), "`a`, `b`")
        names = [f"n{i}" for i in range(summarize_junit.MAX_FAILED_NAMES + 3)]
        text = summarize_junit.failed_cell(names)
        self.assertIn("`n7`", text)
        self.assertNotIn("`n8`", text)
        self.assertTrue(text.endswith("… +3"))


class SummarizeTest(ScratchDir):
    def test_summarize_xml_counts_cases_not_sections(self):
        path = self.write(
            "x.xml",
            junit(
                record("Pass"),
                record("Skip", "skipped"),
                record("PassWithBadSection"),
                record("PassWithBadSection/s", "failure"),
                record("OnlySection/s", "failure"),
            ),
        )
        data = summarize_junit.summarize_xml(path)
        self.assertEqual(data["counts"], {"PASS": 1, "FAIL": 2, "SKIP": 1, "CONFIG": 0})
        self.assertEqual(data["failed"], ["PassWithBadSection", "OnlySection"])
        self.assertEqual(data["total"], 4)

    def test_config_skip_is_not_a_code_skip(self):
        path = self.write(
            "x.xml",
            junit(
                record("Code", "skipped"),
                record("Ruled", "skipped", "hrr-config: linux/gfx1151: because"),
            ),
        )
        data = summarize_junit.summarize_xml(path)
        self.assertEqual(data["counts"], {"PASS": 0, "FAIL": 0, "SKIP": 1, "CONFIG": 1})

    def test_try_summarize_reports_unreadable_files(self):
        path = self.write("bad.xml", "<testsuites><oops")
        errors: list[str] = []
        data = summarize_junit.try_summarize(path, errors)
        self.assertIn("error", data)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("bad.xml:"))
        errors = []
        data = summarize_junit.try_summarize(self.dir / "missing.xml", errors)
        self.assertIn("error", data)

    def test_render_marks_missing_unreadable_and_extra(self):
        self.write("unit-ubuntu-24.04.xml", junit(record("A"), record("B", "failure")))
        self.write("integration-gfx90a.xml", "<broken")
        self.write("integration-extra.xml", junit(record("Z")))
        errors: list[str] = []
        text = summarize_junit.render(self.dir, errors)
        self.assertIn("| Unit | Linux | 1 | 1 | 0 | 0 | `B` |", text)
        self.assertIn("| Unit | Windows | — | — | — | — | _no results_ |", text)
        self.assertIn("| Integration | gfx90a | — | — | — | — | _unreadable XML:", text)
        self.assertIn("| Other | integration-extra.xml | 1 | 0 | 0 | 0 |  |", text)
        self.assertIn("### Counts grid", text)
        self.assertEqual(len(errors), 1)

    def test_render_with_no_files_lists_every_column(self):
        text = summarize_junit.render(self.dir, [])
        for _, _, platform in summarize_junit.COLUMNS:
            self.assertIn(f"| {platform} |", text)


class MainTest(ScratchDir):
    def run_main(self, *argv: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = summarize_junit.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_not_a_directory(self):
        rc, _, err = self.run_main("--dir", str(self.dir / "nope"))
        self.assertEqual(rc, 1)
        self.assertIn("not a directory", err)

    def test_writes_stdout_and_appends_to_the_summary_file(self):
        self.write("unit-windows-2022.xml", junit(record("A")))
        summary = self.write("summary.md", "existing\n")
        rc, out, _ = self.run_main("--dir", str(self.dir), "--summary", str(summary))
        self.assertEqual(rc, 0)
        self.assertIn("| Unit | Windows | 1 | 0 | 0 | 0 |", out)
        written = summary.read_text()
        self.assertTrue(written.startswith("existing\n"))
        self.assertIn("## HRR results by platform", written)

    def test_unreadable_xml_exits_1_with_an_annotation(self):
        self.write("integration-gfx950.xml", "")
        rc, out, _ = self.run_main("--dir", str(self.dir))
        self.assertEqual(rc, 1)
        self.assertIn("::error::unreadable JUnit XML integration-gfx950.xml", out)


if __name__ == "__main__":
    unittest.main()
