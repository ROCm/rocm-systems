#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for run_catch2.py.

The Catch2 binary is replaced by a small Python script that writes whatever
JUnit the test asks for, prints a banner and exits with a chosen status, so the
whole wrapper (command line, subprocess, JUnit parsing, exit status) runs
without Catch2 or a GPU.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_test_config  # noqa: E402
import run_catch2  # noqa: E402

FAKE = """\
import json, sys, time
spec = json.load(open(sys.argv[1]))
out = [a for a in sys.argv if a.startswith("junit::out=")][0].split("=", 1)[1]
if spec.get("xml") is not None:
    open(out, "w").write(spec["xml"])
if "--input-file" in sys.argv:
    names = open(sys.argv[sys.argv.index("--input-file") + 1]).read().split()
    print("CASES=" + ",".join(names))
if spec.get("echo_args"):
    print("ARGS=" + json.dumps(sys.argv[2:]))
print(spec.get("banner", "BANNER"), flush=True)
if spec.get("sleep"):
    time.sleep(spec["sleep"])
sys.exit(spec.get("rc", 0))
"""


def junit(*records: str) -> str:
    return "<testsuites><testsuite>" + "".join(records) + "</testsuite></testsuites>"


def record(name: str, kind: str | None = None, text: str = "") -> str:
    if kind is None:
        return f'<testcase name="{name}"/>'
    return f'<testcase name="{name}"><{kind}>{text}</{kind}></testcase>'


class ParseArgsTest(unittest.TestCase):
    def test_strips_separator_and_reads_options(self):
        args = run_catch2.parse_args(
            ["--xml", "r.xml", "--timeout", "2.5", "--", "./bin", "[cpu]"]
        )
        self.assertEqual(args.xml, Path("r.xml"))
        self.assertEqual(args.timeout, 2.5)
        self.assertEqual(args.command, ["./bin", "[cpu]"])

    def test_timeout_is_optional(self):
        args = run_catch2.parse_args(["--xml", "r.xml", "--", "./bin"])
        self.assertIsNone(args.timeout)

    def test_requires_a_command(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_catch2.parse_args(["--xml", "r.xml", "--"])

    def test_requires_xml(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_catch2.parse_args(["--", "./bin"])


class ResultTest(unittest.TestCase):
    def parse(self, text: str) -> ET.Element:
        return ET.fromstring(text)

    def test_test_result(self):
        self.assertEqual(run_catch2.test_result(self.parse(record("a"))), "PASS")
        self.assertEqual(run_catch2.test_result(self.parse(record("a", "skipped"))), "SKIP")
        self.assertEqual(run_catch2.test_result(self.parse(record("a", "failure"))), "FAIL")
        self.assertEqual(run_catch2.test_result(self.parse(record("a", "error"))), "FAIL")

    def test_group_cases_folds_sections_into_their_case(self):
        root = self.parse(
            junit(record("A"), record("B/x"), record("B/y"), record("C/s/t"), '<testcase/>')
        )
        groups = run_catch2.group_cases(root)
        self.assertEqual(list(groups), ["A", "B", "C"])
        self.assertEqual(len(groups["B"]), 2)

    def test_group_result_precedence(self):
        def result(*kinds):
            root = self.parse(junit(*(record("A/" + str(i), k) for i, k in enumerate(kinds))))
            return run_catch2.group_result(run_catch2.group_cases(root)["A"])

        self.assertEqual(result(None, None), "PASS")
        self.assertEqual(result(None, "skipped"), "SKIP")
        self.assertEqual(result("skipped", "failure", None), "FAIL")
        self.assertEqual(result("error"), "FAIL")


class MainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.fake = self.dir / "fake_catch2.py"
        self.fake.write_text(FAKE)
        self.xml = self.dir / "out.xml"

    def run_main(self, spec: dict, *extra: str):
        spec_path = self.dir / "spec.json"
        spec_path.write_text(json.dumps(spec))
        argv = ["--xml", str(self.xml), *extra, "--", sys.executable, str(self.fake), str(spec_path)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = run_catch2.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_passing_run(self):
        rc, out, _ = self.run_main({"xml": junit(record("A"), record("B"))})
        self.assertEqual(rc, 0)
        self.assertIn("PASS: A", out)
        self.assertIn("2 passed, 0 failed, 0 skipped", out)
        self.assertNotIn("BANNER", out)

    def test_reporters_are_appended(self):
        rc, out, _ = self.run_main({"xml": junit(record("A")), "echo_args": True})
        self.assertEqual(rc, 0)
        self.assertNotIn("ARGS=", out)  # the transcript is only shown on a crash
        spec = {"xml": junit(record("A")), "echo_args": True, "rc": 139}
        _, out, _ = self.run_main(spec)
        args = json.loads(out.split("ARGS=")[1].splitlines()[0])
        self.assertIn("--durations", args)
        self.assertIn(f"junit::out={self.xml}", args)
        self.assertIn("console", args)

    def test_failing_section_fails_its_passing_parent(self):
        xml = junit(
            record("Case"),
            record("Case/ok"),
            record("Case/bad", "failure", "assertion went wrong"),
        )
        rc, out, _ = self.run_main({"xml": xml, "rc": 1})
        self.assertEqual(rc, 1)
        self.assertIn("FAIL: Case", out)
        self.assertIn("assertion went wrong", out)
        self.assertIn("0 passed, 1 failed, 0 skipped", out)
        self.assertNotIn("BANNER", out)

    def test_case_with_only_sections_is_counted(self):
        xml = junit(record("OnlySections/a", "failure", "boom"), record("Fine"))
        rc, out, _ = self.run_main({"xml": xml, "rc": 1})
        self.assertEqual(rc, 1)
        self.assertIn("FAIL: OnlySections", out)
        self.assertIn("1 passed, 1 failed, 0 skipped", out)

    def test_all_skipped_exit_4_is_success_without_transcript(self):
        rc, out, _ = self.run_main({"xml": junit(record("A", "skipped")), "rc": 4})
        self.assertEqual(rc, 0)
        self.assertIn("SKIP: A", out)
        self.assertNotIn("BANNER", out)

    def test_exit_4_with_a_failure_stays_failed(self):
        xml = junit(record("A", "skipped"), record("B", "failure", "x"))
        rc, out, _ = self.run_main({"xml": xml, "rc": 4})
        self.assertEqual(rc, 4)
        self.assertIn("1 failed", out)

    def test_crash_after_skips_keeps_the_transcript(self):
        rc, out, _ = self.run_main({"xml": junit(record("A", "skipped"), record("B")), "rc": 139})
        self.assertEqual(rc, 139)
        self.assertIn("Catch2 terminated without a JUnit failure", out)
        self.assertIn("BANNER", out)

    def test_crash_without_skips_keeps_the_transcript(self):
        rc, out, _ = self.run_main({"xml": junit(record("A")), "rc": 134})
        self.assertEqual(rc, 134)
        self.assertIn("BANNER", out)

    def test_unreadable_junit_prints_transcript(self):
        rc, out, err = self.run_main({"xml": "<testsuites><oops", "rc": 3})
        self.assertEqual(rc, 3)
        self.assertIn("Could not read JUnit results", err)
        self.assertIn("BANNER", out)

    def test_missing_junit_with_clean_exit_still_fails(self):
        rc, out, err = self.run_main({"xml": None, "rc": 0})
        self.assertEqual(rc, 1)
        self.assertIn("Could not read JUnit results", err)

    def _config(self, cases):
        path = self.dir / "suite.yaml"
        path.write_text(hrr_test_config.render({
            "version": 1,
            "suite": "integration",
            "binary": "hrr-integration-tests",
            "select": "[hrr]~[direct]",
            "targets": {"os": ["linux", "windows"],
                        "arch": ["gfx90a", "gfx1151"]},
            "cases": cases,
        }))
        return path

    def test_config_skips_a_case_and_records_why(self):
        path = self._config({
            "Keep": {},
            "Drop": {"skip": [{
                "when": {"os": "linux", "arch": "gfx1151"},
                "kind": "disabled",
                "reason": "npu",
                "issue": "https://github.com/ROCm/rocm-systems/issues/1",
            }]},
        })
        listed = [{"name": "Keep", "tags": ["[hrr]"]}, {"name": "Drop", "tags": ["[hrr]"]}]
        with mock.patch("hrr_test_config.list_cases", return_value=listed):
            rc, out, _ = self.run_main(
                {"xml": junit(record("Keep"))},
                "--config", str(path), "--os", "linux", "--arch", "gfx1151",
            )
        self.assertEqual(rc, 0)
        self.assertIn("PASS: Keep", out)
        self.assertIn("SKIP (config): Drop", out)
        self.assertIn("hrr-config: linux/gfx1151: npu "
                      "(https://github.com/ROCm/rocm-systems/issues/1)", out)
        self.assertIn("1 config-skipped", out)
        root = ET.parse(self.xml).getroot()
        skipped = root.find(".//testcase[@name='Drop']/skipped")
        self.assertIsNotNone(skipped)
        self.assertTrue(skipped.attrib["message"].startswith("hrr-config:"))

    def test_config_runs_nothing_when_every_case_is_skipped(self):
        path = self._config({"Drop": {"skip": [{
            "when": {"os": "linux"}, "kind": "unsupported", "reason": "all",
        }]}})
        with mock.patch("hrr_test_config.list_cases",
                        return_value=[{"name": "Drop", "tags": ["[hrr]"]}]) as listed:
            rc, out, _ = self.run_main(
                {"xml": junit(record("Drop")), "banner": "SHOULD NOT RUN"},
                "--config", str(path), "--os", "linux", "--arch", "none",
            )
        self.assertEqual(rc, 0)
        self.assertIn("SKIP (config): Drop", out)
        self.assertNotIn("SHOULD NOT RUN", out)
        listed.assert_called_once()
        self.assertEqual(listed.call_args.args[1], "[hrr]~[direct]")

    def test_ignore_config_runs_the_skipped_case(self):
        path = self._config({"Drop": {"skip": [{
            "when": {"os": "linux"}, "kind": "unsupported", "reason": "all",
        }]}})
        with mock.patch("hrr_test_config.list_cases",
                        return_value=[{"name": "Drop", "tags": ["[hrr]"]}]) :
            rc, out, _ = self.run_main(
                {"xml": junit(record("Drop")), "rc": 0},
                "--config", str(path), "--os", "linux", "--arch", "gfx1151",
                "--ignore-config",
            )
        self.assertEqual(rc, 0)
        self.assertIn("PASS: Drop", out)
        self.assertNotIn("SKIP (config)", out)
        self.assertNotIn("hrr-config:", out)

    def test_select_overrides_the_config_and_a_trailing_filter_is_dropped(self):
        path = self._config({"Keep": {}})
        spec_path = self.dir / "spec.json"
        # A non-zero status with no JUnit failure dumps the transcript, which is
        # the only place the wrapper's argv (and the case file) are visible.
        spec_path.write_text(json.dumps({
            "xml": junit(record("Keep")), "echo_args": True, "rc": 139,
        }))
        argv = [
            "--xml", str(self.xml), "--config", str(path), "--os", "linux",
            "--arch", "none", "--select", "[cpu]",
            "--", sys.executable, str(self.fake), str(spec_path), "[hrr]~[direct]",
        ]
        with mock.patch("hrr_test_config.list_cases",
                        return_value=[{"name": "Keep", "tags": ["[cpu]"]}]) as listed:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = run_catch2.main(argv)
        self.assertEqual(rc, 139)
        self.assertEqual(listed.call_args.args[1], "[cpu]")
        text = out.getvalue()
        self.assertIn("CASES=Keep", text)
        self.assertIn("--input-file", text)
        self.assertNotIn("[hrr]~[direct]", text)

    def test_auto_arch_with_no_device_fails_closed(self):
        path = self._config({"Keep": {}})
        with mock.patch("hrr_test_config.detect_archs", return_value=[]):
            rc, _, err = self.run_main(
                {"xml": junit(record("Keep"))},
                "--config", str(path), "--os", "linux", "--arch", "auto",
            )
        self.assertEqual(rc, 1)
        self.assertIn("could not detect a GPU architecture", err)

    def test_config_flags_without_a_config_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_catch2.parse_args(["--xml", "r.xml", "--arch", "none", "--", "./bin"])

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX only")
    def test_timeout_kills_the_suite_and_reports(self):
        rc, out, _ = self.run_main({"sleep": 60, "banner": "LAST CASE"}, "--timeout", "1")
        self.assertEqual(rc, 1)
        self.assertIn("timed out after 1 s", out)


if __name__ == "__main__":
    unittest.main()
