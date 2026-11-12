#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_test_config.py. No Catch2 binary and no GPU."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_test_config as cfg  # noqa: E402

ARCHS = ["gfx90a", "gfx1100", "gfx1101", "gfx1151"]


def suite(cases, skip=None, select="[hrr]~[direct]"):
    return {
        "version": 1,
        "suite": "integration",
        "binary": "hrr-integration-tests",
        "select": select,
        "targets": {"os": ["linux", "windows"], "arch": list(ARCHS)},
        "skip": skip or [],
        "cases": cases,
    }


def rule(when, kind="unsupported", reason="because", issue=None):
    item = {"when": when, "kind": kind, "reason": reason}
    if issue:
        item["issue"] = issue
    return item


def listing(*cases):
    tests = [{"name": name, "tags": tags} for name, tags in cases]
    return json.dumps({"version": 1, "listings": {"tests": tests}})


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class ResolveTest(unittest.TestCase):
    def names(self):
        return ["Runs", "Filtered"]

    def test_empty_entry_runs_everywhere(self):
        run, skipped = cfg.resolve(suite({"Runs": {}}), ["Runs"], "linux", ["gfx1151"])
        self.assertEqual(run, ["Runs"])
        self.assertEqual(skipped, [])

    def test_os_and_arch_are_anded(self):
        cases = {"Filtered": {"skip": [
            rule({"os": "linux", "arch": "gfx1151"}, "disabled", "npu",
                 "https://github.com/ROCm/rocm-systems/issues/1"),
        ]}}
        run, skipped = cfg.resolve(suite(cases), ["Filtered"], "windows", ["gfx1151"])
        self.assertEqual(run, ["Filtered"])
        run, skipped = cfg.resolve(suite(cases), ["Filtered"], "linux", ["gfx90a"])
        self.assertEqual(run, ["Filtered"])
        run, skipped = cfg.resolve(suite(cases), ["Filtered"], "linux", ["gfx1151"])
        self.assertEqual(run, [])
        self.assertEqual(skipped[0]["archs"], ["gfx1151"])
        message = cfg.skip_message("linux", skipped[0]["archs"], skipped[0]["rule"])
        self.assertEqual(
            message,
            "hrr-config: linux/gfx1151: npu (https://github.com/ROCm/rocm-systems/issues/1)",
        )

    def test_a_list_is_or_and_a_glob_matches_a_family(self):
        cases = {"Filtered": {"skip": [rule({"arch": ["gfx90a", "gfx11*"]})]}}
        for arch in ("gfx90a", "gfx1100", "gfx1101", "gfx1151"):
            run, _ = cfg.resolve(suite(cases), ["Filtered"], "linux", [arch])
            self.assertEqual(run, [], arch)
        run, _ = cfg.resolve(suite(cases), ["Filtered"], "linux", ["gfx1200"])
        # gfx1200 is not in this suite's vocabulary, but matching is against the
        # machine, not the vocabulary. The glob does not match it.
        self.assertEqual(run, ["Filtered"])

    def test_a_missing_key_matches_anything(self):
        cases = {"Filtered": {"skip": [rule({"os": "windows"})]}}
        run, skipped = cfg.resolve(suite(cases), ["Filtered"], "windows", [])
        self.assertEqual(run, [])
        self.assertEqual(cfg.skip_message("windows", skipped[0]["archs"], skipped[0]["rule"]),
                         "hrr-config: windows: because")

    def test_suite_rule_applies_to_every_case_and_wins(self):
        cases = {"Runs": {}, "Filtered": {"skip": [rule({"os": "windows"})]}}
        config = suite(cases, skip=[rule({"arch": "gfx90a"}, reason="whole arch")])
        run, skipped = cfg.resolve(config, ["Runs", "Filtered"], "linux", ["gfx90a"])
        self.assertEqual(run, [])
        self.assertEqual({item["rule"]["reason"] for item in skipped}, {"whole arch"})

    def test_any_visible_gpu_matches(self):
        cases = {"Filtered": {"skip": [rule({"arch": "gfx1151"})]}}
        run, skipped = cfg.resolve(suite(cases), ["Filtered"], "linux", ["gfx90a", "gfx1151"])
        self.assertEqual(run, [])
        self.assertEqual(skipped[0]["archs"], ["gfx1151"])

    def test_no_device_does_not_match_an_arch_rule(self):
        cases = {"Filtered": {"skip": [rule({"arch": "gfx1151"})]}}
        run, _ = cfg.resolve(suite(cases), ["Filtered"], "linux", [])
        self.assertEqual(run, ["Filtered"])

    def test_ignore_runs_everything(self):
        cases = {"Filtered": {"skip": [rule({"os": "linux"})]}}
        config = suite(cases, skip=[rule({"arch": "gfx90a"})])
        run, skipped = cfg.resolve(config, ["Filtered"], "linux", ["gfx90a"], ignore=True)
        self.assertEqual(run, ["Filtered"])
        self.assertEqual(skipped, [])


class LintTest(unittest.TestCase):
    def test_a_valid_config_is_quiet(self):
        cases = {"Runs": {}, "Filtered": {"skip": [
            rule({"os": "linux", "arch": "gfx11*"}, "disabled", "npu",
                 "https://github.com/ROCm/rocm-systems/issues/1"),
        ]}}
        self.assertEqual(cfg.lint(suite(cases)), [])

    def test_disabled_needs_an_issue_and_every_rule_needs_a_reason(self):
        cases = {"Filtered": {"skip": [{"when": {"os": "linux"}, "kind": "disabled"}]}}
        errors = cfg.lint(suite(cases))
        self.assertTrue(any("reason" in error for error in errors))
        self.assertTrue(any("issue" in error for error in errors))

    def test_unknown_arch_and_a_glob_that_matches_nothing(self):
        cases = {"Filtered": {"skip": [
            rule({"arch": "gfx999"}),
            rule({"arch": "gfx99*"}),
        ]}}
        errors = cfg.lint(suite(cases))
        self.assertTrue(any("gfx999" in error for error in errors))
        self.assertTrue(any("gfx99*" in error for error in errors))

    def test_unknown_keys_and_a_bad_case_name(self):
        config = suite({"Has Space": {}})
        config["extra"] = True
        errors = cfg.lint(config)
        self.assertTrue(any("unknown keys" in error for error in errors))
        self.assertTrue(any("case name" in error for error in errors))

    def test_a_direct_workload_cannot_carry_rules(self):
        cases = {"Unit_HRR_Foo_Direct": {"skip": [rule({"os": "linux"})]}}
        listed = [{"name": "Unit_HRR_Foo_Direct", "tags": ["[.]","[hrr-direct]"]}]
        errors = cfg.lint(suite(cases), cases=listed)
        self.assertTrue(any("workload" in error for error in errors))
        self.assertEqual(cfg.lint(suite({"Unit_HRR_Foo_Direct": {}}), cases=listed), [])

    def test_workflow_archs_must_match(self):
        errors = cfg.lint(suite({"Runs": {}}), workflow_archs=["gfx90a"])
        self.assertTrue(any("HRR_ARCHS" in error for error in errors))
        self.assertEqual(cfg.lint(suite({"Runs": {}}), workflow_archs=list(ARCHS)), [])


class ListingTest(unittest.TestCase):
    def test_json_listing_keeps_tags_and_passes_the_selector(self):
        payload = listing(("B", ["[hrr]"]), ("A", ["[.]","[hrr-direct]"]))
        with mock.patch.object(cfg.subprocess, "run", return_value=completed(payload)) as run:
            cases = cfg.list_cases("./bin", "[hrr]~[direct]")
        self.assertEqual(run.call_args.args[0],
                         ["./bin", "--list-tests", "[hrr]~[direct]", "--reporter", "json"])
        self.assertEqual(cases[0]["name"], "B")
        self.assertEqual(cases[1]["tags"], ["[.]", "[hrr-direct]"])

    def test_a_failing_listing_raises(self):
        with mock.patch.object(cfg.subprocess, "run", return_value=completed("", 2, "bad")):
            with self.assertRaises(cfg.ConfigError):
                cfg.list_cases("./bin")

    def test_an_empty_star_listing_is_a_check_failure(self):
        with mock.patch.object(cfg.subprocess, "run", return_value=completed(listing())):
            with self.assertRaises(cfg.ConfigError):
                cfg.listed_names("./bin")


class CheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "suite.yaml"

    def write(self, config):
        self.path.write_text(cfg.render(config))

    def test_check_reports_both_directions(self):
        self.write(suite({"Kept": {}, "Gone": {}}))
        payload = listing(("Kept", []), ("New", []))
        with mock.patch.object(cfg.subprocess, "run", return_value=completed(payload)):
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = cfg.main(["check", "--binary", "./bin", "--config", str(self.path)])
        self.assertEqual(rc, 1)
        self.assertIn("MISSING", err.getvalue())
        self.assertIn("Gone", err.getvalue())
        self.assertIn("UNEXPECTED", err.getvalue())
        self.assertIn("New", err.getvalue())

    def test_update_keeps_rules_and_adds_empty_entries(self):
        kept = rule({"os": "linux", "arch": "gfx1151"}, "disabled", "npu",
                    "https://github.com/ROCm/rocm-systems/issues/1")
        self.write(suite({"Kept": {"skip": [kept]}, "Gone": {"skip": [rule({"os": "windows"})]}}))
        payload = listing(("Kept", ["[hrr]"]), ("New", ["[hrr]"]))
        with mock.patch.object(cfg.subprocess, "run", return_value=completed(payload)):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cfg.main(["check", "--binary", "./bin", "--config", str(self.path), "--update"])
        self.assertEqual(rc, 0)
        loaded = cfg.load(self.path)
        self.assertEqual(set(loaded["cases"]), {"Kept", "New"})
        self.assertEqual(loaded["cases"]["Kept"]["skip"][0]["reason"], "npu")
        self.assertFalse(loaded["cases"]["New"])
        again = listing(("Kept", ["[hrr]"]), ("New", ["[hrr]"]))
        # None and {} are the same empty entry; a second check is clean.
        self.path.write_text(cfg.render(loaded))
        with mock.patch.object(cfg.subprocess, "run", return_value=completed(again)):
            with contextlib.redirect_stdout(io.StringIO()):
                rc = cfg.main(["check", "--binary", "./bin", "--config", str(self.path)])
        self.assertEqual(rc, 0)

    def test_round_trip_preserves_a_rule(self):
        original = suite(
            {"Filtered": {"skip": [rule({"os": "linux", "arch": ["gfx1151", "gfx11*"]},
                                        "disabled", "npu",
                                        "https://github.com/ROCm/rocm-systems/issues/1")]}},
            skip=[rule({"arch": "gfx90a"}, reason="whole arch")],
        )
        self.path.write_text(cfg.render(original))
        loaded = cfg.load(self.path)
        self.assertEqual(cfg.lint(loaded), [])
        run, skipped = cfg.resolve(loaded, ["Filtered"], "linux", ["gfx90a"])
        self.assertEqual(run, [])
        self.assertEqual(skipped[0]["rule"]["reason"], "whole arch")


class DetectTest(unittest.TestCase):
    def test_first_tool_with_an_arch_wins_and_suffixes_are_dropped(self):
        def run(argv):
            if argv == ["offload-arch"]:
                return "gfx1100:xnack-\ngfx1100\n"
            raise AssertionError(argv)
        self.assertEqual(cfg.detect_archs(run), ["gfx1100"])

    def test_later_tools_are_tried_when_earlier_ones_fail(self):
        def run(argv):
            if argv == ["rocminfo"]:
                return "Name: gfx1151\n"
            return None
        self.assertEqual(cfg.detect_archs(run), ["gfx1151"])

    def test_nothing_visible_is_an_empty_list(self):
        self.assertEqual(cfg.detect_archs(lambda argv: None), [])


class WorkflowTest(unittest.TestCase):
    def test_reads_hrr_archs(self):
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        tmp.write('env:\n  HRR_ARCHS: "gfx90a;gfx942"\n')
        tmp.close()
        self.assertEqual(cfg.workflow_archs(tmp.name), ["gfx90a", "gfx942"])


class LintCommandTest(unittest.TestCase):
    def test_lint_command_fails_on_a_bad_file(self):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        tmp.write("version: 2\n")
        tmp.close()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = cfg.main(["lint", tmp.name])
        self.assertEqual(rc, 1)
        self.assertIn("version", err.getvalue())


if __name__ == "__main__":
    unittest.main()
