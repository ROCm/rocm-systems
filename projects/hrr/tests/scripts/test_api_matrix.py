#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the API matrix tools in tools/api-matrix.

Everything runs against small in-memory manifests and overlays written to
temporary directories, so no GPU, no generator checkout and no file in the
source tree is read or modified. PyYAML is required, as it is for the tools.
"""

import contextlib
import copy
import io
import json
import os
import re
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" /
                       "api-matrix"))

import check_matrix as cm  # noqa: E402
import derive_manifest as dm  # noqa: E402
import matrix_common as mc  # noqa: E402


def api_entry(name, observable="REAL", payload_loss=()):
    return {"api": name, "table": "runtime", "capture_class": "GENERATED",
            "replay_class": "GENERATED" if observable == "REAL" else observable,
            "observable_class": observable, "payload_loss": list(payload_loss)}


MANIFEST = {"apis": [api_entry("hipA"), api_entry("hipB", "NOOP"),
                     api_entry("hipC")]}

# A minimal valid overlay: one explicit tier list, one pattern, one override.
OVERLAY = {
    "version": 1,
    "defaults": {"tier": "T0", "expect": "derived", "workloads": ["W0"]},
    "tiers": {
        "T0": {"title": "first", "rationale": "why", "min_covered": 2,
               "gpus": 1, "workloads": ["W0"], "apis": ["hipA", "hipB"]},
        "T1": {"min_covered": 1, "gpus": 1, "workloads": ["W1", "W2"]},
    },
    "patterns": [{"match": "^hipC", "tier": "T1", "reason": "why"}],
    "unreachable": [{"group": "g", "reason": "measured", "apis": []}],
    "overrides": {"hipB": {"payload_loss": True, "note": "n"}},
}

TIER_WORKLOADS = {"T0": ["W0"], "T1": ["W1", "W2"]}


def observation(tier, **fields):
    data = {"tier": tier, "replay_exit": 0, "workloads": TIER_WORKLOADS.get(tier, []),
            "captured": {}, "observed": {}}
    data.update(fields)
    return data


def run_main(argv):
    """Run check_matrix.main(), returning (exit code, stderr)."""
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(err):
        rc = cm.main(argv)
    return rc, err.getvalue()


class Scratch(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def write_json(self, name, data):
        path = self.dir / name
        path.write_text(json.dumps(data))
        return path

    def write_overlay(self, overlay):
        path = self.dir / "api_matrix.yaml"
        path.write_text(yaml.safe_dump(overlay))
        return path

    def results(self, *files):
        results = self.dir / "results"
        results.mkdir(exist_ok=True)
        for name, data in files:
            (results / name).write_text(json.dumps(data))
        return results


class OverlaySchema(Scratch):
    def test_valid_overlay_loads_and_resolves(self):
        mc.validate_overlay(OVERLAY)
        path = self.write_overlay(OVERLAY)
        overlay = mc.load_overlay(path)
        rows = {r.api: r for r in mc.resolve(MANIFEST, overlay)}
        self.assertEqual(rows["hipC"].tier, "T1")
        self.assertTrue(rows["hipB"].payload_loss)
        self.assertEqual(mc.validate(list(rows.values()), overlay), [])

    def rejected(self, mutate, expect):
        overlay = copy.deepcopy(OVERLAY)
        mutate(overlay)
        with self.assertRaises(mc.MatrixError) as ctx:
            mc.validate_overlay(overlay)
        self.assertIn(expect, str(ctx.exception))

    def test_unknown_and_removed_keys_are_rejected(self):
        cases = {
            "top level typo": (lambda o: o.__setitem__("tires", {}),
                               "unknown key 'tires'"),
            "baseline": (lambda o: o.__setitem__("baseline", "x.json"),
                         "unknown key 'baseline'"),
            "defaults typo": (lambda o: o["defaults"].__setitem__("worklaods", []),
                              "defaults: unknown key 'worklaods'"),
            "tier typo": (lambda o: o["tiers"]["T0"].__setitem__("min_cover", 1),
                          "tiers.T0: unknown key 'min_cover'"),
            "pattern typo": (lambda o: o["patterns"][0].__setitem__("mach", "x"),
                             "patterns[0]: unknown key 'mach'"),
            "unreachable typo": (
                lambda o: o["unreachable"][0].__setitem__("api", []),
                "unreachable[0]: unknown key 'api'"),
            "override typo": (
                lambda o: o["overrides"]["hipB"].__setitem__("payload_los", True),
                "overrides.hipB: unknown key 'payload_los'"),
            "removed key skip_unless_multi_gpu": (
                lambda o: o["overrides"]["hipB"].__setitem__(
                    "skip_unless_multi_gpu", True),
                "unknown key 'skip_unless_multi_gpu'"),
            "removed key skip_by_default": (
                lambda o: o["tiers"]["T0"].__setitem__("skip_by_default", True),
                "tiers.T0: unknown key 'skip_by_default'"),
        }
        for label, (mutate, expect) in cases.items():
            with self.subTest(label):
                self.rejected(mutate, expect)

    def test_wrong_types_are_rejected(self):
        cases = {
            "version": (lambda o: o.__setitem__("version", "1"), "version"),
            "tiers not a mapping": (lambda o: o.__setitem__("tiers", []),
                                    "tiers"),
            "patterns not a list": (lambda o: o.__setitem__("patterns", {}),
                                    "patterns: expected a list"),
            "unreachable not a list": (
                lambda o: o.__setitem__("unreachable", {}),
                "unreachable: expected a list"),
            "overrides not a mapping": (
                lambda o: o.__setitem__("overrides", []),
                "overrides: expected a mapping"),
            "min_covered string": (
                lambda o: o["tiers"]["T0"].__setitem__("min_covered", "2"),
                "tiers.T0.min_covered"),
            "min_covered bool": (
                lambda o: o["tiers"]["T0"].__setitem__("min_covered", True),
                "tiers.T0.min_covered"),
            "gpus float": (lambda o: o["tiers"]["T0"].__setitem__("gpus", 1.0),
                           "tiers.T0.gpus"),
            "apis not a list": (
                lambda o: o["tiers"]["T0"].__setitem__("apis", "hipA"),
                "tiers.T0.apis"),
            "apis entry not a string": (
                lambda o: o["tiers"]["T0"].__setitem__("apis", [{"a": 1}]),
                "tiers.T0.apis"),
            "workloads string": (
                lambda o: o["tiers"]["T0"].__setitem__("workloads", "W0"),
                "tiers.T0.workloads"),
            "payload_loss string": (
                lambda o: o["overrides"]["hipB"].__setitem__(
                    "payload_loss", "yes"),
                "overrides.hipB.payload_loss"),
            "captured int": (
                lambda o: o["overrides"]["hipB"].__setitem__("captured", 0),
                "overrides.hipB.captured"),
            "handler_error_in string": (
                lambda o: o["overrides"]["hipB"].__setitem__(
                    "handler_error_in", "T0"),
                "overrides.hipB.handler_error_in"),
            "handler_error_in integer": (
                lambda o: o["overrides"]["hipB"].__setitem__(
                    "handler_error_in", 1),
                "overrides.hipB.handler_error_in"),
            "override not a mapping": (
                lambda o: o["overrides"].__setitem__("hipB", ["x"]),
                "overrides.hipB: expected a mapping"),
            "expect unknown": (
                lambda o: o["overrides"]["hipB"].__setitem__("expect", "real"),
                "overrides.hipB.expect"),
            "unreachable apis not a list": (
                lambda o: o["unreachable"][0].__setitem__("apis", "hipA"),
                "unreachable[0].apis"),
            "unreachable missing reason": (
                lambda o: o["unreachable"][0].pop("reason"),
                "unreachable[0]: missing required key 'reason'"),
        }
        for label, (mutate, expect) in cases.items():
            with self.subTest(label):
                self.rejected(mutate, expect)

    def test_tier_definitions_are_checked(self):
        cases = {
            "tier outside TIER_ORDER": (
                lambda o: o["tiers"].__setitem__("T9", o["tiers"]["T1"]),
                "'T9' is not one of"),
            "defaults tier outside TIER_ORDER": (
                lambda o: o["defaults"].__setitem__("tier", "T9"),
                "defaults.tier"),
            "pattern tier outside TIER_ORDER": (
                lambda o: o["patterns"][0].__setitem__("tier", "T9"),
                "patterns[0].tier"),
            "override tier outside TIER_ORDER": (
                lambda o: o["overrides"]["hipB"].__setitem__("tier", "T9"),
                "overrides.hipB.tier"),
            "pattern tier not defined": (
                lambda o: o["patterns"][0].__setitem__("tier", "T3"),
                "patterns[0].tier: T3 has no entry under tiers"),
            "missing workloads": (
                lambda o: o["tiers"]["T1"].pop("workloads"),
                "tiers.T1: missing required key 'workloads'"),
            "empty workloads": (
                lambda o: o["tiers"]["T1"].__setitem__("workloads", []),
                "tiers.T1.workloads"),
            "missing min_covered": (
                lambda o: o["tiers"]["T1"].pop("min_covered"),
                "tiers.T1: missing required key 'min_covered'"),
            "min_covered zero": (
                lambda o: o["tiers"]["T1"].__setitem__("min_covered", 0),
                "tiers.T1.min_covered"),
            "min_covered negative": (
                lambda o: o["tiers"]["T1"].__setitem__("min_covered", -1),
                "tiers.T1.min_covered"),
            "no tiers": (lambda o: o.__setitem__("tiers", {}), "tiers"),
        }
        for label, (mutate, expect) in cases.items():
            with self.subTest(label):
                self.rejected(mutate, expect)

    def test_malformed_patterns_are_rejected(self):
        cases = {
            "bad regex": (lambda o: o["patterns"][0].__setitem__("match", "("),
                          "patterns[0].match"),
            "empty regex": (lambda o: o["patterns"][0].__setitem__("match", ""),
                            "patterns[0].match"),
            "regex not a string": (
                lambda o: o["patterns"][0].__setitem__("match", 5),
                "patterns[0].match"),
            "missing match": (lambda o: o["patterns"][0].pop("match"),
                              "patterns[0]: missing required key 'match'"),
            "missing tier": (lambda o: o["patterns"][0].pop("tier"),
                             "patterns[0]: missing required key 'tier'"),
            "entry not a mapping": (
                lambda o: o["patterns"].__setitem__(0, "^hipC"),
                "patterns[0]: expected a mapping"),
        }
        for label, (mutate, expect) in cases.items():
            with self.subTest(label):
                self.rejected(mutate, expect)

    def test_override_tier_conflicting_with_tier_list_is_rejected(self):
        self.rejected(
            lambda o: o["overrides"].__setitem__("hipA", {"tier": "T1"}),
            "overrides.hipA.tier: T1 conflicts with hipA being listed under "
            "tiers.T0.apis")

    def test_override_tier_matching_tier_list_is_accepted(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["overrides"]["hipA"] = {"tier": "T0"}
        mc.validate_overlay(overlay)

    def test_every_problem_is_reported_in_one_error(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["tiers"]["T0"]["min_covered"] = 0
        overlay["overrides"]["hipB"]["payload_los"] = True
        with self.assertRaises(mc.MatrixError) as ctx:
            mc.validate_overlay(overlay)
        self.assertIn("min_covered", str(ctx.exception))
        self.assertIn("payload_los", str(ctx.exception))

    def test_unreadable_yaml_is_a_matrix_error(self):
        for label, text in {"invalid": "tiers: [", "empty": "",
                            "duplicate keys": "tiers: {}\ntiers: {}\n"}.items():
            with self.subTest(label):
                path = self.dir / f"{label.replace(' ', '_')}.yaml"
                path.write_text(text)
                with self.assertRaises(mc.MatrixError):
                    mc.load_overlay(path)

    def test_non_utf8_inputs_are_matrix_errors(self):
        overlay = self.dir / "latin1.yaml"
        overlay.write_bytes(b"tiers: {}\n# \xe9\n")
        with self.assertRaises(mc.MatrixError):
            mc.load_overlay(overlay)
        manifest = self.dir / "latin1.json"
        manifest.write_bytes(b'{"apis": [], "x": "\xe9"}')
        with self.assertRaises(mc.MatrixError):
            mc.load_manifest(manifest)

    def test_utf8_overlay_text_is_read_as_utf8(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["tiers"]["T0"]["title"] = "UC1 \u2014 dense"
        path = self.dir / "utf8.yaml"
        path.write_bytes(yaml.safe_dump(overlay, allow_unicode=True)
                         .encode("utf-8"))
        self.assertEqual(mc.load_overlay(path)["tiers"]["T0"]["title"],
                         "UC1 \u2014 dense")

    def test_unreadable_manifest_is_a_matrix_error(self):
        bad = self.dir / "bad.json"
        bad.write_text("{")
        with self.assertRaises(mc.MatrixError):
            mc.load_manifest(bad)
        with self.assertRaises(mc.MatrixError):
            mc.load_manifest(self.write_json("noapis.json", {"counts": {}}))


class HeaderGeneration(Scratch):
    def emit(self, name):
        overlay = copy.deepcopy(OVERLAY)
        rows = mc.resolve(MANIFEST, overlay)
        path = self.dir / name
        cm.emit_cxx(rows, overlay, path)
        return path.read_text()

    def test_header_is_deterministic_and_undated(self):
        first = self.emit("a.h")
        with mock.patch.object(cm, "datetime") as fake:
            fake.now.side_effect = AssertionError("header must not read the clock")
            second = self.emit("b.h")
        self.assertEqual(first.replace("a.h", "b.h"), second)
        self.assertIn("Generated from 3 HIP APIs.", first)
        self.assertIsNone(re.search(r"\d{4}-\d{2}-\d{2}", first))


class AtomicWrites(Scratch):
    """Both tools carry the same helper, so both are held to the same rules."""

    def writers(self):
        return {"check_matrix": cm.write_atomic,
                "derive_manifest": dm.write_atomic}

    def test_creates_parents_and_honors_umask(self):
        old = os.umask(0o027)
        self.addCleanup(os.umask, old)
        for name, write in self.writers().items():
            with self.subTest(name):
                target = self.dir / name / "a" / "b" / "out.txt"
                write(target, "caf\u00e9 \u2014 ok\n")
                self.assertEqual(target.read_text(encoding="utf-8"),
                                 "caf\u00e9 \u2014 ok\n")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
                self.assertEqual([p.name for p in target.parent.iterdir()],
                                 ["out.txt"])

    def test_preserves_the_mode_of_an_existing_target(self):
        for name, write in self.writers().items():
            with self.subTest(name):
                target = self.dir / f"{name}.txt"
                target.write_text("old")
                target.chmod(0o604)
                write(target, "new")
                self.assertEqual(target.read_text(), "new")
                self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o604)

    def test_failure_keeps_the_old_file_and_leaves_no_temp(self):
        for name, write in self.writers().items():
            with self.subTest(name):
                target = self.dir / f"{name}-keep.txt"
                target.write_text("old")
                module = cm if name == "check_matrix" else dm
                with mock.patch.object(module.os, "replace",
                                       side_effect=OSError("boom")):
                    with self.assertRaises(OSError):
                        write(target, "new")
                self.assertEqual(target.read_text(), "old")
                self.assertFalse([p for p in self.dir.iterdir()
                                  if p.name.startswith(target.name + ".")])


class Observations(Scratch):
    APIS = {"hipA", "hipB", "hipC"}

    def load(self, *files):
        return cm.load_observations(self.results(*files), self.APIS)

    def assertRejected(self, expect, *files):
        with self.assertRaises(mc.MatrixError) as ctx:
            self.load(*files)
        self.assertIn(expect, str(ctx.exception))

    def test_valid_files_merge_across_tiers(self):
        merged = self.load(
            ("T0.json", observation(
                "T0", captured={"hipA": 2, "hipB": 1},
                observed={"hipA": "REAL", "hipB": "NOOP"})),
            ("T1.json", observation(
                "T1", captured={"hipA": 1, "hipC": 1},
                observed={"hipA": "REAL", "hipC": "HANDLER_ERROR"})))
        self.assertEqual(merged["captured"]["hipA"], 3)
        self.assertEqual(merged["observed"]["hipA"], "REAL")
        self.assertEqual(merged["by_tier"]["hipA"], {"T0": "REAL", "T1": "REAL"})
        self.assertEqual(merged["tiers"]["T1"]["workloads"], ["W1", "W2"])

    def test_disagreeing_tiers_merge_to_conflict(self):
        merged = self.load(
            ("T0.json", observation("T0", observed={"hipA": "REAL"})),
            ("T1.json", observation("T1", observed={"hipA": "NOOP"})))
        self.assertEqual(merged["observed"]["hipA"], cm.CLASS_CONFLICT)

    def test_mislabeled_tier(self):
        self.assertRejected("declares tier T0",
                            ("T1.json", observation("T0")))

    def test_duplicate_tier_claim(self):
        self.assertRejected("declares tier T0",
                            ("T0.json", observation("T0")),
                            ("T1.json", observation("T0")))

    def test_unknown_or_missing_tier(self):
        self.assertRejected("unknown tier 'T9'", ("T1.json", observation("T9")))
        no_tier = observation("T1")
        del no_tier["tier"]
        self.assertRejected("unknown tier None", ("T1.json", no_tier))

    def test_nonzero_replay_exit_is_accepted_and_stored(self):
        # --continue-on-error exits nonzero whenever a declared HANDLER_ERROR
        # (or CRASH) API is replayed; the per-API observations stay valid.
        for code in (1, 139):
            with self.subTest(code):
                merged = self.load(
                    ("T0.json", observation(
                        "T0", replay_exit=code, captured={"hipA": 1},
                        observed={"hipA": "HANDLER_ERROR"})))
                self.assertEqual(merged["tiers"]["T0"]["replay_exit"], code)
                self.assertEqual(merged["observed"]["hipA"], "HANDLER_ERROR")

    def test_missing_or_non_integer_replay_exit(self):
        missing = observation("T0")
        del missing["replay_exit"]
        self.assertRejected("replay_exit", ("T0.json", missing))
        for bad in (None, "0", 0.0, False, [0]):
            with self.subTest(repr(bad)):
                self.assertRejected(
                    "replay_exit", ("T0.json", observation("T0", replay_exit=bad)))

    def test_unknown_api(self):
        self.assertRejected(
            "hipNope", ("T0.json", observation(
                "T0", captured={"hipNope": 1}, observed={"hipNope": "REAL"})))
        self.assertRejected(
            "hipNope", ("T0.json", observation(
                "T0", observed={"hipNope": "REAL"})))

    def test_malformed_maps(self):
        cases = {
            "captured list": {"captured": ["hipA"]},
            "captured string count": {"captured": {"hipA": "1"}},
            "captured negative": {"captured": {"hipA": -1}},
            "captured bool": {"captured": {"hipA": True}},
            "captured missing": {"captured": None},
            "observed list": {"observed": ["hipA"]},
            "observed bad class": {"observed": {"hipA": "real"}},
            "observed conflict class": {"observed": {"hipA": cm.CLASS_CONFLICT}},
            "observed missing": {"observed": None},
            "workloads string": {"workloads": "W0"},
        }
        for label, fields in cases.items():
            with self.subTest(label):
                self.assertRejected(
                    "T0.json", ("T0.json", observation("T0", **fields)))

    def test_unreadable_files(self):
        results = self.results()
        (results / "T0.json").write_text("{")
        with self.assertRaises(mc.MatrixError):
            cm.load_observations(results, self.APIS)
        (results / "T0.json").write_text("[]")
        with self.assertRaises(mc.MatrixError):
            cm.load_observations(results, self.APIS)
        with self.assertRaises(mc.MatrixError):
            cm.load_observations(self.dir / "missing", self.APIS)
        empty = self.dir / "empty"
        empty.mkdir()
        with self.assertRaises(mc.MatrixError):
            cm.load_observations(empty, self.APIS)


class Verdicts(unittest.TestCase):
    def rows(self, overlay=OVERLAY):
        return {r.api: r for r in mc.resolve(MANIFEST, overlay)}

    def observations(self, **by_api):
        return {
            "tiers": {"T0": {"workloads": ["W0"]}},
            "captured": {api: 1 for api in by_api},
            "observed": dict(by_api),
            "by_tier": {api: {"T0": k} for api, k in by_api.items()},
        }

    def test_payload_loss_on_noop_is_not_a_real_handler_xfail(self):
        row = self.rows()["hipB"]
        self.assertEqual((row.expect_class, row.payload_loss), ("NOOP", True))
        result = cm.verdict_for(row, self.observations(hipB="NOOP"))
        self.assertEqual(result["verdict"], cm.VERDICT_PASS)
        self.assertNotIn("real handler", result["detail"])
        self.assertNotIn("cause", result)

    def test_payload_loss_on_real_is_xfail_with_cause(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["overrides"]["hipA"] = {"payload_loss": True}
        row = self.rows(overlay)["hipA"]
        result = cm.verdict_for(row, self.observations(hipA="REAL"))
        self.assertEqual(result["verdict"], cm.VERDICT_XFAIL)
        self.assertEqual(result["cause"], "payload_loss")
        self.assertIn("real handler runs", result["detail"])

    def handler_error_row(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["overrides"]["hipA"] = {"handler_error_in": ["T0", "T1"]}
        return self.rows(overlay)["hipA"]

    def test_declared_handler_error_reports_the_observed_class(self):
        result = cm.verdict_for(self.handler_error_row(),
                                self.observations(hipA="HANDLER_ERROR"))
        self.assertEqual(result["verdict"], cm.VERDICT_XFAIL)
        self.assertEqual(result["cause"], "handler_error")
        self.assertEqual(result["expect"], "REAL")
        self.assertEqual(result["observed"], "HANDLER_ERROR")
        self.assertEqual(result["observed_by_tier"], {"T0": "HANDLER_ERROR"})

    def test_declared_handler_error_in_one_tier_keeps_the_merged_class(self):
        observations = {
            "tiers": {"T0": {"workloads": ["W0"]}},
            "captured": {"hipA": 2},
            "observed": {"hipA": cm.CLASS_CONFLICT},
            "by_tier": {"hipA": {"T0": "REAL", "T1": "HANDLER_ERROR"}},
        }
        result = cm.verdict_for(self.handler_error_row(), observations)
        self.assertEqual(result["verdict"], cm.VERDICT_XFAIL)
        self.assertEqual(result["observed"], cm.CLASS_CONFLICT)
        self.assertEqual(result["observed_by_tier"],
                         {"T0": "REAL", "T1": "HANDLER_ERROR"})

    def test_undeclared_disagreement_fails_with_per_tier_detail(self):
        observations = {
            "tiers": {"T0": {"workloads": ["W0"]}},
            "captured": {"hipA": 2},
            "observed": {"hipA": cm.CLASS_CONFLICT},
            "by_tier": {"hipA": {"T0": "REAL", "T1": "NOOP"}},
        }
        result = cm.verdict_for(self.rows()["hipA"], observations)
        self.assertEqual(result["verdict"], cm.VERDICT_FAIL)
        self.assertEqual(result["observed_by_tier"],
                         {"T0": "REAL", "T1": "NOOP"})

    def test_unexercised_api_and_low_floor_are_coverage_problems(self):
        overlay = copy.deepcopy(OVERLAY)
        rows = mc.resolve(MANIFEST, overlay)
        observations = {
            "tiers": {"T0": {"workloads": ["W0"]}},
            "captured": {"hipA": 1}, "observed": {"hipA": "REAL"},
            "by_tier": {"hipA": {"T0": "REAL"}},
        }
        report = cm.build_report(rows, overlay, observations)
        self.assertFalse(report["ok"])
        self.assertEqual(report["failures"], [])
        self.assertIn("T0: hipB was not exercised and has no unreachable: "
                      "declaration (no workload in this tier called it)",
                      report["coverage_problems"])
        self.assertIn("T0: its own workloads exercised 1 APIs, floor is 2",
                      report["coverage_problems"])

    def test_xfail_report_does_not_promise_an_automatic_xpass(self):
        overlay = copy.deepcopy(OVERLAY)
        overlay["overrides"]["hipA"] = {"payload_loss": True}
        rows = mc.resolve(MANIFEST, overlay)
        observations = {
            "tiers": {"T0": {"workloads": ["W0"]}},
            "captured": {"hipA": 1, "hipB": 1},
            "observed": {"hipA": "REAL", "hipB": "NOOP"},
            "by_tier": {"hipA": {"T0": "REAL"}, "hipB": {"T0": "NOOP"}},
        }
        report = cm.build_report(rows, overlay, observations)
        text = cm.render_markdown(report, overlay)
        self.assertIn("## Known failures (XFAIL)", text)
        self.assertNotIn("flip to XPASS", text)
        self.assertNotIn("P2", text)
        self.assertIn("payload_loss", text)

    def test_tier_missing_a_configured_workload_fails_the_report(self):
        overlay = copy.deepcopy(OVERLAY)
        rows = mc.resolve(MANIFEST, overlay)

        def report_for(workloads):
            observations = {
                "tiers": {"T0": {"workloads": ["W0"]},
                          "T1": {"workloads": workloads}},
                "captured": {}, "observed": {}, "by_tier": {},
            }
            return cm.build_report(rows, overlay, observations)

        short = report_for(["W1"])
        self.assertFalse(short["ok"])
        self.assertIn("T1: configured workload W2 is missing from its "
                      "observation", short["coverage_problems"])
        self.assertFalse(any("configured workload" in p
                             for p in report_for(["W1", "W2"])[
                                 "coverage_problems"]))


class MainExitCodes(Scratch):
    def setUp(self):
        super().setUp()
        self.manifest = self.write_json("api_classes.json", MANIFEST)
        self.overlay = self.write_overlay(OVERLAY)

    def argv(self, *extra, overlay=None, manifest=None):
        return ["--manifest", str(manifest or self.manifest),
                "--overlay", str(overlay or self.overlay), *extra]

    def test_valid_inputs_succeed(self):
        rc, err = run_main(self.argv("--summary"))
        self.assertEqual((rc, err), (0, ""))

    def report_args(self, observed):
        results = self.results(("T0.json", observation(
            "T0", captured={"hipA": 1, "hipB": 1},
            observed={"hipA": observed, "hipB": "NOOP"})))
        report_dir = self.dir / "out" / "report"
        return report_dir, self.argv("--results", str(results),
                                     "--report-dir", str(report_dir))

    def test_clean_run_writes_reports_into_a_new_directory(self):
        report_dir, argv = self.report_args("REAL")
        rc, err = run_main(argv)
        self.assertEqual((rc, err), (0, ""))
        self.assertTrue(json.loads(
            (report_dir / "matrix_report.json").read_text())["ok"])
        self.assertIn("# HRR per-API playback matrix",
                      (report_dir / "matrix_report.md").read_text())

    def test_divergence_returns_1_and_still_writes_the_report(self):
        report_dir, argv = self.report_args("NOOP")
        rc, err = run_main(argv)
        self.assertEqual(rc, 1)
        self.assertIn("hipA", err)
        report = json.loads((report_dir / "matrix_report.json").read_text())
        self.assertFalse(report["ok"])
        self.assertEqual([r["api"] for r in report["failures"]], ["hipA"])

    def test_unwritable_report_dir_returns_2(self):
        blocker = self.dir / "blocker"
        blocker.write_text("a file, not a directory")
        _, argv = self.report_args("REAL")
        argv[argv.index("--report-dir") + 1] = str(blocker)
        self.assertEqual(run_main(argv)[0], 2)

    def test_check_workloads_succeeds_when_sources_define_every_workload(self):
        overlay = copy.deepcopy(OVERLAY)
        names = {"W0": "Unit_HRR_W0_Direct", "W1": "Unit_HRR_W1_Direct",
                 "W2": "Unit_HRR_W2_Direct"}
        overlay["defaults"]["workloads"] = [names["W0"]]
        for tier in overlay["tiers"].values():
            tier["workloads"] = [names[w] for w in tier["workloads"]]
        tests = self.dir / "integration"
        tests.mkdir()
        (tests / "hrr_api_matrix_workload_test.cc").write_text(
            "// caf\u00e9 \u2014 workloads\n" + "".join(
                f'TEST_CASE("{n}", "[.]") {{}}\n' for n in names.values()),
            encoding="utf-8")
        rc, err = run_main(self.argv(
            "--check-workloads", "--test-dir", str(tests),
            overlay=self.write_overlay(overlay)))
        self.assertEqual((rc, err), (0, ""))

    def test_check_workloads_fails_for_an_undefined_workload(self):
        tests = self.dir / "integration"
        tests.mkdir()
        rc, err = run_main(self.argv("--check-workloads",
                                     "--test-dir", str(tests)))
        self.assertEqual(rc, 1)
        self.assertIn("W0", err)

    def test_malformed_overlay_returns_2(self):
        bad = copy.deepcopy(OVERLAY)
        bad["tiers"]["T0"]["min_cover"] = 1
        for extra in (["--summary"], ["--check-workloads"]):
            with self.subTest(extra):
                rc, err = run_main(self.argv(
                    *extra, overlay=self.write_overlay(bad)))
                self.assertEqual(rc, 2)
                self.assertIn("min_cover", err)

    def test_bad_manifest_returns_2(self):
        broken = self.dir / "broken.json"
        broken.write_text("{")
        self.assertEqual(run_main(self.argv("--summary", manifest=broken))[0], 2)

    def test_bad_results_return_2(self):
        missing = self.dir / "nowhere"
        self.assertEqual(run_main(self.argv("--results", str(missing)))[0], 2)
        mislabeled = self.results(("T0.json", observation("T1")))
        rc, err = run_main(self.argv("--results", str(mislabeled)))
        self.assertEqual(rc, 2)
        self.assertIn("declares tier T1", err)


class DeriveManifest(Scratch):
    COUNTS = {"total": 2, "observable_class.REAL": 2}

    def setUp(self):
        super().setUp()
        self.out = self.dir / "api_classes.json"
        self.baseline = self.dir / "baseline.json"
        self.manifest = {"counts": dict(self.COUNTS), "apis": []}

    def publish(self, update=False):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return dm.publish(self.manifest, self.out, self.baseline,
                              update, True)

    def test_missing_baseline_is_an_error_and_writes_nothing(self):
        with self.assertRaises(SystemExit) as ctx:
            self.publish()
        self.assertIn("--update-baseline", str(ctx.exception))
        self.assertFalse(self.out.exists())
        self.assertFalse(self.baseline.exists())

    def test_unreadable_baseline_is_an_error_and_writes_nothing(self):
        for text in ("{", "[]"):
            with self.subTest(text):
                self.baseline.write_text(text)
                with self.assertRaises(SystemExit):
                    self.publish()
                self.assertFalse(self.out.exists())

    def test_drift_fails_without_writing_or_replacing_the_manifest(self):
        self.baseline.write_text(json.dumps({"total": 3}))
        self.assertEqual(self.publish(), 1)
        self.assertFalse(self.out.exists())

        self.out.write_text("previous")
        self.assertEqual(self.publish(), 1)
        self.assertEqual(self.out.read_text(), "previous")
        self.assertEqual(json.loads(self.baseline.read_text()), {"total": 3})

    def test_matching_baseline_writes_the_manifest(self):
        self.baseline.write_text(json.dumps(self.COUNTS))
        self.assertEqual(self.publish(), 0)
        self.assertEqual(json.loads(self.out.read_text()), self.manifest)

    def test_update_baseline_writes_manifest_then_baseline(self):
        order = []
        real = dm.write_atomic

        def recording(path, text):
            order.append(path)
            real(path, text)

        with mock.patch.object(dm, "write_atomic", recording):
            self.assertEqual(self.publish(update=True), 0)
        self.assertEqual(order, [self.out, self.baseline])
        self.assertEqual(json.loads(self.baseline.read_text()), self.COUNTS)

    def test_update_baseline_accepts_a_drifted_baseline(self):
        self.baseline.write_text(json.dumps({"total": 3}))
        self.assertEqual(self.publish(update=True), 0)
        self.assertEqual(json.loads(self.baseline.read_text()), self.COUNTS)

    def test_failed_manifest_write_leaves_the_baseline_alone(self):
        self.baseline.write_text(json.dumps({"total": 3}))

        def failing(path, text):
            raise OSError("disk full")

        with mock.patch.object(dm, "write_atomic", failing):
            with self.assertRaises(OSError):
                self.publish(update=True)
        self.assertEqual(json.loads(self.baseline.read_text()), {"total": 3})

    def test_write_atomic_leaves_no_partial_file_on_failure(self):
        target = self.dir / "target.json"
        target.write_text("old")
        with mock.patch.object(dm.os, "replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                dm.write_atomic(target, "new")
        self.assertEqual(target.read_text(), "old")
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()),
                         ["target.json"])

    def test_generator_missing_required_api_is_an_error(self):
        tools = self.dir / "tools"
        tools.mkdir()
        (tools / "gen_hrr_api_args.py").write_text(
            "NOOP_PLAYBACK_APIS = set()\n")
        with self.assertRaises(SystemExit) as ctx:
            dm.load_generator(tools)
        message = str(ctx.exception)
        self.assertIn("UNREPLAYABLE_PLAYBACK_APIS", message)
        self.assertIn("deref_covered_params", message)
        self.assertNotIn("NOOP_PLAYBACK_APIS,", message)


if __name__ == "__main__":
    unittest.main()
