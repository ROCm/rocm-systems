#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_ci_report.py."""

import contextlib
import io
import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_ci_report as rpt  # noqa: E402

JOBS = [
    {"name": "Configure", "conclusion": "success", "html_url": "u0"},
    {"name": "Generated sources", "conclusion": "success", "html_url": "c0"},
    {"name": "Unit (ubuntu-24.04)", "conclusion": "success", "html_url": "u1"},
    {"name": "Unit (windows-2022)", "conclusion": "failure", "html_url": "u2"},
    {"name": "gpu / Build capture-enabled HIP and HRR", "conclusion": "success", "html_url": "u3"},
    {"name": "gpu / Integration (gfx94x)", "conclusion": "success", "html_url": "u4"},
    {"name": "gpu / Integration (gfx950)", "conclusion": "cancelled", "html_url": "u5"},
]

FAMILIES = [{"family": "gfx94x"}, {"family": "gfx950"}, {"family": "gfx110x"}]


class ClassifyTest(unittest.TestCase):
    def test_picks_hrr_jobs(self):
        rows = rpt.classify_jobs(JOBS)
        self.assertEqual(
            [(r["kind"], r["name"]) for r in rows],
            [
                ("Generated sources", ""),
                ("Unit", "ubuntu-24.04"),
                ("Unit", "windows-2022"),
                ("Build", ""),
                ("Integration", "gfx94x"),
                ("Integration", "gfx950"),
            ],
        )

    def test_missing_family(self):
        rows = rpt.classify_jobs(JOBS)
        self.assertEqual(rpt.missing_families(FAMILIES, rows), ["gfx110x"])

    def test_skipped_matrix_job_with_unexpanded_name_is_dropped(self):
        rows = rpt.classify_jobs(
            [
                {"name": "GPU / Integration (${{ matrix.family }})", "conclusion": "skipped"},
                {"name": "GPU / Integration (gfx94x)", "conclusion": "success"},
            ]
        )
        self.assertEqual([(r["kind"], r["name"]) for r in rows], [("Integration", "gfx94x")])

    def test_queued_job_reports_status(self):
        rows = rpt.classify_jobs([{"name": "gpu / Integration (gfx90a)", "status": "queued"}])
        self.assertEqual(rows[0]["conclusion"], "queued")
        self.assertFalse(rpt.overall_ok(rows, []))


class RocJitsuJobTest(unittest.TestCase):
    def test_rocjitsu_job_counts_as_the_family_integration_job(self):
        rows = rpt.classify_jobs(
            [
                {"name": "GPU / rocJitsu (gfx90a)", "conclusion": "success", "html_url": "u1"},
                {"name": "GPU / rocJitsu (gfx94x)", "conclusion": "failure", "html_url": "u2"},
            ]
        )
        self.assertEqual(
            [(r["kind"], r["name"], r["conclusion"]) for r in rows],
            [("Integration", "gfx90a", "success"), ("Integration", "gfx94x", "failure")],
        )
        families = [{"family": "gfx90a"}, {"family": "gfx94x"}]
        self.assertEqual(rpt.missing_families(families, rows), [])


    def test_a_failed_rocjitsu_build_fails_the_summary_and_is_no_missing_family(self):
        rows = rpt.classify_jobs(
            [
                {"name": "GPU / Build rocJitsu", "conclusion": "failure", "html_url": "u0"},
                {"name": "GPU / rocJitsu (gfx90a)", "conclusion": "skipped", "html_url": "u1"},
            ]
        )
        self.assertEqual(
            [(r["kind"], r["name"], r["conclusion"]) for r in rows],
            [("Build", "", "failure"), ("Integration", "gfx90a", "skipped")],
        )
        self.assertEqual(rpt.missing_families([{"family": "gfx90a"}], rows), [])
        self.assertFalse(rpt.overall_ok(rows, []))


class OverallTest(unittest.TestCase):
    def test_failure_or_cancel_fails(self):
        self.assertFalse(rpt.overall_ok(rpt.classify_jobs(JOBS), []))

    def test_all_green(self):
        rows = rpt.classify_jobs([JOBS[1], JOBS[2], JOBS[5]])
        self.assertTrue(rpt.overall_ok(rows, []))

    def test_missing_family_fails(self):
        rows = rpt.classify_jobs([JOBS[5]])
        self.assertFalse(rpt.overall_ok(rows, ["gfx950"]))


class RenderTest(unittest.TestCase):
    def test_header_names_base(self):
        base = {"repo": "ROCm/rocm-systems", "run_id": "7", "run_url": "https://r/7", "commit": "0123456789abcdef"}
        families = [{"family": "gfx94x", "runner": "linux-gfx942-1gpu"}, {"family": "gfx950"}]
        text = rpt.render_header("the change touches capture or replay", True, families, base)
        self.assertIn("GPU families and runners: gfx94x (`linux-gfx942-1gpu`), gfx950.", text)
        self.assertIn("[ROCm/rocm-systems run 7](https://r/7) at `0123456789ab`", text)

    def test_header_without_gpu(self):
        self.assertIn("GPU suite: not run.", rpt.render_header("why", False, [], None))

    def test_jobs_table_lists_missing(self):
        rows = rpt.classify_jobs(JOBS)
        text = rpt.render_jobs(rows, ["gfx110x"])
        self.assertIn("| Integration | gfx950 | [cancelled](u5) |", text)
        self.assertIn("| Integration | gfx110x | _did not run_ |", text)

    def test_main_appends_results_table(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=JOBS[:3]), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--reason", "why", "--results-dir", tmp])
        self.assertEqual(rc, 0)
        self.assertIn("## HRR results by platform", out.getvalue())

    def test_main_gives_skipped_families_a_row(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=JOBS[5:6]), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main([
                "--run-gpu", "true", "--results-dir", tmp,
                "--families", '[{"family":"gfx94x","runner":"r"}]',
                "--skipped-families", '[{"family":"gfx125x","reason":"disabled in ci.yaml"}]',
            ])
        self.assertEqual(rc, 0)
        self.assertIn("| Integration | gfx94x | — | — | — | — | _no results_ |", out.getvalue())
        self.assertIn("| Integration | gfx125x | — | — | — | — | _not run: disabled in ci.yaml_ |",
                      out.getvalue())

    def test_main_reports_job_listing_failure(self):
        err = subprocess.CalledProcessError(1, "gh", stderr="HTTP 403: rate limited\n")
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", side_effect=err), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--run-id", "7", "--results-dir", tmp])
        self.assertEqual(rc, 1)
        self.assertIn("::error::could not list the jobs of run 7: HTTP 403: rate limited", out.getvalue())

    def test_main_fails_when_configure_did_not_succeed(self):
        for result in ("failure", "cancelled"):
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(rpt, "list_jobs", return_value=JOBS[:1]), \
                    mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                    contextlib.redirect_stdout(io.StringIO()) as out:
                rc = rpt.main(["--configure-result", result, "--run-gpu", "", "--families", "",
                               "--results-dir", tmp])
            self.assertEqual(rc, 1, result)
            self.assertIn(f"Configure: {result}, so nothing else ran", out.getvalue())

    def test_run_39793856571_does_not_pass(self):
        """The jobs API was still queued, needs.gpu had failed, and the gfx950
        JUnit file already contained Unit_HRR_PinnedHostFork. All three must
        fail the summary; any one of them is enough."""
        jobs = [
            {"name": "Generated sources", "conclusion": "success", "html_url": "u0"},
            {"name": "HRR script tests", "conclusion": "success"},
            {"name": "Unit (ubuntu-24.04)", "conclusion": "success"},
            {"name": "Unit (windows-2022)", "conclusion": "success"},
            {"name": "GPU / Build capture-enabled HIP and HRR", "conclusion": "success"},
            {"name": "GPU / Build rocJitsu", "status": "in_progress", "html_url": "u1"},
            {"name": "GPU / rocJitsu (gfx950)", "status": "queued", "html_url": "u2"},
        ]
        needs = {
            "configure": {"result": "success"},
            "codegen": {"result": "success"},
            "scripts": {"result": "success"},
            "unit": {"result": "success"},
            "gpu": {"result": "failure"},
        }
        failed = (
            '<?xml version="1.0"?><testsuites><testsuite>'
            '<testcase name="Unit_HRR_PinnedHostFork"><failure message="42 == 0"/>'
            "</testcase></testsuite></testsuites>"
        )
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=jobs), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            (Path(tmp) / "integration-gfx950.xml").write_text(failed, encoding="utf-8")
            rc = rpt.main([
                "--run-gpu", "true", "--results-dir", tmp,
                "--families", '[{"family":"gfx950","runner":"ubuntu-24.04"}]',
                "--needs-json", json.dumps(needs),
            ])
        self.assertEqual(rc, 1)
        text = out.getvalue()
        self.assertIn("gpu is failure", text)
        self.assertIn("failed cases in Integration gfx950", text)
        self.assertIn("| Integration | gfx950 | [queued](u2) |", text)

    def test_junit_failure_fails_when_jobs_and_needs_succeeded(self):
        jobs = [
            {"name": "Generated sources", "conclusion": "success"},
            {"name": "HRR script tests", "conclusion": "success"},
            {"name": "Unit (ubuntu-24.04)", "conclusion": "success"},
            {"name": "GPU / rocJitsu (gfx94x)", "conclusion": "success"},
        ]
        needs = {name: {"result": "success"} for name in (
            "configure", "codegen", "scripts", "unit", "gpu")}
        failed = (
            '<?xml version="1.0"?><testsuites><testsuite>'
            '<testcase name="Unit_HRR_PinnedHostFork"><failure message="failed"/>'
            "</testcase></testsuite></testsuites>"
        )
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=jobs), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / "integration-gfx94x.xml").write_text(failed, encoding="utf-8")
            rc = rpt.main([
                "--run-gpu", "true", "--results-dir", tmp,
                "--families", '[{"family":"gfx94x"}]',
                "--needs-json", json.dumps(needs),
            ])
        self.assertEqual(rc, 1)

    def test_a_selected_cpu_job_that_was_skipped_fails(self):
        needs = {
            "configure": {"result": "success"},
            "codegen": {"result": "skipped"},
            "scripts": {"result": "success"},
            "unit": {"result": "success"},
            "gpu": {"result": "skipped"},
        }
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=JOBS[:3]), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--results-dir", tmp, "--needs-json", json.dumps(needs)])
        self.assertEqual(rc, 1)
        self.assertIn("codegen is skipped", out.getvalue())

    def test_green_needs_jobs_and_junit_pass(self):
        jobs = [
            {"name": "Generated sources", "conclusion": "success"},
            {"name": "HRR script tests", "conclusion": "success"},
            {"name": "Unit (ubuntu-24.04)", "conclusion": "success"},
            {"name": "Unit (windows-2022)", "conclusion": "success"},
            {"name": "GPU / rocJitsu (gfx94x)", "conclusion": "success"},
        ]
        needs = {name: {"result": "success"} for name in (
            "configure", "codegen", "scripts", "unit", "gpu")}
        passed = (
            '<?xml version="1.0"?><testsuites><testsuite>'
            '<testcase name="Unit_HRR_Ok"/></testsuite></testsuites>'
        )
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rpt, "list_jobs", return_value=jobs), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            (Path(tmp) / "integration-gfx94x.xml").write_text(passed, encoding="utf-8")
            rc = rpt.main([
                "--run-gpu", "true", "--results-dir", tmp,
                "--families", '[{"family":"gfx94x"}]',
                "--needs-json", json.dumps(needs),
            ])
        self.assertEqual(rc, 0)


class GhApiTest(unittest.TestCase):
    def test_reads_every_value_across_pages(self):
        out = subprocess.CompletedProcess([], 0, stdout='{"a": 1}\n{"a": 2}{"a": 3}\n7\n')
        with mock.patch.object(rpt.subprocess, "run", return_value=out) as run:
            self.assertEqual(rpt.gh_api("p", ".x[]"), [{"a": 1}, {"a": 2}, {"a": 3}, 7])
        self.assertEqual(run.call_args.args[0], ["gh", "api", "--paginate", "p", "--jq", ".x[]"])

    def test_empty_output_is_no_values(self):
        out = subprocess.CompletedProcess([], 0, stdout="\n")
        with mock.patch.object(rpt.subprocess, "run", return_value=out):
            self.assertEqual(rpt.gh_api("p", ".x[]"), [])

    def test_unreadable_workflow_id_fails_the_summary(self):
        with mock.patch.object(rpt, "gh_api", return_value=[]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--repo", "R", "--pass-through", "true", "--head-sha", "abc",
                           "--run-id", "100"])
        self.assertEqual(rc, 1)
        self.assertIn("could not find the HRR CI run that tested abc", out.getvalue())


class PassThroughTest(unittest.TestCase):
    """A label run with nothing to run reports the run that tested the
    commit; it must never report greener than that run."""
    LABEL_ARGS = ["--pass-through", "true", "--head-sha", "abc",
                  "--run-id", "100", "--run-attempt", "1"]
    GREEN = [JOBS[1], JOBS[2], JOBS[5]]  # generated, unit ubuntu, integration gfx94x
    SUMMARY_OK = {"name": "HRR CI Summary", "conclusion": "success", "html_url": "s0"}
    SUMMARY_RED = {"name": "HRR CI Summary", "conclusion": "failure", "html_url": "s1"}
    GREEN = GREEN + [SUMMARY_OK]
    # What a run whose configure selected nothing lists: its jobs, skipped.
    LABEL_ONLY = [
        {"name": "Configure", "conclusion": "success", "html_url": "u0"},
        {"name": "Generated sources", "conclusion": "skipped", "html_url": "k1"},
        {"name": "HRR script tests", "conclusion": "skipped", "html_url": "k2"},
        {"name": "Unit (${{ matrix.os }})", "conclusion": "skipped", "html_url": "k3"},
        {"name": "GPU", "conclusion": "skipped", "html_url": "k4"},
        SUMMARY_OK,
    ]

    def run_main(self, runs, extra=()):
        """runs: [(run id, attempt, created_at, jobs[, run_started_at])], as
        the API lists them."""
        listing = [{"id": rid, "run_attempt": att, "created_at": at,
                    "run_started_at": started[0] if started else at, "html_url": f"r{rid}"}
                   for rid, att, at, _, *started in runs]
        jobs = {(str(rid), str(att)): j for rid, att, _, j, *_ in runs}

        def gh_api(path, jq):
            if path == "repos/R/actions/runs/100":
                return [7]
            if path.startswith("repos/R/actions/workflows/7/runs?event=pull_request&head_sha=abc&"):
                return listing
            rid, att = path.split("/runs/")[1].split("/attempts/")
            return jobs[(rid, att.split("/")[0])]

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rpt, "gh_api", side_effect=gh_api), \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--results-dir", tmp.name, "--repo", "R",
                           *self.LABEL_ARGS, *extra])
        return rc, out.getvalue()

    def test_delivers_the_earlier_run_jobs(self):
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", self.GREEN)])
        self.assertEqual(rc, 0, text)
        self.assertIn("[run 99 attempt 1](r99)", text)
        self.assertIn("| Generated sources | — | [success](c0) |", text)

    def test_fails_when_the_earlier_run_failed(self):
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", JOBS + [self.SUMMARY_RED])])
        self.assertEqual(rc, 1)
        self.assertIn("a label event must not turn it into a pass", text)

    def test_fails_when_only_the_earlier_summary_failed(self):
        # e.g. a family that never reported: every HRR job is green.
        jobs = self.GREEN[:-1] + [self.SUMMARY_RED]
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", jobs)])
        self.assertEqual(rc, 1)
        self.assertIn("HRR CI Summary is failure", text)

    def test_skips_label_runs_and_takes_the_newest_tested_run(self):
        rc, text = self.run_main([
            (98, 1, "2026-10-15T09:00:00Z", self.GREEN),
            (99, 2, "2026-10-15T10:00:00Z", JOBS + [self.SUMMARY_RED]),  # newest that tested
            (101, 1, "2026-10-15T11:00:00Z", self.LABEL_ONLY),  # an earlier label run
            # This run, still in progress, with a green-looking job of its own.
            (100, 1, "2026-10-15T12:00:00Z", self.GREEN[:-1]),
        ])
        self.assertEqual(rc, 1, text)
        self.assertIn("[run 99 attempt 2](r99)", text)

    def test_fails_on_a_job_that_has_not_finished(self):
        jobs = [{"name": "HRR script tests", "conclusion": None, "status": "in_progress",
                 "html_url": "u9"},
                {"name": "HRR CI Summary", "conclusion": None, "status": "queued"}]
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", jobs)])
        self.assertEqual(rc, 1)
        self.assertIn("is in_progress", text)

    def test_fails_when_the_earlier_configure_failed(self):
        jobs = [{**j, "conclusion": "failure"} if j["name"] == "Configure" else j
                for j in self.LABEL_ONLY if j["name"] != "HRR CI Summary"] + [self.SUMMARY_RED]
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", jobs)])
        self.assertEqual(rc, 1, text)
        self.assertIn("[run 99 attempt 1](r99)", text)
        self.assertIn("HRR CI Summary is failure", text)

    def test_an_unfinished_configure_has_a_verdict_to_wait_for(self):
        jobs = [{"name": "Configure", "conclusion": None, "status": "in_progress"},
                {"name": "HRR CI Summary", "conclusion": None, "status": "queued"}]
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", jobs)])
        self.assertEqual(rc, 1, text)
        self.assertIn("[run 99 attempt 1](r99)", text)

    def test_a_rerun_counts_from_its_latest_attempt(self):
        rc, text = self.run_main([
            (98, 2, "2026-10-15T09:00:00Z", JOBS + [self.SUMMARY_RED], "2026-10-15T12:00:00Z"),
            (99, 1, "2026-10-15T10:00:00Z", self.GREEN),
        ])
        self.assertEqual(rc, 1, text)
        self.assertIn("[run 98 attempt 2](r98)", text)

    def test_needs_the_head_sha(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            rpt.main(["--repo", "R", "--pass-through", "true", "--run-id", "100"])

    def test_passes_when_no_run_tested_the_commit(self):
        # A pull request that touches no HRR path: no run ran an HRR job.
        rc, text = self.run_main([(99, 1, "2026-10-15T10:00:00Z", self.LABEL_ONLY)])
        self.assertEqual(rc, 0, text)
        self.assertIn("nothing to report", text)

    def test_fails_when_the_lookup_fails(self):
        err = subprocess.CalledProcessError(1, "gh", stderr="HTTP 502")
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rpt, "gh_api", side_effect=err), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--results-dir", tmp.name, "--repo", "R", *self.LABEL_ARGS])
        self.assertEqual(rc, 1)
        self.assertIn("HTTP 502", out.getvalue())

    def test_ordinary_run_reports_its_own_jobs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rpt, "list_jobs", return_value=self.GREEN) as listed, \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()):
            rc = rpt.main(["--results-dir", tmp.name, "--repo", "R", "--pass-through", "false",
                           "--run-id", "100", "--run-attempt", "2"])
        self.assertEqual(rc, 0)
        listed.assert_called_once_with("R", "100", "2")


if __name__ == "__main__":
    unittest.main()
