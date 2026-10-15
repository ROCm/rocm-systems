#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for hrr_ci_report.py."""

import contextlib
import io
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

    def test_queued_job_reports_status(self):
        rows = rpt.classify_jobs([{"name": "gpu / Integration (gfx90a)", "status": "queued"}])
        self.assertEqual(rows[0]["conclusion"], "queued")


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


class PassThroughTest(unittest.TestCase):
    """A label run with nothing to run reports the run that tested the
    commit; it must never report greener than that run."""
    LABEL_ARGS = ["--pass-through", "--upstream-run", "99", "--upstream-attempt", "1",
                  "--run-id", "100", "--run-attempt", "1"]
    GREEN = [JOBS[1], JOBS[2], JOBS[5]]  # generated, unit ubuntu, integration gfx94x

    def run_main(self, jobs, extra=()):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rpt, "list_jobs", return_value=jobs) as listed, \
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": ""}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            rc = rpt.main(["--results-dir", tmp.name, *self.LABEL_ARGS, *extra])
        return rc, out.getvalue(), listed

    def test_delivers_the_earlier_run_jobs(self):
        rc, text, listed = self.run_main(self.GREEN)
        self.assertEqual(rc, 0, text)
        listed.assert_called_once_with(mock.ANY, "99", "1")
        self.assertIn("| Generated sources | — | [success](c0) |", text)

    def test_fails_when_the_earlier_run_failed(self):
        rc, text, _ = self.run_main(JOBS)  # windows unit failed, gfx950 cancelled
        self.assertEqual(rc, 1)
        self.assertIn("a label event must not turn it into a pass", text)

    def test_fails_on_a_job_that_has_not_finished(self):
        jobs = [{"name": "HRR script tests", "conclusion": "in_progress", "html_url": "u9"}]
        rc, text, _ = self.run_main(jobs)
        self.assertEqual(rc, 1)
        self.assertIn("is in_progress", text)

    def test_fails_when_no_earlier_run_was_found(self):
        rc, text, _ = self.run_main(self.GREEN, extra=["--upstream-attempt", ""])
        self.assertEqual(rc, 1)
        self.assertIn("no earlier run of HRR CI found", text)

    def test_fails_when_the_earlier_run_has_no_hrr_jobs(self):
        rc, text, _ = self.run_main([JOBS[0]])  # only Configure matches nothing
        self.assertEqual(rc, 1)
        self.assertIn("lists no HRR jobs", text)


if __name__ == "__main__":
    unittest.main()
