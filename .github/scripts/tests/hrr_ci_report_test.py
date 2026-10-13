# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for hrr_ci_report.py."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import hrr_ci_report as rpt

REPO_ROOT = Path(__file__).resolve().parents[3]

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
        text = rpt.render_header("the change touches capture or replay", True, FAMILIES, base)
        self.assertIn("GPU families: gfx94x, gfx950, gfx110x.", text)
        self.assertIn("[ROCm/rocm-systems run 7](https://r/7) at `0123456789ab`", text)

    def test_header_without_gpu(self):
        self.assertIn("GPU suite: not run.", rpt.render_header("why", False, [], None))

    def test_jobs_table_lists_missing(self):
        rows = rpt.classify_jobs(JOBS)
        text = rpt.render_jobs(rows, ["gfx110x"])
        self.assertIn("| Integration | gfx950 | [cancelled](u5) |", text)
        self.assertIn("| Integration | gfx110x | _did not run_ |", text)

    def test_summarize_junit_loads(self):
        module = rpt.load_summarize_junit(
            REPO_ROOT / "projects/hrr/tests/scripts/summarize_junit.py"
        )
        self.assertTrue(callable(module.render))


if __name__ == "__main__":
    unittest.main()
