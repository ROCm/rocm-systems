# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_coco_matrix.py."""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_coco_matrix as matrix


class ScheduledRunsTest(unittest.TestCase):
    def test_nightly_includes_all_nightly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("nightly")
        payload = json.loads(emit.call_args.args[0]["matrix"])["include"]
        jobs = {entry["job"] for entry in payload}
        self.assertIn("ruby64-rccl-unit-nightly", jobs)
        self.assertIn("oci-nixl-nightly", jobs)
        self.assertNotIn("ruby64-rccl-perf-weekly", jobs)

    def test_weekly_emits_only_weekly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("weekly")
        jobs = {entry["job"] for entry in json.loads(emit.call_args.args[0]["matrix"])["include"]}
        self.assertEqual(jobs, {"ruby64-rccl-perf-weekly", "oci-rccl-perf-weekly"})

    def test_monthly_emits_only_monthly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("monthly")
        outputs = emit.call_args.args[0]
        self.assertEqual(outputs["has_runs"], "true")
        jobs = {entry["job"] for entry in json.loads(outputs["matrix"])["include"]}
        self.assertEqual(jobs, {"ruby64-rccl-perf-monthly", "oci-rccl-perf-monthly"})


class PrClustersTest(unittest.TestCase):
    def test_runners_are_on_the_cluster_allowlist(self):
        allowlist = (
            Path(__file__).resolve().parents[1]
            / "allowlist"
            / "cluster-runners.allowlist"
        )
        listed = {
            line.split("#", 1)[0].strip()
            for line in allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith(".github/")
        }
        for entry in matrix.PR_CLUSTERS:
            self.assertIn(entry["runner"], listed)
        for runner in matrix.SCHEDULED_RUNNERS.values():
            self.assertIn(runner, listed)

    def test_emits_ruby_and_oci(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.pr_clusters()
        include = json.loads(emit.call_args.args[0]["matrix"])["include"]
        self.assertEqual({entry["prefix"] for entry in include}, {"ruby64", "oci"})


if __name__ == "__main__":
    unittest.main()
