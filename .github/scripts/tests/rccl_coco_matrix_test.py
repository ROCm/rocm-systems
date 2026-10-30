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
        self.assertIn("tensorwave-rccl-perf-nightly", jobs)
        self.assertIn("tensorwave-rccl-unit-nightly", jobs)
        self.assertIn("crusoe-rccl-perf-nightly", jobs)
        self.assertIn("crusoe-rccl-unit-nightly", jobs)
        self.assertIn("tensorwave-rccl-gin-unit-nightly", jobs)
        self.assertIn("crusoe-rccl-gin-unit-nightly", jobs)
        self.assertNotIn("ruby64-rccl-perf-weekly", jobs)

    def test_weekly_emits_only_weekly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("weekly")
        jobs = {entry["job"] for entry in json.loads(emit.call_args.args[0]["matrix"])["include"]}
        self.assertEqual(
            jobs,
            {
                "ruby64-rccl-perf-weekly",
                "oci-rccl-perf-weekly",
                "tensorwave-rccl-perf-weekly",
                "crusoe-rccl-perf-weekly",
            },
        )

    def test_monthly_emits_only_monthly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("monthly")
        outputs = emit.call_args.args[0]
        self.assertEqual(outputs["has_runs"], "true")
        jobs = {entry["job"] for entry in json.loads(outputs["matrix"])["include"]}
        self.assertEqual(
            jobs,
            {
                "ruby64-rccl-perf-monthly",
                "oci-rccl-perf-monthly",
                "tensorwave-rccl-perf-monthly",
                "crusoe-rccl-perf-monthly",
            },
        )


# Job names use the coco prefix, which is not the cluster key (ruby64 vs ruby).
JOB_PREFIX_CLUSTERS = {
    "ruby64-": "ruby",
    "oci-": "oci",
    "tensorwave-": "tw",
    "crusoe-": "crusoe",
}


class ScheduledRunnerPairingTest(unittest.TestCase):
    """Each tRCCL job is bound to one cluster, so its runner must be that cluster's."""

    def _emitted(self, mode):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs(mode)
        return json.loads(emit.call_args.args[0]["matrix"])["include"]

    def test_every_catalogue_cluster_has_a_runner(self):
        for _mode, job, cluster, _kind, _timeout in matrix.SCHEDULED_CATALOGUE:
            self.assertIn(cluster, matrix.SCHEDULED_RUNNERS, job)

    def test_every_job_gets_the_runner_for_its_prefix(self):
        for mode in ("nightly", "weekly", "monthly"):
            for entry in self._emitted(mode):
                prefixes = [p for p in JOB_PREFIX_CLUSTERS if entry["job"].startswith(p)]
                self.assertEqual(len(prefixes), 1, entry["job"])
                cluster = JOB_PREFIX_CLUSTERS[prefixes[0]]
                self.assertEqual(
                    entry["runner"], matrix.SCHEDULED_RUNNERS[cluster], entry["job"]
                )

    def test_tw_and_crusoe_legs_carry_group_and_timeout(self):
        emitted = {
            mode: self._emitted(mode) for mode in ("nightly", "weekly", "monthly")
        }
        rows = [
            row for row in matrix.SCHEDULED_CATALOGUE if row[2] in {"tw", "crusoe"}
        ]
        self.assertTrue(rows)
        for mode, job, cluster, kind, timeout in rows:
            entry = next(e for e in emitted[mode] if e["job"] == job)
            self.assertEqual(entry["group"], f"rccl-coco-{mode}-{cluster}-{kind}")
            self.assertEqual(entry["timeout"], timeout)


class PrClustersTest(unittest.TestCase):
    def test_runners_are_on_the_cluster_allowlist(self):
        allowlist = (
            Path(__file__).resolve().parents[1]
            / "allowlist"
            / "rccl-cluster-runners.allowlist"
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
