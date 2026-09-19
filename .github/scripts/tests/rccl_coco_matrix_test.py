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

    def test_monthly_emits_only_monthly_jobs(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.scheduled_runs("monthly")
        outputs = emit.call_args.args[0]
        self.assertEqual(outputs["has_runs"], "true")
        jobs = {entry["job"] for entry in json.loads(outputs["matrix"])["include"]}
        self.assertEqual(jobs, {"ruby64-rccl-perf-monthly", "oci-rccl-perf-monthly"})


class PrClustersTest(unittest.TestCase):
    def test_emits_ruby_and_oci(self):
        with mock.patch.object(matrix, "set_github_output") as emit:
            matrix.pr_clusters()
        include = json.loads(emit.call_args.args[0]["matrix"])["include"]
        self.assertEqual({entry["prefix"] for entry in include}, {"ruby64", "oci"})


if __name__ == "__main__":
    unittest.main()
