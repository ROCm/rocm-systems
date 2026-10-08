#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the GPU schedule budget. No runner and no GitHub API."""

import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hrr_ci_schedule as sched  # noqa: E402


class ScheduleActionTest(unittest.TestCase):
    def test_queued_inside_budget_keeps_waiting(self):
        self.assertEqual(sched.schedule_action(0, "queued"), "poll")
        self.assertEqual(sched.schedule_action(sched.SCHEDULE_LIMIT_S - 1, ""), "poll")
        self.assertEqual(sched.schedule_action(10, None, "queued"), "poll")

    def test_runner_acceptance_is_not_a_skip(self):
        self.assertEqual(sched.schedule_action(0, "in_progress"), "run")
        self.assertEqual(
            sched.schedule_action(sched.SCHEDULE_LIMIT_S + 5, "completed"), "run"
        )

    def test_finished_run_is_propagated_even_without_a_job_status(self):
        self.assertEqual(sched.schedule_action(5, None, "completed"), "run")

    def test_queue_past_budget_skips(self):
        for status in ("queued", "waiting", "pending", "", None):
            self.assertEqual(
                sched.schedule_action(sched.SCHEDULE_LIMIT_S, status, "in_progress"),
                "skip",
                status,
            )

    def test_in_progress_run_does_not_count_as_the_gpu_job(self):
        # The dispatched run leaves the queue as soon as its hosted gate
        # starts. That must not consume the GPU job's schedule budget.
        self.assertEqual(
            sched.schedule_action(1, "queued", "in_progress"), "poll"
        )


class ChooseDispatchedRunTest(unittest.TestCase):
    def test_rerun_ignores_the_previous_attempt(self):
        title = "hrr-gpu-37789160788-gfx90a"
        runs = [
            {
                "databaseId": 37794673233,
                "displayTitle": title,
                "name": "HRR CI",
                "event": "workflow_dispatch",
                "createdAt": "2026-10-08T14:42:32Z",
            },
            {
                "databaseId": 37789851414,
                "displayTitle": title,
                "name": "HRR CI",
                "event": "workflow_dispatch",
                "createdAt": "2026-10-08T14:07:11Z",
            },
        ]
        # The list can put the finished attempt first. That result is already
        # known and must not be downloaded again.
        self.assertEqual(
            sched.choose_dispatched_run(list(reversed(runs)), title, ["37789851414"]),
            "37794673233",
        )

    def test_no_new_run_yet(self):
        title = "hrr-gpu-1-2-gfx90a"
        runs = [{
            "databaseId": 9,
            "displayTitle": title,
            "event": "workflow_dispatch",
            "createdAt": "2026-10-08T14:07:11Z",
        }]
        self.assertEqual(sched.choose_dispatched_run(runs, title, ["9"]), "")


class SkipJunitTest(unittest.TestCase):
    def test_message_is_a_single_skip(self):
        root = ET.fromstring(
            sched.skip_junit('GPU runner has no visible ROCm-capable device')
        )
        skipped = root.findall(".//skipped")
        self.assertEqual(len(skipped), 1)
        self.assertEqual(
            skipped[0].attrib["message"],
            "GPU runner has no visible ROCm-capable device",
        )
        self.assertEqual(root.find(".//testcase").attrib["name"], "integration suite")

    def test_cli_writes_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "integration-gfx1153.xml"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(sched.__file__).resolve()),
                    "--write-skip",
                    str(path),
                    "--message",
                    "not scheduled within 30 minutes",
                ],
                check=True,
            )
            message = ET.parse(path).find(".//skipped").attrib["message"]
            self.assertEqual(message, "not scheduled within 30 minutes")


if __name__ == "__main__":
    unittest.main()
