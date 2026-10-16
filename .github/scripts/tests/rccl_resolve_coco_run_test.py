# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_resolve_coco_run.py."""

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rccl_resolve_coco_run as resolve

SCHEDULED_WORKFLOW = (
    Path(__file__).resolve().parents[2] / "workflows" / "rccl-coco-scheduled.yml"
)


class CronConsistencyTest(unittest.TestCase):
    def test_workflow_crons_are_all_recognised(self):
        """schedule_to_mode raises on an unknown cron, failing every scheduled run."""
        text = SCHEDULED_WORKFLOW.read_text(encoding="utf-8")
        crons = set(re.findall(r"^\s*-\s*cron:\s*'([^']+)'", text, re.MULTILINE))
        self.assertTrue(crons, f"no cron entries found in {SCHEDULED_WORKFLOW}")
        self.assertEqual(
            crons, {resolve.WEEKDAY_NIGHTLY_CRON, resolve.SATURDAY_CRON}
        )


class ScheduleToModeTest(unittest.TestCase):
    def test_weekday_cron_is_nightly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 1-5", day_of_month=10),
            "nightly",
        )

    def test_saturday_before_third_week_is_weekly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 6", day_of_month=14),
            "weekly",
        )

    def test_saturday_third_week_is_monthly(self):
        self.assertEqual(
            resolve.schedule_to_mode("47 3 * * 6", day_of_month=18),
            "monthly",
        )

    def test_unknown_schedule_raises(self):
        with self.assertRaises(ValueError):
            resolve.schedule_to_mode("0 0 * * *", day_of_month=1)


class ResolveModeAndTriggerTest(unittest.TestCase):
    def test_pr_number_wins(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="42",
            event_name="schedule",
            schedule="47 3 * * 1-5",
            periodic_mode="nightly",
        )
        self.assertEqual((mode, kind), ("pr", "ci"))

    def test_schedule_event(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="",
            event_name="schedule",
            schedule="47 3 * * 1-5",
            periodic_mode="nightly",
            day_of_month=3,
        )
        self.assertEqual((mode, kind), ("nightly", "scheduled"))

    def test_dispatch_uses_periodic_mode(self):
        mode, kind = resolve.resolve_mode_and_trigger(
            pr_number="",
            event_name="workflow_dispatch",
            schedule="",
            periodic_mode="weekly",
        )
        self.assertEqual((mode, kind), ("weekly", "manual"))


class BuildCocoArgsTest(unittest.TestCase):
    def test_pr_skips_baselines_not_prune(self):
        args = resolve.build_coco_args(mode="pr", dry_run=False)
        self.assertIn("--skip-baselines", args)
        self.assertNotIn("--prune-build", args)

    def test_nightly_prunes(self):
        args = resolve.build_coco_args(mode="nightly", dry_run=False)
        self.assertIn("--prune-build", args)
        self.assertNotIn("--skip-baselines", args)


class ResolveRunTest(unittest.TestCase):
    def test_rejects_unknown_manual_mode(self):
        with self.assertRaises(ValueError):
            resolve.resolve_run(
                pr_number="",
                pr_sha="",
                pr_repo="",
                event_name="workflow_dispatch",
                schedule="",
                periodic_mode="not-a-mode",
                dry_run=False,
                coco_sha="abc123",
            )


if __name__ == "__main__":
    unittest.main()
