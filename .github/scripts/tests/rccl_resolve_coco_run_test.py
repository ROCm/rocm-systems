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

GITHUB = Path(__file__).resolve().parents[2]
SCHEDULED_WORKFLOW = GITHUB / "workflows" / "rccl-coco-scheduled.yml"
PR_WORKFLOW = GITHUB / "workflows" / "rccl-coco-pr.yml"
CHECKOUT_ACTION = GITHUB / "actions" / "rccl-checkout-coco-harness" / "action.yml"


class HarnessCheckoutTest(unittest.TestCase):
    def test_checkout_targets_coco_repo(self):
        text = CHECKOUT_ACTION.read_text(encoding="utf-8")
        self.assertIn("repository: ROCm/coco", text)
        self.assertIn("repositories: coco", text)
        self.assertNotIn("tRCCL", text)

    def test_callers_select_the_ref_before_checkout(self):
        for path in (PR_WORKFLOW, SCHEDULED_WORKFLOW):
            text = path.read_text(encoding="utf-8")
            self.assertIn("rccl_resolve_coco_run.py select-ref", text, path.name)
            self.assertIn("steps.harness_ref.outputs.coco_ref", text, path.name)

    def test_pr_workflow_runs_on_release_branches(self):
        text = PR_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("'release/**'", text)


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


class SelectCocoRefTest(unittest.TestCase):
    def test_explicit_ref_wins(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="abc123",
                base_ref="release/rocm-rel-7.2",
                ref_name="develop",
            ),
            "abc123",
        )

    def test_release_base_uses_the_same_branch(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="",
                base_ref="release/therock-10.1",
                ref_name="42/merge",
            ),
            "release/therock-10.1",
        )

    def test_integration_base_tracks_main(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="",
                base_ref="develop",
                ref_name="42/merge",
            ),
            "main",
        )

    def test_dispatch_from_a_release_branch_follows_it(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="",
                base_ref="",
                ref_name="release/rocm-rel-7.2",
            ),
            "release/rocm-rel-7.2",
        )

    def test_schedule_on_develop_tracks_main(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="",
                base_ref="",
                ref_name="develop",
            ),
            "main",
        )

    def test_whitespace_explicit_ref_is_ignored(self):
        self.assertEqual(
            resolve.select_coco_ref(
                explicit="  ",
                base_ref="release/bkc/therock-10.2-20260922",
                ref_name="",
            ),
            "release/bkc/therock-10.2-20260922",
        )


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
