# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rccl_resolve_coco_run.py."""

import contextlib
import io
import os
import re
import shutil
import sys
import tempfile
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
        self.assertEqual(crons, {resolve.WEEKDAY_NIGHTLY_CRON, resolve.SATURDAY_CRON})


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


HEAD = "a" * 40
MERGE_BASE = "b" * 40


def fake_github(merge_base=MERGE_BASE, base_ref="develop"):
    """A GitHub REST stub serving one PR and its compare against the base branch."""
    responses = {
        "https://api.github.com/repos/ROCm/rocm-systems/pulls/42": {
            "base": {"ref": base_ref}
        },
        f"https://api.github.com/repos/ROCm/rocm-systems/compare/{base_ref}...{HEAD}": {
            "merge_base_commit": {"sha": merge_base}
        },
    }
    return responses.__getitem__


class BaselineShaArgsTest(unittest.TestCase):
    def test_pr_args_pin_the_baseline_to_the_given_commit(self):
        args = resolve.build_coco_args(
            mode="pr", dry_run=False, baseline_sha=MERGE_BASE
        )
        self.assertIn(f"--baseline-sha {MERGE_BASE}", args)

    def test_pr_args_without_a_commit_keep_the_configured_baseline(self):
        self.assertNotIn(
            "--baseline-sha", resolve.build_coco_args(mode="pr", dry_run=False)
        )

    def test_periodic_args_never_pin(self):
        args = resolve.build_coco_args(
            mode="nightly", dry_run=False, baseline_sha=MERGE_BASE
        )
        self.assertNotIn("--baseline-sha", args)


class ResolveMergeBaseTest(unittest.TestCase):
    def test_returns_the_merge_base_with_the_prs_base_branch(self):
        """The baseline arm is built here, so a PR is measured only against code it contains."""
        self.assertEqual(
            resolve.resolve_merge_base(
                pr_repo="ROCm/rocm-systems",
                pr_number="42",
                pr_sha=HEAD,
                get=fake_github(),
            ),
            MERGE_BASE,
        )

    def test_uses_the_prs_own_base_branch(self):
        self.assertEqual(
            resolve.resolve_merge_base(
                pr_repo="ROCm/rocm-systems",
                pr_number="42",
                pr_sha=HEAD,
                get=fake_github(base_ref="rccl-develop"),
            ),
            MERGE_BASE,
        )

    def test_a_head_that_is_its_own_merge_base_is_refused(self):
        """Both arms on one commit would compare the PR with itself and always pass."""
        with self.assertRaises(ValueError):
            resolve.resolve_merge_base(
                pr_repo="ROCm/rocm-systems",
                pr_number="42",
                pr_sha=HEAD,
                get=fake_github(merge_base=HEAD),
            )

    def test_a_malformed_merge_base_is_refused(self):
        with self.assertRaises(ValueError):
            resolve.resolve_merge_base(
                pr_repo="ROCm/rocm-systems",
                pr_number="42",
                pr_sha=HEAD,
                get=fake_github(merge_base="deadbeef"),
            )


class ResolveBaselineShaTest(unittest.TestCase):
    def _harness(self, pr_command: str | None) -> Path:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        if pr_command is not None:
            path = root / resolve.HARNESS_PR_COMMAND
            path.parent.mkdir(parents=True)
            path.write_text(pr_command, encoding="utf-8")
        return root

    def test_periodic_runs_keep_the_configured_baseline(self):
        self.assertEqual(
            resolve.resolve_baseline_sha(
                pr_number="",
                pr_sha="",
                pr_repo="",
                harness_dir=self._harness('"--baseline-sha"'),
                get=fake_github(),
            ),
            "",
        )

    def test_a_pr_run_pins_to_the_merge_base_when_the_harness_can(self):
        self.assertEqual(
            resolve.resolve_baseline_sha(
                pr_number="42",
                pr_sha=HEAD,
                pr_repo="ROCm/rocm-systems",
                harness_dir=self._harness('@click.option(\n    "--baseline-sha",'),
                get=fake_github(),
            ),
            MERGE_BASE,
        )

    def test_an_older_harness_is_not_passed_a_flag_it_would_reject(self):
        """The harness ref (1.3.x) may predate --baseline-sha; passing it would fail every gate."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            sha = resolve.resolve_baseline_sha(
                pr_number="42",
                pr_sha=HEAD,
                pr_repo="ROCm/rocm-systems",
                harness_dir=self._harness('"--skip-baselines"'),
                get=fake_github(),
            )
        self.assertEqual(sha, "")
        self.assertIn("::warning::", out.getvalue())
