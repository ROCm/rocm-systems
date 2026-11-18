# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise independently emitted check results through the policy CLI."""

import contextlib
import io
import itertools
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "tools/systems_pr_bot"))

import policy_check

POLICY_PATH = REPOSITORY / "tools/systems_pr_bot/policy.yml"
POLICY = policy_check.load_policy(POLICY_PATH)


class RequiredCheckIdentityTests(unittest.TestCase):
    def checks(self, duplicate, reverse=False):
        scoped = {"name": "pre-commit", "status": "completed", "conclusion": "success"}
        checks = [duplicate, scoped] if not reverse else [scoped, duplicate]
        return checks + [
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in POLICY.required_checks
            if name != "pre-commit"
        ]

    def invoke(self, batches, timeout=5):
        output = io.StringIO()
        reads = Mock(side_effect=batches)
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        "GH_TOKEN": "fixture-token",
                        "OWNER": "ROCm",
                        "REPO": "rocm-systems",
                        "PR_NUMBER": "1",
                        "SHA": "fixture-head",
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    policy_check,
                    "get_pr",
                    return_value={
                        "body": "ISSUE ID: #1\nCheck independently emitted formatting results.",
                        "user": {"login": "human"},
                    },
                )
            )
            stack.enter_context(
                patch.object(
                    policy_check,
                    "iter_pr_files",
                    return_value=iter(
                        [
                            {
                                "filename": "projects/rccl/src/init.cc",
                                "status": "modified",
                            }
                        ]
                    ),
                )
            )
            stack.enter_context(patch.object(policy_check, "get_check_runs", reads))
            for operation in (
                "upsert_comment",
                "remove_label",
                "update_comment_if_exists",
                "add_label",
            ):
                stack.enter_context(patch.object(policy_check, operation))
            stack.enter_context(patch.object(policy_check.time, "sleep"))
            stack.enter_context(
                patch.object(policy_check.time, "time", side_effect=itertools.count())
            )
            stack.enter_context(contextlib.redirect_stdout(output))
            result = policy_check.main(
                [
                    "--policy",
                    str(POLICY_PATH),
                    "--timeout-seconds",
                    str(timeout),
                    "--poll-seconds",
                    "0",
                ]
            )
        return result, reads.call_count, output.getvalue()

    def test_failure_from_either_producer_blocks_readiness(self):
        duplicate = {
            "name": "pre-commit",
            "status": "completed",
            "conclusion": "failure",
        }
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                checks = self.checks(duplicate, reverse)
                result, _, output = self.invoke([checks])
                self.assertEqual(result, 1, output)
                self.assertIn("pre-commit=failure", output)
                rows = {
                    r.name: r for r in policy_check.build_check_results(POLICY, checks)
                }
                self.assertFalse(rows["pre-commit"].passed)
                self.assertFalse(rows["pre-commit"].pending)

    def test_pending_producer_requires_another_poll(self):
        for conclusion in (None, "success"):
            duplicate = {
                "name": "pre-commit",
                "status": "in_progress",
                "conclusion": conclusion,
            }
            for reverse in (False, True):
                with self.subTest(conclusion=conclusion, reverse=reverse):
                    checks = self.checks(duplicate, reverse)
                    completed = self.checks(
                        {
                            "name": "pre-commit",
                            "status": "completed",
                            "conclusion": "success",
                        },
                        reverse,
                    )
                    result, polls, output = self.invoke([checks, completed])
                    self.assertEqual(result, 0, output)
                    self.assertEqual(polls, 2, output)
                    rows = {
                        r.name: r
                        for r in policy_check.build_check_results(POLICY, checks)
                    }
                    self.assertFalse(rows["pre-commit"].passed)
                    self.assertTrue(rows["pre-commit"].pending)

    def test_completed_success_neutral_and_skipped_remain_accepted(self):
        for conclusion in ("success", "neutral", "skipped"):
            with self.subTest(conclusion=conclusion):
                result, _, output = self.invoke(
                    [
                        self.checks(
                            {
                                "name": "pre-commit",
                                "status": "completed",
                                "conclusion": conclusion,
                            }
                        )
                    ]
                )
                self.assertEqual(result, 0, output)

    def test_missing_required_name_cannot_pass(self):
        checks = [
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in POLICY.required_checks
            if name != "pre-commit / cuid"
        ]
        result, _, output = self.invoke([checks], timeout=0)
        self.assertEqual(result, 1, output)
        self.assertIn("Timed out", output)


if __name__ == "__main__":
    unittest.main()
