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
        duplicate = duplicate | {"check_suite": {"id": 200}}
        scoped = {
            "name": "pre-commit",
            "status": "completed",
            "conclusion": "success",
            "check_suite": {"id": 100},
        }
        checks = [duplicate, scoped] if not reverse else [scoped, duplicate]
        return checks + [
            {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "check_suite": {"id": 100},
            }
            for name in (check.name for check in POLICY.required_checks)
            if name != "pre-commit"
        ]

    def invoke(self, batches, timeout=5):
        output = io.StringIO()
        reads = Mock(side_effect=batches)
        active = []

        def get(url, token):
            nonlocal active
            if "/actions/workflows/" in url:
                active = reads()
                return {
                    "total_count": 1,
                    "workflow_runs": [
                        {
                            "id": 10,
                            "run_number": 10,
                            "check_suite_id": 100,
                            "event": "pull_request",
                            "head_sha": "fixture-head",
                            "display_title": "PR #1: develop@fixture-base",
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ],
                }
            if "/check-suites/100/" in url:
                # The first same-name record is the legacy producer; the
                # other record and distinct names belong to the scoped suite.
                scoped = [
                    run for run in active if run.get("check_suite", {}).get("id") == 100
                ]
                return {"total_count": len(scoped), "check_runs": scoped}
            if "/commits/fixture-head/" in url:
                return {"total_count": len(active), "check_runs": active}
            raise AssertionError(url)

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
                        "BASE_REF": "develop",
                        "BASE_SHA": "fixture-base",
                        "ACTIONS_TOKEN": "actions-token",
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
                        "head": {"sha": "fixture-head"},
                        "base": {"ref": "develop", "sha": "fixture-base"},
                        "changed_files": 1,
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
            stack.enter_context(patch.object(policy_check, "gh_get", side_effect=get))
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
            {
                "name": name,
                "status": "completed",
                "conclusion": "success",
                "check_suite": {"id": 100},
            }
            for name in (check.name for check in POLICY.required_checks)
            if name != "pre-commit / cuid"
        ]
        result, _, output = self.invoke([checks], timeout=0)
        self.assertEqual(result, 1, output)
        self.assertIn("Timed out", output)


if __name__ == "__main__":
    unittest.main()
