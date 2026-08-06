#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import rocjitsu_corpus_translation_comment as HELPER

REPOSITORY = "ROCm/rocm-systems"
HEAD_SHA = "a" * 40
NEW_HEAD_SHA = "b" * 40


class FakeApi:
    def __init__(self) -> None:
        self.pull = {
            "base": {"ref": "develop", "repo": {"full_name": REPOSITORY}},
            "head": {
                "repo": {"full_name": REPOSITORY},
                "sha": HEAD_SHA,
            },
        }
        self.runs = [{"conclusion": "success", "event": "push", "id": 200}]
        self.artifacts = {
            100: [self._artifact(HELPER.CANDIDATE_ARTIFACT)],
            200: [self._artifact(HELPER.BASELINE_ARTIFACT)],
        }
        self.comments = []
        self.calls = []

    def get(self, path):
        if path == f"/repos/{REPOSITORY}/pulls/7":
            return self.pull
        raise AssertionError(path)

    def paginate(self, path, key=None):
        if path.endswith(f"/commits/{HEAD_SHA}/pulls"):
            return []
        if path.endswith("/runs?branch=develop&status=completed"):
            self.assert_key(key, "workflow_runs")
            return self.runs
        if path.endswith("/comments"):
            self.assert_key(key, None)
            return self.comments
        if "/actions/runs/" in path and path.endswith("/artifacts"):
            self.assert_key(key, "artifacts")
            run_id = int(path.split("/actions/runs/")[1].split("/")[0])
            return self.artifacts.get(run_id, [])
        raise AssertionError(path)

    def create(self, path, payload):
        self.calls.append(("create", path, payload))

    def update(self, path, payload):
        self.calls.append(("update", path, payload))

    def delete(self, path):
        self.calls.append(("delete", path, None))

    @staticmethod
    def _artifact(name):
        return {"expired": False, "name": name, "size_in_bytes": 5_000_000}

    @staticmethod
    def assert_key(actual, expected):
        if actual != expected:
            raise AssertionError((actual, expected))


class PlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeApi()
        self.event = {
            "workflow_run": {
                "event": "pull_request",
                "conclusion": "success",
                "head_repository": {"full_name": REPOSITORY},
                "head_sha": HEAD_SHA,
                "id": 100,
                "path": HELPER.TRUSTED_WORKFLOW_PATH,
                "pull_requests": [{"number": 7}],
            }
        }

    def test_resolves_candidate_and_latest_baseline(self) -> None:
        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertTrue(plan.ready)
        self.assertEqual(plan.pull_number, 7)
        self.assertEqual(plan.candidate_run_id, 100)
        self.assertEqual(plan.baseline_run_id, 200)
        self.assertEqual(plan.candidate_head_sha, HEAD_SHA)

    def test_allows_bootstrap_without_baseline(self) -> None:
        self.api.runs = []

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertTrue(plan.ready)
        self.assertIsNone(plan.baseline_run_id)

    def test_skips_unsuccessful_source_run(self) -> None:
        self.event["workflow_run"]["conclusion"] = "failure"

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("did not succeed", plan.reason)

    def test_skips_untrusted_source_workflow_path(self) -> None:
        self.event["workflow_run"]["path"] = ".github/workflows/untrusted.yml"

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("workflow path", plan.reason)

    def test_skips_source_run_from_fork(self) -> None:
        self.event["workflow_run"]["head_repository"]["full_name"] = "someone/fork"

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("did not run from", plan.reason)

    def test_skips_stale_pull_request_revision(self) -> None:
        self.api.pull["head"]["sha"] = NEW_HEAD_SHA

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("does not match", plan.reason)

    def test_skips_pull_request_from_fork(self) -> None:
        self.api.pull["head"]["repo"]["full_name"] = "someone/fork"

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("same-repository", plan.reason)

    def test_ignores_unsuccessful_baseline_run(self) -> None:
        self.api.runs.insert(0, {"conclusion": "failure", "event": "push", "id": 201})
        self.api.artifacts[201] = [self.api._artifact(HELPER.BASELINE_ARTIFACT)]

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertEqual(plan.baseline_run_id, 200)

    def test_skips_pull_request_with_different_base(self) -> None:
        self.api.pull["base"]["ref"] = "release/7.0"

        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)

        self.assertFalse(plan.ready)
        self.assertIn("does not target", plan.reason)

    def test_rejects_oversized_candidate(self) -> None:
        self.api.artifacts[100][0]["size_in_bytes"] = HELPER.MAXIMUM_ARTIFACT_BYTES + 1

        with self.assertRaisesRegex(ValueError, "candidate artifact"):
            HELPER.resolve_plan(self.event, self.api, REPOSITORY)

    def test_writes_canonical_outputs(self) -> None:
        plan = HELPER.resolve_plan(self.event, self.api, REPOSITORY)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "outputs"

            HELPER.write_plan_outputs(output, plan)

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "ready=true\n"
                "pull-number=7\n"
                "candidate-run-id=100\n"
                f"candidate-head-sha={HEAD_SHA}\n"
                "baseline-found=true\n"
                "baseline-run-id=200\n",
            )


class GitHubApiTest(unittest.TestCase):
    def test_paginate_rejects_non_object_keyed_response(self) -> None:
        api = HELPER.GitHubApi(REPOSITORY, "token", "https://api.github.com")
        with mock.patch.object(api, "get", return_value=[]):
            with self.assertRaisesRegex(ValueError, "is not an object"):
                api.paginate("/artifacts", "artifacts")


class CommentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeApi()

    def test_renders_warning_and_resolution(self) -> None:
        warning = HELPER.render_comment(
            REPOSITORY,
            100,
            HEAD_SHA,
            True,
            2,
            "## report\n\n> [!WARNING]\n> changed\n",
            "https://github.com",
        )

        self.assertIn(HELPER.COMMENT_MARKER, warning)
        self.assertIn("[!WARNING]", warning)
        self.assertIn(HEAD_SHA, warning)
        self.assertIsNone(
            HELPER.render_comment(
                REPOSITORY,
                100,
                HEAD_SHA,
                True,
                0,
                "## clean",
                "https://github.com",
            )
        )

    def test_renders_missing_baseline_warning(self) -> None:
        body = HELPER.render_comment(
            REPOSITORY, 100, HEAD_SHA, False, None, None, "https://github.com"
        )

        self.assertIn("No canonical develop baseline", body)

    def test_creates_updates_and_removes_sticky_comment(self) -> None:
        HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, "warning")
        self.assertEqual(self.api.calls[0][0], "create")

        self.api.calls.clear()
        self.api.comments = [
            {
                "body": HELPER.COMMENT_MARKER,
                "id": 41,
                "user": {"login": "github-actions[bot]"},
            },
            {
                "body": HELPER.COMMENT_MARKER,
                "id": 42,
                "user": {"login": "github-actions[bot]"},
            },
        ]
        HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, "new warning")
        self.assertEqual([call[0] for call in self.api.calls], ["update", "delete"])

        self.api.calls.clear()
        HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, None)
        self.assertEqual([call[0] for call in self.api.calls], ["delete", "delete"])

    def test_ignores_comment_with_null_body(self) -> None:
        self.api.comments = [
            {
                "body": None,
                "id": 41,
                "user": {"login": "github-actions[bot]"},
            }
        ]

        HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, "warning")

        self.assertEqual([call[0] for call in self.api.calls], ["create"])

    def test_stale_head_cannot_create_update_or_delete_comment(self) -> None:
        existing = {
            "body": HELPER.COMMENT_MARKER,
            "id": 41,
            "user": {"login": "github-actions[bot]"},
        }
        for body, comments in (
            ("warning", []),
            ("new warning", [existing]),
            (None, [existing]),
        ):
            with self.subTest(body=body):
                self.api.calls.clear()
                self.api.comments = comments
                self.api.pull["head"]["sha"] = NEW_HEAD_SHA

                HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, body)

                self.assertEqual(self.api.calls, [])

    def test_rechecks_head_before_removing_duplicate_after_update(self) -> None:
        self.api.comments = [
            {
                "body": HELPER.COMMENT_MARKER,
                "id": 41,
                "user": {"login": "github-actions[bot]"},
            },
            {
                "body": HELPER.COMMENT_MARKER,
                "id": 42,
                "user": {"login": "github-actions[bot]"},
            },
        ]
        current_pull = self.api.pull
        stale_pull = {
            **self.api.pull,
            "head": {**self.api.pull["head"], "sha": NEW_HEAD_SHA},
        }
        with mock.patch.object(self.api, "get", side_effect=[current_pull, stale_pull]):
            HELPER.synchronize_comment(self.api, REPOSITORY, 7, HEAD_SHA, "new warning")

        self.assertEqual([call[0] for call in self.api.calls], ["update"])

    def test_comparison_timeout_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            baseline = directory / "baseline.json"
            candidate = directory / "candidate.json"
            baseline.write_text("{}", encoding="utf-8")
            candidate.write_text("{}", encoding="utf-8")
            timeout = HELPER.subprocess.TimeoutExpired(
                "compare", HELPER.COMPARISON_TIMEOUT_SECONDS
            )
            with mock.patch.object(
                HELPER.subprocess, "run", side_effect=timeout
            ) as run:
                with self.assertRaisesRegex(ValueError, "exceeded 60 seconds"):
                    HELPER.compare_candidate(
                        Path(__file__), baseline, candidate, baseline_found=True
                    )

            self.assertEqual(
                run.call_args.kwargs["timeout"], HELPER.COMPARISON_TIMEOUT_SECONDS
            )


if __name__ == "__main__":
    unittest.main()
