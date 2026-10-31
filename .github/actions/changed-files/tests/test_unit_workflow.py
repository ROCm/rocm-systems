# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Verify scope selection and event wiring for the generic unit-test matrix."""

from pathlib import Path
import sys
import unittest

import yaml

ACTION_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ACTION_DIRECTORY))

from scope import matches_scope


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = yaml.load(
            (ACTION_DIRECTORY.parents[1] / "workflows/unit-tests.yml").read_text(),
            Loader=yaml.BaseLoader,
        )
        self.job = self.workflow["jobs"]["test"]
        self.scopes = self.job["strategy"]["matrix"]["include"]
        self.inspector = next(
            step for step in self.job["steps"] if step.get("id") == "scope"
        )

    def test_all_suites_inspect_every_pr_and_develop_push(self) -> None:
        self.assertEqual(self.workflow["on"]["push"], {"branches": ["develop"]})
        self.assertEqual(
            self.workflow["on"]["pull_request"],
            {
                "types": [
                    "opened",
                    "synchronize",
                    "reopened",
                    "ready_for_review",
                    "edited",
                ]
            },
        )
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        self.assertNotIn("if", self.inspector)
        self.assertEqual(self.job["strategy"]["fail-fast"], "false")
        self.assertEqual(self.inspector["uses"], "./.github/actions/changed-files")
        self.assertEqual(self.inspector["with"]["include"], "${{ matrix.include }}")
        self.assertEqual(
            len(self.scopes), len({scope["suite"] for scope in self.scopes})
        )

    def test_shared_changes_activate_every_suite_and_docs_activate_none(self) -> None:
        for scope in self.scopes:
            for path, expected in (
                (".github/actions/changed-files/scope.py", True),
                (".github/workflows/unit-tests.yml", True),
                ("docs/readme.md", False),
            ):
                with self.subTest(suite=scope["suite"], path=path):
                    self.assertEqual(
                        matches_scope(
                            [path],
                            scope["include"],
                            "",
                            self.inspector["with"]["always-include"],
                        ),
                        expected,
                    )

    def test_unaffected_suites_skip_setup_and_execution(self) -> None:
        steps = self.job["steps"]
        self.assertEqual(steps[0]["with"]["fetch-depth"], "0")
        self.assertNotIn("if", steps[0])
        for step in steps[2:]:
            self.assertTrue(
                step["if"].startswith("steps.scope.outputs.affected == 'true'")
            )
            self.assertNotIn("continue-on-error", step)
        self.assertEqual(steps[-1]["run"], "${{ matrix.run }}")

    def test_selector_compares_pinned_event_commits(self) -> None:
        action = yaml.load(
            (ACTION_DIRECTORY / "action.yml").read_text(), Loader=yaml.BaseLoader
        )
        environment = action["runs"]["steps"][0]["env"]
        self.assertEqual(
            environment["BASE_SHA"],
            "${{ github.event.pull_request.base.sha || github.event.before }}",
        )
        self.assertEqual(
            environment["HEAD_SHA"],
            "${{ github.event.pull_request.head.sha || github.sha }}",
        )


if __name__ == "__main__":
    unittest.main()
