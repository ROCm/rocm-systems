# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for rocm_merge_gate.py.

The freeze is a merge gate, so these tests focus on the two properties that
matter: it must never pass a PR it cannot fully verify (fail closed), and the
freeze list in rocm-merge-gate.yml must mean what it looks like it means. The GitHub
API is stubbed, so nothing here touches the network.
"""

import io
import json
import os
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import rocm_merge_gate as gate

WORKFLOW = Path(__file__).parents[1].parent / "workflows" / "rocm-merge-gate.yml"
RCCL_FREEZE = [{"name": "rccl", "enabled": True, "paths": ["projects/rccl"]}]


def env(freezes, freeze_all=False):
    """Environment as the workflow hands it to the script (values are strings)."""
    raw = freezes if isinstance(freezes, str) else json.dumps(freezes)
    return {"FREEZES": raw, "FREEZE_ENTIRE_REPO": str(freeze_all).lower()}


def rest_stub(files, changed_files=None):
    """Stand in for the REST calls rocm_merge_gate makes for a single PR."""
    count = len(files) if changed_files is None else changed_files

    def fake_gh(*args, stdin=None):
        if args[0] == "--paginate":
            return json.dumps(files)
        return json.dumps({"changed_files": count})

    return fake_gh


def evaluate(files, freezes=RCCL_FREEZE, freeze_all=False, changed_files=None):
    with patch.dict(os.environ, env(freezes, freeze_all), clear=False):
        with patch.object(gate, "gh", rest_stub(files, changed_files)):
            return gate.evaluate("owner/repo", 1)


class TruthyTest(unittest.TestCase):
    def test_falsey_spellings(self):
        for value in ("false", "FALSE", " off ", "0", "no"):
            self.assertFalse(gate.truthy(value, default=True), value)

    def test_everything_else_is_true(self):
        for value in ("true", "on", "1", "yes"):
            self.assertTrue(gate.truthy(value, default=False), value)

    def test_unset_and_blank_use_the_default(self):
        for value in (None, "", "   "):
            self.assertTrue(gate.truthy(value, default=True), repr(value))
            self.assertFalse(gate.truthy(value, default=False), repr(value))


class LoadFreezesTest(unittest.TestCase):
    def load(self, freezes, freeze_all=False):
        with patch.dict(os.environ, env(freezes, freeze_all), clear=False):
            return gate.load_freezes()

    def test_entries_are_enabled_by_default(self):
        _, freezes = self.load([{"name": "rccl", "paths": ["projects/rccl"]}])
        self.assertEqual([f.name for f in freezes], ["rccl"])

    def test_disabled_entries_are_dropped(self):
        _, freezes = self.load(
            [
                {"name": "rccl", "enabled": False, "paths": ["projects/rccl"]},
                {"name": "shmem", "enabled": True, "paths": ["projects/rocshmem"]},
            ]
        )
        self.assertEqual([f.name for f in freezes], ["shmem"])

    def test_repo_wide_switch(self):
        self.assertFalse(self.load([])[0])
        self.assertTrue(self.load([], freeze_all=True)[0])

    def test_unnamed_entry_gets_a_placeholder_name(self):
        _, freezes = self.load([{"paths": ["projects/rccl"]}])
        self.assertEqual(freezes[0].name, "freeze-0")

    def test_missing_config_means_no_freezes(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gate.load_freezes(), (False, []))

    def test_unusable_config_raises(self):
        # A typo must not silently unfreeze the repository.
        for bad in (
            "{not json",
            '{"name": "rccl"}',
            '["rccl"]',
            '[{"name": "rccl", "paths": "projects/rccl"}]',
            '[{"name": "rccl", "paths": []}]',
            '[{"name": "rccl", "paths": [""]}]',
            '[{"name": "rccl"}]',
        ):
            with self.subTest(config=bad), self.assertRaises(gate.ConfigError):
                self.load(bad)


class MatchTest(unittest.TestCase):
    def match(self, pattern, path):
        return gate.Freeze(name="t", paths=(pattern,)).match(path)

    def test_directory_prefix_covers_the_tree(self):
        self.assertTrue(self.match("projects/rccl", "projects/rccl"))
        self.assertTrue(self.match("projects/rccl", "projects/rccl/src/a.cc"))
        self.assertTrue(self.match("projects/rccl/", "projects/rccl/src/a.cc"))

    def test_prefix_does_not_leak_into_sibling_directories(self):
        self.assertFalse(self.match("projects/rccl", "projects/rccl-tests/a"))
        self.assertFalse(self.match("projects/rccl", "projects/rccl-extra/a"))

    def test_glob_patterns(self):
        self.assertTrue(self.match("projects/*/docs/*", "projects/hip/docs/a.md"))
        self.assertFalse(self.match("projects/*/docs/*", "projects/hip/src/a.cc"))
        self.assertTrue(self.match("**/*.pem", "projects/rccl/key.pem"))


class VerdictTest(unittest.TestCase):
    def test_no_active_freeze_passes(self):
        allowed, summary = gate.verdict(["projects/rccl/a"], False, [])
        self.assertTrue(allowed)
        self.assertEqual(summary, "No freezes are active.")

    def test_blocked_summary_names_the_freeze_and_path(self):
        freezes = [gate.Freeze("rccl", ("projects/rccl",))]
        allowed, summary = gate.verdict(["projects/rccl/a"], False, freezes)
        self.assertFalse(allowed)
        self.assertIn("freeze `rccl`: `projects/rccl/a`", summary)
        self.assertIn("rocm-merge-gate.yml", summary)

    def test_unfrozen_paths_pass_and_name_the_active_freezes(self):
        freezes = [gate.Freeze("rccl", ("projects/rccl",))]
        allowed, summary = gate.verdict(["docs/x.md"], False, freezes)
        self.assertTrue(allowed)
        self.assertIn("active: rccl", summary)

    def test_repo_wide_freeze_blocks_any_path(self):
        allowed, summary = gate.verdict(["docs/x.md"], True, [])
        self.assertFalse(allowed)
        self.assertIn("repository-wide freeze: `docs/x.md`", summary)

    def test_overlapping_freezes_list_each_path_once_per_freeze(self):
        freezes = [gate.Freeze("rccl", ("projects/rccl",))]
        _, summary = gate.verdict(["projects/rccl/a", "projects/rccl/a"], True, freezes)
        self.assertEqual(
            summary.count("- repository-wide freeze: `projects/rccl/a`"), 1
        )
        self.assertEqual(summary.count("- freeze `rccl`: `projects/rccl/a`"), 1)


class ExemptPathTest(unittest.TestCase):
    def test_the_freeze_can_always_be_lifted(self):
        # Without this, FREEZE_ENTIRE_REPO would block the PR that turns it off.
        allowed, _ = evaluate(
            [{"filename": path} for path in gate.EXEMPT_PATHS],
            freezes=[],
            freeze_all=True,
        )
        self.assertTrue(allowed)

    def test_exempt_paths_do_not_excuse_the_rest_of_the_pr(self):
        allowed, _ = evaluate(
            [{"filename": gate.EXEMPT_PATHS[0]}, {"filename": "projects/rccl/a"}]
        )
        self.assertFalse(allowed)


class EvaluateTest(unittest.TestCase):
    def test_frozen_tree_is_blocked(self):
        self.assertFalse(evaluate([{"filename": "projects/rccl/src/a.cc"}])[0])

    def test_unrelated_change_passes(self):
        self.assertTrue(evaluate([{"filename": "docs/x.md"}])[0])

    def test_moving_a_file_out_of_a_frozen_tree_is_blocked(self):
        allowed, summary = evaluate(
            [{"filename": "docs/new.cc", "previous_filename": "projects/rccl/old.cc"}]
        )
        self.assertFalse(allowed)
        self.assertIn("projects/rccl/old.cc", summary)

    def test_inactive_freeze_skips_the_api_entirely(self):
        def explode(*args, stdin=None):
            raise AssertionError("no API call expected when nothing is frozen")

        with patch.dict(os.environ, env([]), clear=False):
            with patch.object(gate, "gh", explode):
                self.assertEqual(
                    gate.evaluate("owner/repo", 1), (True, "No freezes are active.")
                )


class FailClosedTest(unittest.TestCase):
    """Anything the check cannot verify has to fail, never pass."""

    def test_api_error(self):
        def broken(*args, stdin=None):
            raise RuntimeError("gh api ... failed: 503")

        with patch.dict(os.environ, env(RCCL_FREEZE), clear=False):
            with patch.object(gate, "gh", broken):
                with self.assertRaises(RuntimeError):
                    gate.evaluate("owner/repo", 1)

    def test_missing_changed_files_metadata(self):
        def no_metadata(*args, stdin=None):
            return json.dumps([]) if args[0] == "--paginate" else json.dumps({})

        with patch.dict(os.environ, env(RCCL_FREEZE), clear=False):
            with patch.object(gate, "gh", no_metadata):
                with self.assertRaisesRegex(RuntimeError, "changed_files"):
                    gate.evaluate("owner/repo", 1)

    def test_pr_larger_than_the_rest_endpoint_can_list(self):
        with self.assertRaisesRegex(RuntimeError, "at most 3000"):
            evaluate(
                [{"filename": "docs/x.md"}], changed_files=gate.MAX_LISTED_FILES + 1
            )

    def test_truncated_file_list(self):
        with self.assertRaisesRegex(RuntimeError, "Listed 1 files but PR reports 2"):
            evaluate([{"filename": "docs/x.md"}], changed_files=2)

    def test_cli_exit_code(self):
        for path, expected in (("projects/rccl/a", 1), ("docs/x.md", 0)):
            with self.subTest(path=path):
                with patch.dict(os.environ, env(RCCL_FREEZE), clear=False):
                    with patch.object(gate, "gh", rest_stub([{"filename": path}])):
                        with patch.object(gate, "report", lambda _text: None):
                            # cmd_evaluate emits a ::error:: annotation on failure;
                            # keep it out of this job's own log.
                            with redirect_stdout(io.StringIO()):
                                exit_code = gate.cmd_evaluate("owner/repo", 1)
                self.assertEqual(exit_code, expected)


class PaginationTest(unittest.TestCase):
    def test_pages_are_flattened(self):
        pages = [
            # A patch body containing "][" would break naive string splicing.
            [{"filename": "a", "patch": "weird ][ body"}],
            [{"filename": "projects/rccl/b"}],
        ]
        payload = "\n".join(json.dumps(page) for page in pages)
        with patch.object(gate, "gh", lambda *a, stdin=None: payload):
            self.assertEqual(
                [entry["filename"] for entry in gate.gh_paginate("x")],
                ["a", "projects/rccl/b"],
            )


class SweepTest(unittest.TestCase):
    """The sweep re-posts the check so idle PRs follow the current config."""

    TOTAL = 120
    RENAMED_PR = 7
    BIG_PR = 9

    def setUp(self):
        self.nodes = []
        for number in range(1, self.TOTAL + 1):
            if number == self.RENAMED_PR:
                files, count, truncated = (
                    [self.file("docs/new.md", "RENAMED")],
                    1,
                    False,
                )
            elif number == self.BIG_PR:
                files = [self.file(f"docs/f{n}.md") for n in range(gate.FILE_PAGE_SIZE)]
                count, truncated = 250, True
            elif number % 3 == 0:
                files, count, truncated = [self.file("projects/rccl/a.cc")], 1, False
            else:
                files, count, truncated = [self.file("docs/x.md")], 1, False
            self.nodes.append(
                {
                    "number": number,
                    "headRefOid": f"{number:040d}",
                    "changedFiles": count,
                    "files": {
                        "totalCount": count,
                        "pageInfo": {"hasNextPage": truncated},
                        "nodes": files,
                    },
                }
            )
        self.calls = {"graphql": 0, "rest": 0, "posted": 0}
        self.posted = {}
        self.reports = []

    @staticmethod
    def file(path, change_type="MODIFIED"):
        return {"path": path, "changeType": change_type}

    def fake_gh(self, *args, stdin=None):
        if args[0] == "graphql":
            self.calls["graphql"] += 1
            cursor = json.loads(stdin)["variables"]["cursor"]
            start = 0 if cursor is None else int(cursor)
            page = self.nodes[start : start + gate.PR_PAGE_SIZE]
            end = start + len(page)
            return json.dumps(
                {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {
                                    "hasNextPage": end < len(self.nodes),
                                    "endCursor": str(end),
                                },
                                "nodes": page,
                            }
                        }
                    }
                }
            )
        if args[0] == "--method":
            self.calls["posted"] += 1
            body = json.loads(stdin)
            self.posted[body["head_sha"]] = body["conclusion"]
            return "{}"

        # REST fallback for a single PR.
        self.calls["rest"] += 1
        number = int(re.search(r"/pulls/(\d+)", args[-1]).group(1))
        if args[0] == "--paginate":
            if number == self.RENAMED_PR:
                return json.dumps(
                    [
                        {
                            "filename": "docs/new.md",
                            "previous_filename": "projects/rccl/gone.cc",
                        }
                    ]
                )
            return json.dumps([{"filename": f"docs/f{n}.md"} for n in range(250)])
        return json.dumps({"changed_files": self.nodes[number - 1]["changedFiles"]})

    def sweep(self, freezes=RCCL_FREEZE, freeze_all=False):
        with patch.dict(os.environ, env(freezes, freeze_all), clear=False):
            with patch.object(gate, "gh", self.fake_gh):
                with patch.object(gate, "report", self.reports.append):
                    return gate.cmd_reevaluate_open("owner/repo", "develop")

    def failures(self):
        return {
            int(sha) for sha, verdict in self.posted.items() if verdict == "failure"
        }

    def test_graphql_batches_the_reads(self):
        self.assertEqual(self.sweep(), 0)
        pages = -(-self.TOTAL // gate.PR_PAGE_SIZE)
        self.assertEqual(self.calls["graphql"], pages)
        self.assertEqual(self.calls["posted"], self.TOTAL)
        # Only the rename and the >100-file PR need per-PR REST reads (2 each).
        self.assertEqual(self.calls["rest"], 4)
        self.assertIn(f"Re-checked {self.TOTAL} open `develop` PR(s)", self.reports[0])

    def test_every_open_pr_gets_a_verdict(self):
        self.sweep()
        expected = {n for n in range(1, self.TOTAL + 1) if n % 3 == 0}
        expected.discard(self.BIG_PR)  # its 250 files are all outside the freeze
        expected.add(self.RENAMED_PR)  # only the pre-rename path is frozen
        self.assertEqual(self.failures(), expected)

    def test_rename_is_caught_by_the_rest_fallback(self):
        # GraphQL never returns the pre-rename path, so the fallback is load-bearing.
        self.sweep()
        self.assertIn(self.RENAMED_PR, self.failures())

    def test_lifting_a_freeze_clears_every_stale_failure(self):
        self.sweep(
            freezes=[{"name": "rccl", "enabled": False, "paths": ["projects/rccl"]}]
        )
        self.assertEqual(self.calls["posted"], self.TOTAL)
        self.assertEqual(self.failures(), set())
        self.assertEqual(self.calls["rest"], 0)

    def test_repo_wide_freeze_blocks_everything(self):
        self.sweep(freezes=[], freeze_all=True)
        self.assertEqual(len(self.failures()), self.TOTAL)

    def test_a_broken_pr_evaluation_posts_a_failure(self):
        original = gate.evaluate

        def flaky(repo, number):
            if number == self.RENAMED_PR:
                raise RuntimeError("boom")
            return original(repo, number)

        with patch.object(gate, "evaluate", flaky):
            self.sweep()
        self.assertIn(self.RENAMED_PR, self.failures())

    def test_sweep_survives_a_failed_check_post(self):
        real_gh = self.fake_gh

        def refuse_one_post(*args, stdin=None):
            if args[0] == "--method" and json.loads(stdin)["head_sha"] == f"{1:040d}":
                raise RuntimeError("check-run POST rejected")
            return real_gh(*args, stdin=stdin)

        with patch.dict(os.environ, env(RCCL_FREEZE), clear=False):
            with patch.object(gate, "gh", refuse_one_post):
                with patch.object(gate, "report", self.reports.append):
                    self.assertEqual(
                        gate.cmd_reevaluate_open("owner/repo", "develop"), 0
                    )
        self.assertEqual(self.calls["posted"], self.TOTAL - 1)
        self.assertIn("#1: could not post check", self.reports[0])


class OpenPRsTest(unittest.TestCase):
    def page(self, nodes, has_next=False, cursor="c"):
        return json.dumps(
            {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                            "nodes": nodes,
                        }
                    }
                }
            }
        )

    def node(self, number, files, changed_files=None, has_next=False):
        return {
            "number": number,
            "headRefOid": f"{number:040d}",
            "changedFiles": len(files) if changed_files is None else changed_files,
            "files": {
                "totalCount": len(files),
                "pageInfo": {"hasNextPage": has_next},
                "nodes": files,
            },
        }

    def collect(self, payloads):
        with patch.object(gate, "gh", lambda *a, stdin=None: payloads.pop(0)):
            return list(gate.open_prs("owner/repo", "develop"))

    def test_paginates_until_exhausted(self):
        payloads = [
            self.page(
                [self.node(1, [{"path": "a", "changeType": "MODIFIED"}])], has_next=True
            ),
            self.page([self.node(2, [{"path": "b", "changeType": "MODIFIED"}])]),
        ]
        self.assertEqual([pull.number for pull in self.collect(payloads)], [1, 2])

    def test_flags_prs_graphql_cannot_answer(self):
        modified = {"path": "a", "changeType": "MODIFIED"}
        payloads = [
            self.page(
                [
                    self.node(1, [modified]),
                    self.node(2, [{"path": "a", "changeType": "RENAMED"}]),
                    self.node(3, [modified], has_next=True),
                    self.node(4, [modified], changed_files=9),
                ]
            )
        ]
        self.assertEqual(
            {pull.number: pull.needs_rest for pull in self.collect(payloads)},
            {1: False, 2: True, 3: True, 4: True},
        )

    def test_exempt_paths_are_dropped_from_batched_results(self):
        node = self.node(1, [{"path": gate.EXEMPT_PATHS[0], "changeType": "MODIFIED"}])
        self.assertEqual(self.collect([self.page([node])])[0].paths, ())

    def test_graphql_errors_fail_closed(self):
        payload = json.dumps({"errors": [{"message": "bad query"}]})
        with patch.object(gate, "gh", lambda *a, stdin=None: payload):
            with self.assertRaisesRegex(RuntimeError, "GraphQL query failed"):
                list(gate.open_prs("owner/repo", "develop"))


class ValidateConfigTest(unittest.TestCase):
    """`rocm_merge_gate.py validate` is the guard against a bad freeze list landing."""

    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def workflow(self, freezes, freeze_all="false"):
        raw = freezes if isinstance(freezes, str) else json.dumps(freezes)
        path = self.root / f"wf{len(list(self.root.iterdir()))}.yml"
        path.write_text(
            yaml.safe_dump({"env": {"FREEZE_ENTIRE_REPO": freeze_all, "FREEZES": raw}})
        )
        return path

    def problems(self, freezes, freeze_all="false", tracked=(), existing=()):
        """Validate against a stubbed repository containing only `existing`."""
        known = set(existing) | set(gate.EXEMPT_PATHS)
        with patch.object(gate, "path_in_repo", lambda _root, path: path in known):
            with patch.object(gate, "tracked_paths", lambda _root: list(tracked)):
                return gate.validate_config(
                    self.workflow(freezes, freeze_all), self.root
                )

    def test_clean_config_has_no_problems(self):
        self.assertEqual(
            self.problems(RCCL_FREEZE, existing=["projects/rccl"]),
            [],
        )

    def test_invalid_json_is_reported(self):
        problems = self.problems('[{"name": "x", "paths": ["projects/rccl"],}]')
        self.assertTrue(any("not valid JSON" in p for p in problems), problems)

    def test_path_that_does_not_exist_is_reported(self):
        problems = self.problems(
            [{"name": "typo", "paths": ["projects/rcl"]}], existing=["projects/rccl"]
        )
        self.assertIn(
            "Freeze `typo` path `projects/rcl` is not in the repository.", problems
        )

    def test_glob_matching_nothing_is_reported(self):
        problems = self.problems(
            [{"name": "g", "paths": ["projects/*/nope/*"]}],
            tracked=["projects/rccl/src/a.cc"],
        )
        self.assertIn(
            "Freeze `g` pattern `projects/*/nope/*` matches nothing.", problems
        )

    def test_glob_that_matches_is_accepted(self):
        self.assertEqual(
            self.problems(
                [{"name": "g", "paths": ["projects/*/docs/*"]}],
                tracked=["projects/hip/docs/a.md"],
            ),
            [],
        )

    def test_disabled_entries_are_validated_too(self):
        # A disabled freeze with a bad path is a trap for whoever re-enables it.
        problems = self.problems(
            [{"name": "off", "enabled": False, "paths": ["projects/gone"]}]
        )
        self.assertIn(
            "Freeze `off` path `projects/gone` is not in the repository.", problems
        )

    def test_absolute_and_parent_paths_are_reported(self):
        for path in ("/projects/rccl", "../projects/rccl"):
            with self.subTest(path=path):
                problems = self.problems([{"name": "p", "paths": [path]}])
                self.assertIn(
                    f"Freeze `p` path `{path}` must be repo-relative.", problems
                )

    def test_duplicate_names_are_reported(self):
        problems = self.problems(
            [
                {"name": "dup", "paths": ["projects/rccl"]},
                {"name": "dup", "paths": ["projects/rccl"]},
            ],
            existing=["projects/rccl"],
        )
        self.assertIn("Duplicate freeze names: dup.", problems)

    def test_unrecognized_switch_values_are_reported(self):
        problems = self.problems(
            RCCL_FREEZE, freeze_all="maybe", existing=["projects/rccl"]
        )
        self.assertTrue(any("FREEZE_ENTIRE_REPO must be one of" in p for p in problems))

        problems = self.problems(
            [{"name": "x", "enabled": "flase", "paths": ["projects/rccl"]}],
            existing=["projects/rccl"],
        )
        self.assertTrue(any("has enabled='flase'" in p for p in problems), problems)

    def test_missing_exempt_path_is_reported(self):
        with patch.object(gate, "path_in_repo", lambda _root, _path: False):
            problems = gate.validate_config(self.workflow([]), self.root)
        for path in gate.EXEMPT_PATHS:
            self.assertTrue(any(path in p for p in problems), problems)

    def test_exit_codes(self):
        clean = self.workflow(RCCL_FREEZE)
        broken = self.workflow([{"name": "x", "paths": ["projects/gone"]}])
        with patch.object(gate, "report", lambda _text: None):
            with redirect_stdout(io.StringIO()):
                with patch.object(
                    gate, "path_in_repo", lambda _root, path: path != "projects/gone"
                ):
                    self.assertEqual(gate.cmd_validate(clean, self.root), 0)
                    self.assertEqual(gate.cmd_validate(broken, self.root), 1)


class RealRepoValidationTest(unittest.TestCase):
    """The same lint the CI step runs, against the checked-out repository."""

    def test_shipped_config_paths_exist(self):
        repo_root = WORKFLOW.parents[2]
        self.assertEqual(gate.validate_config(WORKFLOW, repo_root), [])


class ShippedConfigTest(unittest.TestCase):
    """Guard the freeze list that rocm-merge-gate.yml actually ships."""

    @classmethod
    def setUpClass(cls):
        cls.workflow_env = yaml.safe_load(WORKFLOW.read_text())["env"]
        cls.env = {
            "FREEZES": cls.workflow_env["FREEZES"],
            "FREEZE_ENTIRE_REPO": str(cls.workflow_env["FREEZE_ENTIRE_REPO"]).lower(),
        }

    def verdict_for(self, path):
        with patch.dict(os.environ, self.env, clear=False):
            with patch.object(gate, "gh", rest_stub([{"filename": path}])):
                return gate.evaluate("owner/repo", 1)[0]

    def test_config_is_parseable(self):
        with patch.dict(os.environ, self.env, clear=False):
            gate.load_freezes()

    def test_repo_is_not_globally_frozen_by_default(self):
        self.assertFalse(gate.truthy(self.env["FREEZE_ENTIRE_REPO"], default=False))

    def test_frozen_trees(self):
        for path in ("projects/rccl/src/a.cc", "projects/rccl-tests/a.cc"):
            self.assertFalse(self.verdict_for(path), path)

    def test_unfrozen_paths(self):
        for path in ("docs/x.md", "projects/rocshmem/a.cc", "projects/rccl-extra/a.cc"):
            self.assertTrue(self.verdict_for(path), path)

    def test_workflow_runs_on_every_develop_pr(self):
        # A path filter here would leave unrelated PRs waiting on a required
        # check that never reports.
        triggers = yaml.safe_load(WORKFLOW.read_text())[True]
        self.assertEqual(triggers["pull_request_target"], {"branches": ["develop"]})

    def test_exempt_paths_match_the_shipped_filenames(self):
        repo_root = WORKFLOW.parents[2]
        for path in gate.EXEMPT_PATHS:
            self.assertTrue((repo_root / path).is_file(), path)


if __name__ == "__main__":
    unittest.main()
