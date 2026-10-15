"""
Unit + integration tests for policy_check.py.

These let us iterate on policies WITHOUT pushing branches or running workflows:
  • Unit tests   — exercise individual validators / regex patterns.
  • Integration  — feed blobs of [branch, title, description, files] through
                   the higher-level ensure_* functions.

Run locally:
    python tools/systems_pr_bot/test_policy_check_ut.py -v
    # or
    pytest tools/systems_pr_bot/test_policy_check_ut.py
"""

import io
import os
import re
import sys
import unittest
import urllib.parse
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest import mock

# Make `policy_check` importable regardless of the working directory.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import policy_check as pc  # noqa: E402

# ----------------------------- helpers ---------------------------------------

_ISSUE_PATTERNS = [
    r"(?im)^\s*JIRA\s*ID\s*[:\-]?\s*(#?\d+|[A-Z][A-Z0-9]+-\d+|https?:\/\/\S+)",
    r"(?im)^\s*ISSUE\s*ID\s*[:\-]?\s*(#?\d+|[A-Z][A-Z0-9]+-\d+|https?:\/\/\S+)",
    # JIRA/ISSUE ID on a separate line (blank lines + trailing spaces allowed).
    r"(?im)^[ \t]*JIRA[ \t]+ID[ \t]*\r?\n[ \t\r\n]*([A-Z][A-Z0-9]+-\d+)",
    r"(?im)^[ \t]*ISSUE[ \t]+ID[ \t]*\r?\n[ \t\r\n]*([A-Z][A-Z0-9]+-\d+|\d+)",
    r"(?im)\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b\s*:?\s*"
    r"(?:[A-Za-z0-9._\-]+\/[A-Za-z0-9._\-]+)?#\d+",
    # Bare GitHub issue reference, e.g. #123
    r"(?m)(?:^|\s)#\d+\b",
    # GitHub issue URL
    r"(?i)https?:\/\/github\.com\/[^\/\s]+\/[^\/\s]+\/issues\/\d+",
]

_CHECKLIST_PATTERNS = [
    r"(?im)^\s*-\s*\[[xX]\]\s*.*contributing guidelines",
]


def make_policy(**overrides: Any) -> pc.Policy:
    """Build a Policy with sensible defaults; override any field per-test.

    Independent of policy.yml so regex/validator behaviour can be pinned even
    if the shipped config changes. Note: title and branch-name policies have
    been removed — Policy no longer carries any title/branch fields.
    """
    defaults: Dict[str, Any] = dict(
        description_min_length=30,
        description_issue_patterns=[re.compile(p) for p in _ISSUE_PATTERNS],
        description_checklist_patterns=[re.compile(p) for p in _CHECKLIST_PATTERNS],
        block_draft=True,
        forbidden_paths=["**/*.pem", "**/.env", "**/id_rsa"],
        unit_test_code_extensions=[".py", ".cpp"],
        unit_test_patterns=[
            "test_*",
            "testing_*",
            "*_test.*",
            "*_tests.*",
            "*_gtest.*",
            "Test*",
            "**/test/gtest/**",
        ],
        unit_test_exempt_paths=[],
        bump_bot_authors=["assistant-librarian", "systems-assistant", "dependabot"],
        required_checks=[pc.RequiredCheck("pre-commit", [])],
        precommit_failure_comment=None,
    )
    defaults.update(overrides)
    return pc.Policy(**defaults)


def make_file(
    filename: str,
    status: str = "modified",
    additions: int = 0,
    deletions: int = 0,
    changes: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "status": status,
        "additions": additions,
        "deletions": deletions,
        "changes": changes if changes is not None else additions + deletions,
    }


def make_check_run(
    conclusion: Optional[str], name: str = "pre-commit"
) -> Dict[str, Any]:
    return {"name": name, "conclusion": conclusion}


def make_workflow_run(**overrides: Any) -> Dict[str, Any]:
    run = {
        "id": 100,
        "run_attempt": 1,
        "event": "pull_request",
        "head_sha": "sha",
        "display_title": "PR #7: develop@base-sha",
        "status": "completed",
        "conclusion": "success",
        # GitHub can omit associations for fork PRs; identity comes from the
        # event-pinned run title and head SHA, not this mutable association.
        "pull_requests": [],
    }
    run.update(overrides)
    run.setdefault("run_number", run["id"])
    run.setdefault("check_suite_id", run["id"] * 10 if type(run["id"]) is int else None)
    return run


# ----------------------------- PR description --------------------------------


class DescriptionTests(unittest.TestCase):
    def test_too_short(self) -> None:
        policy = make_policy()
        e: List[str] = []
        pc.ensure_pr_description(policy, "short", e)
        self.assertTrue(any("too short" in x for x in e))

    def test_missing_issue_reference(self) -> None:
        # No checklist patterns so only the reference check fires.
        policy = make_policy(description_checklist_patterns=[])
        e: List[str] = []
        pc.ensure_pr_description(policy, "A long enough description with no ref.", e)
        self.assertTrue(any("must reference a JIRA ID" in x for x in e))

    def test_issue_reference_in_comment_does_not_pass(self) -> None:
        # Isolate reference detection (skip min-length and checklist).
        policy = make_policy(
            description_min_length=0, description_checklist_patterns=[]
        )
        multiline_comment = """<!--
Fixes #1234
-->"""
        multiple_comments = """This description has no visible issue reference.
<!-- Related to #1234 -->
Some visible text between the comments.
<!-- https://github.com/ROCm/TheRock/issues/5678 -->"""
        for body in [
            "<!-- GitHub issue: https://github.com/ROCm/TheRock/issues/1234 -->",
            multiline_comment,
            multiple_comments,
        ]:
            with self.subTest(body=body):
                e: List[str] = []
                pc.ensure_pr_description(policy, body, e)
                self.assertTrue(any("must reference a JIRA ID" in x for x in e))

    def test_issue_reference_variants_pass(self) -> None:
        # Isolate reference detection (skip min-length and checklist).
        policy = make_policy(
            description_min_length=0, description_checklist_patterns=[]
        )
        for body in [
            "JIRA ID : TESTAUTO-6039",
            "JIRA ID - #330",
            "JIRA ID #330",
            "ISSUE ID : TESTUTO-3334",
            "ISSUE ID - TESTAUTO-3433",
            "ISSUE ID : https://github.com/org/repo/issues/1234",
            # Multiline format with JIRA ID
            "JIRA ID\nROCM-25757",
            "JIRA ID\n\nROCM-25757",
            "jira id\nROCM-25757",  # case-insensitive
            # Trailing spaces after label + blank line before key
            "JIRA ID  \n\nAIRUNTIME-2352",
            "JIRA ID\t\n\n\nROCM-25757",
            "JIRA ID  \r\n\r\nROCM-25757",  # CRLF line endings
            # Multiline format with ISSUE ID
            "ISSUE ID\nAIRUNTIME-2352",
            "ISSUE ID\n\nAIRUNTIME-2352",
            "issue id\nAIRUNTIME-2352",  # case-insensitive
            "ISSUE ID  \n\nAIRUNTIME-2352",  # trailing spaces + blank line
        ]:
            with self.subTest(body=body):
                e: List[str] = []
                pc.ensure_pr_description(policy, body, e)
                self.assertEqual(e, [])

    def test_closing_keyword_variants_pass(self) -> None:
        # GitHub closing keywords are also accepted as a tracking ref.
        policy = make_policy(
            description_min_length=0, description_checklist_patterns=[]
        )
        for body in [
            "Closes #10",
            "Fixes octo-org/octo-repo#100",
            "Resolves #10",
            "resolves #123",
            "resolves octo-org/octo-repo#100",
            "Closes: #10",
            "CLOSES #10",
            "CLOSES: #10",
            "This change fixes the bug.\nFixes #4321\n",
        ]:
            with self.subTest(body=body):
                e: List[str] = []
                pc.ensure_pr_description(policy, body, e)
                self.assertEqual(e, [])

    def test_plain_github_issue_refs_pass(self) -> None:
        # Bare '#<number>' and GitHub issue URLs are accepted without a keyword.
        policy = make_policy(
            description_min_length=0, description_checklist_patterns=[]
        )
        for body in [
            "Related to #123",
            "#4321",
            "See https://github.com/ROCm/TheRock/issues/6043",
        ]:
            with self.subTest(body=body):
                e: List[str] = []
                pc.ensure_pr_description(policy, body, e)
                self.assertEqual(e, [])

    def test_reference_inside_larger_body(self) -> None:
        policy = make_policy(description_checklist_patterns=[])
        body = "This change fixes the parser.\n\nISSUE ID : TESTUTO-3334\n"
        e: List[str] = []
        pc.ensure_pr_description(policy, body, e)
        self.assertEqual(e, [])

    def test_checklist_ticked_passes(self) -> None:
        policy = make_policy(description_min_length=0, description_issue_patterns=[])
        body = "- [x] Look over the contributing guidelines at https://..."
        e: List[str] = []
        pc.ensure_pr_description(policy, body, e)
        self.assertEqual(e, [])

    def test_checklist_unticked_fails(self) -> None:
        policy = make_policy(description_min_length=0, description_issue_patterns=[])
        body = "- [ ] Look over the contributing guidelines at https://..."
        e: List[str] = []
        pc.ensure_pr_description(policy, body, e)
        self.assertTrue(any("Checklist" in x or "checklist" in x for x in e))


# ----------------------------- forbidden files -------------------------------


class ForbiddenFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = make_policy()

    def _errs(self, files: List[Dict[str, Any]]) -> List[str]:
        e: List[str] = []
        pc.ensure_no_forbidden_files(self.policy, files, e)
        return e

    def test_flags_secret_files(self) -> None:
        for name in ["secret.pem", "config/.env", "deploy/id_rsa"]:
            with self.subTest(name=name):
                self.assertTrue(self._errs([make_file(name)]))

    def test_allows_normal_files(self) -> None:
        files = [make_file("src/app.py"), make_file("README.md")]
        self.assertEqual(self._errs(files), [])

    def test_removed_forbidden_file_is_ignored(self) -> None:
        self.assertEqual(self._errs([make_file("secret.pem", status="removed")]), [])

    def test_forbidden_files_is_warning_only_row(self) -> None:
        # The Forbidden Files row is warning-only: passed=True + warn=True when a
        # forbidden file is present, so it never blocks the workflow.
        result = pc.CheckResult(
            "Forbidden Files",
            "⛔",
            passed=True,
            details=["Forbidden file present in PR: `secret.pem`"],
            warn=True,
        )
        marker = "<!-- test -->"
        body = pc.build_policy_table_comment([result], marker, ready=True)
        self.assertIn("⚠️ Warning", body)
        self.assertIn("secret.pem", body)
        # Forbidden Files is warning-only — it never adds the label.


# ----------------------------- unit tests check ------------------------------


class UnitTestRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = make_policy()

    def _errs(self, files: List[Dict[str, Any]]) -> List[str]:
        e: List[str] = []
        pc.ensure_unit_tests(self.policy, files, e)
        return e

    def test_code_without_test_fails(self) -> None:
        self.assertTrue(self._errs([make_file("src/module.py")]))

    def test_code_with_test_passes(self) -> None:
        files = [make_file("src/module.py"), make_file("tests/test_module.py")]
        self.assertEqual(self._errs(files), [])

    def test_docs_only_passes(self) -> None:
        files = [make_file("README.md"), make_file("config/settings.yml")]
        self.assertEqual(self._errs(files), [])

    def test_source_anywhere_requires_test(self) -> None:
        # Unit tests are required for source code placed ANYWHERE in the repo —
        # no folder is special. Each non-test source file, on its own, fails.
        for src in [
            "policy_check.py",
            "src/app.py",
            "deep/nested/dir/module.py",
            ".github/therock_pr_bot/policy_check.py",
            "lib/foo.cpp",
            # 'test.py' is NOT a test file — 'test_*' needs the 'test_' prefix.
            "test.py",
        ]:
            with self.subTest(src=src):
                self.assertTrue(self._errs([make_file(src)]))

    def test_test_file_anywhere_satisfies_requirement(self) -> None:
        # A real test_* file in ANY folder satisfies the requirement.
        for test_path in [
            "tests/test_module.py",
            "deep/nested/test_module.py",
            "any/where/module_test.py",
        ]:
            with self.subTest(test_path=test_path):
                files = [make_file("src/module.py"), make_file(test_path)]
                self.assertEqual(self._errs(files), [])

    def test_test_prefix_capitalized_satisfies_requirement(self) -> None:
        # The 'Test*' pattern recognises capitalised test files (e.g.
        # TestUtils.cpp, TestParser.py) as valid test files.
        for test_path in [
            "TestUtils.cpp",
            "tests/TestParser.py",
            "deep/nested/TestFeature.cpp",
        ]:
            with self.subTest(test_path=test_path):
                files = [make_file("src/module.py"), make_file(test_path)]
                self.assertEqual(self._errs(files), [])

    def test_path_based_pattern_satisfies_requirement(self) -> None:
        # Patterns containing '/' are matched against the full file path, not
        # just the basename. This allows entire test directories to be
        # recognised as test locations even if their files use no special naming
        # convention (e.g. hip-tests files named after the API they test:
        # atomicAdd.cc, acquire_release.cc).
        policy = make_policy(
            unit_test_patterns=[
                "test_*",
                "*_test.*",
                "**/test/gtest/**",
                "projects/hip-tests/**",
            ]
        )
        errs: List[str] = []

        # A .cpp file under projects/hip-tests/ satisfies the requirement even
        # though its basename ('atomicAdd.cpp') matches no name-based pattern.
        files = [
            make_file("projects/clr/hipamd/src/hip_memory.cpp"),
            make_file("projects/hip-tests/catch/unit/memory/atomicAdd.cpp"),
        ]
        pc.ensure_unit_tests(policy, files, errs)
        self.assertEqual(errs, [])

    def test_path_based_pattern_code_only_fails(self) -> None:
        # A path-based pattern only helps when the PR actually touches a file
        # under that path. Source changes with no matching test path still fail.
        policy = make_policy(
            unit_test_patterns=[
                "test_*",
                "*_test.*",
                "projects/hip-tests/**",
            ]
        )
        errs: List[str] = []
        files = [make_file("projects/clr/hipamd/src/hip_memory.cpp")]
        pc.ensure_unit_tests(policy, files, errs)
        self.assertTrue(errs)

    def test_unit_test_is_warning_only_row(self) -> None:
        # The Unit Test row is warning-only: passed=True + warn=True when a
        # code file has no accompanying test, so it never blocks the workflow.
        result = pc.CheckResult(
            "Unit Test", "🧪", passed=True, details=["missing test"], warn=True
        )
        marker = "<!-- test -->"
        body = pc.build_policy_table_comment([result], marker, ready=True)
        self.assertIn("⚠️ Warning", body)
        self.assertIn("missing test", body)


# ----------------------------- draft + bump ----------------------------------


class DraftAndBumpTests(unittest.TestCase):
    def test_draft_blocked_when_enabled(self) -> None:
        policy = make_policy(block_draft=True)
        e: List[str] = []
        pc.ensure_pr_not_draft(policy, True, e)
        self.assertTrue(e)

    def test_draft_allowed_when_not_draft(self) -> None:
        policy = make_policy(block_draft=True)
        e: List[str] = []
        pc.ensure_pr_not_draft(policy, False, e)
        self.assertEqual(e, [])

    def test_bump_author_detection(self) -> None:
        policy = make_policy()
        self.assertTrue(pc.is_bump_pr(policy, "assistant-librarian"))
        self.assertTrue(pc.is_bump_pr(policy, "assistant-librarian[bot]"))
        self.assertTrue(pc.is_bump_pr(policy, "SYSTEMS-ASSISTANT"))
        self.assertTrue(pc.is_bump_pr(policy, "dependabot"))
        self.assertTrue(pc.is_bump_pr(policy, "dependabot[bot]"))
        self.assertFalse(pc.is_bump_pr(policy, "some-human"))
        self.assertFalse(pc.is_bump_pr(policy, ""))


# ----------------------------- skip tag --------------------------------------


class SkipTagTests(unittest.TestCase):
    def test_skip_tag_detected(self) -> None:
        for body in [
            "@skip-pr-bot",
            "Please skip this one @skip-pr-bot thanks",
            "line one\n@SKIP-PR-BOT\nline three",  # case-insensitive
            "Skipping: @Skip-PR-Bot",
        ]:
            with self.subTest(body=body):
                self.assertTrue(pc.pr_wants_skip(body))

    def test_skip_tag_absent(self) -> None:
        for body in [
            "",
            "A normal description with a JIRA ID : ABC-1",
            "email me at skip-pr-bot@example.com",  # not the @-prefixed tag
            "@skip-pr-bottling",  # not a whole-word match
        ]:
            with self.subTest(body=body):
                self.assertFalse(pc.pr_wants_skip(body))

    def test_skip_tag_ignored_inside_comment(self) -> None:
        # Tags inside HTML comments (e.g. a PR template) do not trigger a skip.
        self.assertFalse(pc.pr_wants_skip("<!-- @skip-pr-bot -->"))


# ----------------------------- required checks ------------------------------


class RequiredCheckRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = make_policy()

    def test_missing_required_check(self) -> None:
        missing, failing, conclusions = pc.summarize_required_checks(self.policy, [])
        self.assertEqual(missing, ["pre-commit"])
        self.assertEqual(failing, [])
        self.assertEqual(conclusions, {})

        result = pc.build_check_results(self.policy, [])[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.pending)

    def test_all_same_name_successes_pass(self) -> None:
        runs = [make_check_run("success"), make_check_run("success")]
        missing, failing, conclusions = pc.summarize_required_checks(self.policy, runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, [])
        self.assertEqual(conclusions, {"pre-commit": "success, success"})

        result = pc.build_check_results(self.policy, runs)[0]
        self.assertTrue(result.passed)
        self.assertFalse(result.pending)

    def test_pending_same_name_run_keeps_combined_check_pending(self) -> None:
        runs = [make_check_run("success"), make_check_run(None)]
        missing, failing, conclusions = pc.summarize_required_checks(self.policy, runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, [])
        self.assertEqual(conclusions, {"pre-commit": "null, success"})

        result = pc.build_check_results(self.policy, runs)[0]
        self.assertFalse(result.passed)
        self.assertTrue(result.pending)

    def test_failure_wins_in_either_input_order(self) -> None:
        for runs in (
            [make_check_run("success"), make_check_run("failure")],
            [make_check_run("failure"), make_check_run("success")],
        ):
            with self.subTest(runs=runs):
                missing, failing, _ = pc.summarize_required_checks(self.policy, runs)
                self.assertEqual(missing, [])
                self.assertEqual(failing, ["pre-commit=failure"])

                result = pc.build_check_results(self.policy, runs)[0]
                self.assertFalse(result.passed)
                self.assertFalse(result.pending)
                self.assertIn("failure", result.details[0])

    def test_failure_is_reported_while_same_name_run_is_pending(self) -> None:
        runs = [make_check_run(None), make_check_run("failure")]
        missing, failing, _ = pc.summarize_required_checks(self.policy, runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, ["pre-commit=failure"])

        result = pc.build_check_results(self.policy, runs)[0]
        self.assertFalse(result.passed)
        self.assertFalse(result.pending)

    def test_every_accepted_conclusion_passes(self) -> None:
        runs = [
            make_check_run("success"),
            make_check_run("neutral"),
            make_check_run("skipped"),
        ]
        missing, failing, _ = pc.summarize_required_checks(self.policy, runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, [])
        self.assertTrue(pc.build_check_results(self.policy, runs)[0].passed)

    def test_failure_comment_checks_every_same_name_run(self) -> None:
        policy = make_policy(
            precommit_failure_comment=pc.FailureComment(
                title="Formatting failed", body="Run pre-commit locally."
            )
        )
        for runs in (
            [make_check_run("success"), make_check_run("failure")],
            [make_check_run("failure"), make_check_run("success")],
        ):
            with self.subTest(runs=runs), mock.patch.object(
                pc, "upsert_comment"
            ) as upsert_comment:
                pc.maybe_comment_precommit_failure(
                    "owner", "repo", 7, "token", policy, runs
                )
                upsert_comment.assert_called_once()

    def test_failure_help_uses_only_selected_scoped_formatting_checks(self) -> None:
        policy = make_policy(
            required_checks=[pc.RequiredCheck("pre-commit / runtimes", [])],
            precommit_failure_comment=pc.FailureComment(
                title="Formatting failed", body="Run pre-commit locally."
            ),
        )
        for name, expected_calls in (
            ("pre-commit / runtimes", 1),
            ("pre-commit / cuid", 0),
            ("pre-commit", 0),
        ):
            with self.subTest(name=name), mock.patch.object(
                pc, "upsert_comment"
            ) as comment:
                pc.maybe_comment_precommit_failure(
                    "owner",
                    "repo",
                    7,
                    "token",
                    policy,
                    [make_check_run("failure", name)],
                )
                self.assertEqual(comment.call_count, expected_calls)


class CheckRunPaginationTests(unittest.TestCase):
    def test_failure_and_required_run_on_second_page(self) -> None:
        for first_required in ([], [make_check_run("success")]):
            with self.subTest(first_required=first_required):
                first_page = first_required + [
                    make_check_run("success", f"other-{index}")
                    for index in range(100 - len(first_required))
                ]
                with mock.patch.object(
                    pc,
                    "gh_get",
                    side_effect=[
                        {"total_count": 101, "check_runs": first_page},
                        {"total_count": 101, "check_runs": [make_check_run("failure")]},
                    ],
                ) as get:
                    runs = pc.get_check_runs("owner", "repo", "sha", "token")
                self.assertEqual(len(runs), 101)
                missing, failing, _ = pc.summarize_required_checks(make_policy(), runs)
                self.assertEqual(missing, [])
                self.assertEqual(failing, ["pre-commit=failure"])
                self.assertEqual(
                    get.call_args_list,
                    [
                        mock.call(
                            "https://api.github.com/repos/owner/repo/commits/sha/check-runs"
                            f"?filter=latest&per_page=100&page={page}",
                            "token",
                        )
                        for page in (1, 2)
                    ],
                )

    def test_empty_page_stops_pagination(self) -> None:
        with mock.patch.object(
            pc,
            "gh_get",
            side_effect=[
                {"total_count": 2, "check_runs": [make_check_run("success")]},
                {"total_count": 2, "check_runs": []},
            ],
        ) as get:
            self.assertEqual(
                pc.get_check_runs("owner", "repo", "sha", "token"),
                [make_check_run("success")],
            )
        self.assertEqual(get.call_count, 2)

    def test_error_on_later_page_does_not_return_partial_success(self) -> None:
        with mock.patch.object(
            pc,
            "gh_get",
            side_effect=[
                {"total_count": 101, "check_runs": [make_check_run("success")] * 100},
                RuntimeError("GET page 2 -> 403"),
            ],
        ), self.assertRaisesRegex(RuntimeError, "GET page 2 -> 403"):
            pc.get_check_runs("owner", "repo", "sha", "token")

    def test_malformed_payload_fails_loudly(self) -> None:
        for payload in (
            [],
            {},
            {"total_count": 1, "check_runs": None},
            {"total_count": -1, "check_runs": []},
            {"total_count": True, "check_runs": []},
            {"total_count": 1, "check_runs": [None]},
        ):
            with self.subTest(payload=payload), mock.patch.object(
                pc, "gh_get", return_value=payload
            ), self.assertRaisesRegex(RuntimeError, "Unexpected check-runs payload"):
                pc.get_check_runs("owner", "repo", "sha", "token")


class RequiredCheckScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = pc.load_policy(THIS_DIR / "policy.yml")

    def test_develop_requires_all_named_jobs(self) -> None:
        selected = pc.select_required_checks(self.policy, "develop")
        self.assertEqual(
            [check.name for check in selected.required_checks],
            [
                "pre-commit / runtimes",
                "pre-commit",
                "pre-commit / cuid",
                "pre-commit / rocprofiler-compute",
            ],
        )

    def test_other_targets_require_the_same_named_jobs(self) -> None:
        for branch in ("pr-bot-test", "release/v1", "feature/topic"):
            with self.subTest(branch=branch):
                selected = pc.select_required_checks(self.policy, branch)
                self.assertEqual(selected.required_checks, self.policy.required_checks)

    def test_branch_filter_includes_and_excludes(self) -> None:
        policy = make_policy(
            required_checks=[
                pc.RequiredCheck("check", ["release/**", "!release/private/**"])
            ]
        )
        self.assertEqual(
            pc.select_required_checks(policy, "release/v1").required_checks,
            policy.required_checks,
        )
        self.assertEqual(
            pc.select_required_checks(policy, "release/private/v1").required_checks, []
        )
        self.assertEqual(
            pc.select_required_checks(policy, "develop").required_checks, []
        )

    def test_unconditional_requirement_applies_to_any_pr(self) -> None:
        policy = make_policy()
        selected = pc.select_required_checks(policy, "anything")
        self.assertEqual(selected.required_checks, policy.required_checks)


class ActionsListingTests(unittest.TestCase):
    def test_invalid_or_truncated_listings_fail_loudly(self) -> None:
        for payload in (
            [],
            {},
            {"total_count": True, "workflow_runs": []},
            {"total_count": -1, "workflow_runs": []},
            {"total_count": 1, "workflow_runs": [None]},
            {"total_count": 1, "workflow_runs": []},
            {"total_count": 1000, "workflow_runs": [make_workflow_run()]},
        ):
            with self.subTest(payload=payload), mock.patch.object(
                pc, "gh_get", return_value=payload
            ), self.assertRaises(RuntimeError):
                list(
                    pc.iter_workflow_runs(
                        "https://api.github.com/runs?event=pull_request",
                        "token",
                    )
                )

    def test_later_page_error_does_not_accept_partial_results(self) -> None:
        with mock.patch.object(
            pc,
            "gh_get",
            side_effect=[
                {"total_count": 101, "workflow_runs": [make_workflow_run()] * 100},
                RuntimeError("GET page 2 -> 403"),
            ],
        ), self.assertRaisesRegex(RuntimeError, "GET page 2 -> 403"):
            list(pc.iter_workflow_runs("https://api.github.com/listing", "token"))

    def test_unbound_external_check_still_uses_commit_check_runs(self) -> None:
        with mock.patch.object(
            pc,
            "gh_get",
            return_value={"total_count": 1, "check_runs": [make_check_run("success")]},
        ) as get:
            result = pc.get_required_check_runs(
                "owner",
                "repo",
                "checks-token",
                "actions-token",
                make_policy(),
                pc.PullRequestContext(7, "sha", "develop", "base-sha"),
            )
        self.assertEqual(result, [make_check_run("success")])
        get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/commits/sha/check-runs?filter=latest&per_page=100&page=1",
            "checks-token",
        )

    def test_malformed_matching_run_fails_loudly(self) -> None:
        for overrides in (
            {"id": None},
            {"id": True},
            {"run_number": None},
            {"check_suite_id": 0},
            {"status": None},
        ):
            with self.subTest(overrides=overrides), mock.patch.object(
                pc,
                "gh_get",
                return_value={
                    "total_count": 1,
                    "workflow_runs": [make_workflow_run(**overrides)],
                },
            ), self.assertRaises(RuntimeError):
                pc.get_required_check_runs(
                    "owner",
                    "repo",
                    "checks-token",
                    "actions-token",
                    pc.load_policy(THIS_DIR / "policy.yml"),
                    pc.PullRequestContext(7, "sha", "develop", "base-sha"),
                )


class RequiredCheckPollingTests(unittest.TestCase):
    """Exercise main(), mocking only GitHub I/O and the polling delay."""

    def _run(
        self,
        snapshots: List[List[Dict[str, Any]]],
        *,
        paths: Optional[List[str]] = None,
        body: Optional[str] = None,
        branch: str = "develop",
        changed_files: Optional[int] = None,
        workflow_snapshots: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> tuple[int, int, int]:
        remaining = iter(snapshots)
        workflows = iter(workflow_snapshots) if workflow_snapshots is not None else None
        active: List[Dict[str, Any]] = []
        active_workflows: List[Dict[str, Any]] = []
        self.check_requests = []
        polls = 0
        files = [
            make_file(path)
            for path in (
                paths
                if paths is not None
                else ["runtimes/Cargo.toml", "emulation/rocjitsu/main.py"]
            )
        ]

        def get(url: str, token: str) -> Any:
            nonlocal active, active_workflows, polls
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path.endswith("/pulls/7"):
                return {
                    "body": (
                        body
                        if body is not None
                        else "Adds runtime tests and formatting.\nFixes #1234\n"
                        "- [x] Look over the contributing guidelines"
                    ),
                    # The base branch may advance after the triggering event.
                    "base": {"ref": branch, "sha": "advanced-base"},
                    "head": {"sha": "sha"},
                    "changed_files": (
                        len(files) if changed_files is None else changed_files
                    ),
                    "user": {"login": "human"},
                }
            if parsed.path.endswith("/pulls/7/files"):
                page = int(query["page"][0])
                return files[(page - 1) * 100 : page * 100]
            if parsed.path.endswith("/issues/7/comments"):
                return []
            if parsed.path.endswith("/actions/workflows/scoped-formatting.yml/runs"):
                self.assertEqual(token, "actions-token")
                self.assertEqual(query["head_sha"], ["sha"])
                self.assertEqual(query["event"], ["pull_request"])
                page = int(query["page"][0])
                if page == 1:
                    # These independent project jobs have already completed.
                    active = next(remaining) + [
                        make_check_run("success", "pre-commit / cuid"),
                        make_check_run("success", "pre-commit / rocprofiler-compute"),
                    ]
                    active_workflows = (
                        next(workflows)
                        if workflows is not None
                        else [
                            make_workflow_run(display_title=f"PR #7: {branch}@base-sha")
                        ]
                    )
                    polls += 1
                return {
                    "total_count": len(active_workflows),
                    "workflow_runs": active_workflows[(page - 1) * 100 : page * 100],
                }
            if re.fullmatch(
                r"/repos/owner/repo/check-suites/\d+/check-runs", parsed.path
            ):
                self.assertEqual(token, "token")
                self.assertEqual(query["filter"], ["latest"])
                self.check_requests.append(parsed.path)
                page = int(query["page"][0])
                return {
                    "total_count": len(active),
                    "check_runs": active[(page - 1) * 100 : page * 100],
                }
            self.fail(f"Unexpected GitHub API request: {url}")

        with mock.patch.dict(
            os.environ,
            {
                "GH_TOKEN": "token",
                "OWNER": "owner",
                "REPO": "repo",
                "PR_NUMBER": "7",
                "SHA": "sha",
                "BASE_REF": branch,
                "BASE_SHA": "base-sha",
                "ACTIONS_TOKEN": "actions-token",
            },
            clear=True,
        ), mock.patch.object(pc, "gh_get", side_effect=get), mock.patch.object(
            pc, "CAN_POST_COMMENTS", True
        ), mock.patch.object(
            pc, "CAN_MUTATE_PR", False
        ), mock.patch.object(
            pc.time, "sleep"
        ) as sleep, mock.patch.object(
            pc, "gh_post", return_value={}
        ) as post, redirect_stdout(
            io.StringIO()
        ):
            result = pc.main([])
        self.posted_comments = [call.args[2]["body"] for call in post.call_args_list]
        return result, polls, sleep.call_count

    def test_retarget_waits_for_new_comparison_before_accepting_or_failing(
        self,
    ) -> None:
        old = make_workflow_run(display_title="PR #7: release@old-base")
        queued = make_workflow_run(id=101, status="queued", conclusion=None)
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        for conclusion, result in (("success", 0), ("failure", 1)):
            with self.subTest(conclusion=conclusion):
                final = [
                    make_check_run("success"),
                    make_check_run(conclusion, "pre-commit / runtimes"),
                ]
                self.assertEqual(
                    self._run(
                        [passed, passed, final],
                        workflow_snapshots=[
                            [old],
                            [old, queued],
                            [old, make_workflow_run(id=101)],
                        ],
                    ),
                    (result, 3, 2),
                )
                self.assertEqual(
                    self.check_requests,
                    ["/repos/owner/repo/check-suites/1010/check-runs"],
                )

    def test_other_pr_base_commit_head_and_event_results_are_ignored(self) -> None:
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        for stale in (
            make_workflow_run(display_title="PR #8: develop@base-sha"),
            make_workflow_run(display_title="PR #7: release@base-sha"),
            make_workflow_run(display_title="PR #7: develop@old-base"),
            make_workflow_run(head_sha="old-head"),
            make_workflow_run(event="push"),
        ):
            with self.subTest(stale=stale):
                self.assertEqual(
                    self._run(
                        [passed, passed],
                        workflow_snapshots=[[stale], [make_workflow_run(id=101)]],
                    ),
                    (0, 2, 1),
                )
                self.assertEqual(
                    self.check_requests,
                    ["/repos/owner/repo/check-suites/1010/check-runs"],
                )

    def test_queued_replacement_and_rerun_do_not_reuse_old_success(self) -> None:
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        failed = [
            make_check_run("success"),
            make_check_run("failure", "pre-commit / runtimes"),
        ]
        for first in (
            [
                make_workflow_run(id=999, run_number=99),
                make_workflow_run(status="queued", conclusion=None),
            ],
            [make_workflow_run(status="in_progress", run_attempt=2, conclusion=None)],
        ):
            with self.subTest(first=first):
                self.assertEqual(
                    self._run(
                        [passed, failed],
                        workflow_snapshots=[first, [make_workflow_run(run_attempt=2)]],
                    ),
                    (1, 2, 1),
                )
                self.assertEqual(
                    self.check_requests,
                    ["/repos/owner/repo/check-suites/1000/check-runs"],
                )

    def test_matching_run_on_second_page_is_not_lost(self) -> None:
        unrelated = [
            make_workflow_run(id=index + 101, display_title="PR #8: develop@base-sha")
            for index in range(100)
        ]
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        self.assertEqual(
            self._run([passed], workflow_snapshots=[unrelated + [make_workflow_run()]]),
            (0, 1, 0),
        )

    def test_cancelled_run_with_missing_jobs_fails_without_timeout(self) -> None:
        self.assertEqual(
            self._run(
                [[]], workflow_snapshots=[[make_workflow_run(conclusion="cancelled")]]
            ),
            (1, 1, 0),
        )

    def test_missing_then_pending_then_failing_workflow_never_passes_early(
        self,
    ) -> None:
        first = make_check_run("success", "pre-commit / runtimes")
        self.assertEqual(
            self._run(
                [
                    [first],
                    [first, make_check_run(None, "pre-commit")],
                    [first, make_check_run("failure", "pre-commit")],
                ]
            ),
            (1, 3, 2),
        )

    def test_completed_partial_rerun_retains_other_successful_jobs(self) -> None:
        unchanged = {**make_check_run("success"), "started_at": "2026-10-14T01:00:00Z"}
        rerun = {
            **make_check_run("success", "pre-commit / runtimes"),
            "started_at": "2026-10-14T02:00:00Z",
        }
        self.assertEqual(
            self._run(
                [[unchanged, rerun]],
                workflow_snapshots=[[make_workflow_run(run_attempt=2)]],
            ),
            (0, 1, 0),
        )

    def test_cancelled_rerun_cannot_accept_all_old_successful_checks(self) -> None:
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        with self.assertRaisesRegex(RuntimeError, "concluded cancelled"):
            self._run(
                [passed],
                workflow_snapshots=[
                    [make_workflow_run(run_attempt=2, conclusion="cancelled")]
                ],
            )

    def test_missing_then_pending_then_successful_workflow_passes_only_at_end(
        self,
    ) -> None:
        first = make_check_run("success", "pre-commit / runtimes")
        self.assertEqual(
            self._run(
                [
                    [first],
                    [first, make_check_run(None, "pre-commit")],
                    [first, make_check_run("success", "pre-commit")],
                ]
            ),
            (0, 3, 2),
        )

    def test_main_observes_failure_on_second_page(self) -> None:
        runs = (
            [make_check_run("success", "pre-commit / runtimes")]
            + [make_check_run("success", f"other-{index}") for index in range(99)]
            + [make_check_run("failure", "pre-commit")]
        )
        self.assertEqual(self._run([runs]), (1, 1, 0))

    def test_unconfigured_checks_do_not_block(self) -> None:
        initial = [
            make_check_run("failure", "unconfigured-check"),
            make_check_run("success", "pre-commit / runtimes"),
        ]
        self.assertEqual(
            self._run(
                [initial, initial + [make_check_run("success", "pre-commit")]],
                branch="pr-bot-test",
            ),
            (0, 2, 1),
        )

    def test_runtime_failure_blocks_other_target_branches(self) -> None:
        self.assertEqual(
            self._run(
                [
                    [
                        make_check_run("success"),
                        make_check_run("failure", "pre-commit / runtimes"),
                    ]
                ],
                branch="pr-bot-test",
            ),
            (1, 1, 0),
        )

    def test_description_failure_still_waits_for_every_applicable_check(self) -> None:
        first = make_check_run("success", "pre-commit / runtimes")
        final = [first, make_check_run("failure", "pre-commit")]
        self.assertEqual(
            self._run([[first], [first], final], body="Missing tracking reference."),
            (1, 3, 1),
        )

    def test_docs_only_pr_waits_for_jobs_to_report_no_work(self) -> None:
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        for paths in ([], ["docs/README.md"]):
            with self.subTest(paths=paths):
                self.assertEqual(self._run([[], passed], paths=paths), (0, 2, 1))

    def test_truncated_file_list_does_not_remove_any_required_checks(self) -> None:
        passed = [
            make_check_run("success", name)
            for name in ("pre-commit", "pre-commit / runtimes")
        ]
        self.assertEqual(
            self._run(
                [[], passed],
                paths=[f"docs/{index}.md" for index in range(3000)],
                changed_files=3001,
            ),
            (0, 2, 1),
        )
        table = next(
            comment
            for comment in reversed(self.posted_comments)
            if "therock-pr-bot-policy-check" in comment
        )
        self.assertIn("3000 of 3001", table)
        self.assertIn("**Forbidden Files** | ⚠️ Warning", table)
        self.assertIn("**Unit Test** | ⚠️ Warning", table)
        self.assertNotIn("Unit Test auto-passed", table)


# ----------------------------- integration -----------------------------------


class IntegrationBlobTests(unittest.TestCase):
    """Feed full [title, description, files] blobs through validators."""

    def setUp(self) -> None:
        self.policy = make_policy()

    def _evaluate(
        self, *, title: str, body: str, files: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}

        e: List[str] = []

        pc.ensure_pr_description(self.policy, body, e)
        out["title_desc"] = e

        e = []
        pc.ensure_no_forbidden_files(self.policy, files, e)
        out["forbidden"] = e

        e = []
        pc.ensure_unit_tests(self.policy, files, e)
        out["unit"] = e
        return out

    def test_fully_compliant_pr(self) -> None:
        result = self._evaluate(
            title="feat(ci): add policy unit tests",
            body=(
                "Adds unit tests for the policy checker.\n"
                "ISSUE ID : TESTUTO-3334\n"
                "- [x] Look over the contributing guidelines at https://..."
            ),
            files=[make_file("src/feature.py"), make_file("tests/test_feature.py")],
        )
        for key, errs in result.items():
            with self.subTest(check=key):
                self.assertEqual(errs, [])

    def test_fully_noncompliant_pr(self) -> None:
        result = self._evaluate(
            title="wip",
            body="too short",
            files=[make_file("secret.pem"), make_file("src/module.py")],
        )
        self.assertTrue(result["title_desc"])
        self.assertTrue(result["forbidden"])
        self.assertTrue(result["unit"])

    def test_docs_only_pr_is_compliant(self) -> None:
        result = self._evaluate(
            title="docs: clarify contributing guide",
            body=(
                "Improves the contributing docs.\n"
                "JIRA ID : DOCS-42\n"
                "- [x] Look over the contributing guidelines at https://..."
            ),
            files=[make_file("docs/CONTRIBUTING.md"), make_file("README.md")],
        )
        for key, errs in result.items():
            with self.subTest(check=key):
                self.assertEqual(errs, [])


# ----------------------------- load_policy -----------------------------------


class LoadPolicyTests(unittest.TestCase):
    """Smoke-test the shipped policy.yml so config drift is caught."""

    def test_load_shipped_policy(self) -> None:
        policy_path = THIS_DIR / "policy.yml"
        if not policy_path.exists():
            self.skipTest("policy.yml not present next to tests")
        policy = pc.load_policy(policy_path)
        self.assertIn(
            "pre-commit / runtimes", [check.name for check in policy.required_checks]
        )
        # Title policy has been removed from policy.yml — the description
        # min-length is the meaningful text-length gate now.
        self.assertGreaterEqual(policy.description_min_length, 0)

    def test_policy_requires_every_scheduled_formatting_scope(self) -> None:
        policy = pc.load_policy(THIS_DIR / "policy.yml")
        workflow = pc.yaml.load(
            (
                THIS_DIR.parents[1] / ".github/workflows/scoped-formatting.yml"
            ).read_text(),
            Loader=pc.yaml.BaseLoader,
        )
        job = workflow["jobs"]["pre-commit"]
        scopes = job["strategy"]["matrix"]["include"]
        self.assertEqual(job["name"], "${{ matrix.check-name }}")
        self.assertEqual(len(scopes), len(policy.required_checks))
        self.assertEqual(
            {scope["check-name"] for scope in scopes},
            {check.name for check in policy.required_checks},
        )
        for check in policy.required_checks:
            with self.subTest(check=check.name):
                self.assertEqual(check.workflow, "scoped-formatting.yml")
                self.assertEqual(check.branches, [])

    def test_other_workflows_do_not_publish_required_check_names(self) -> None:
        policy = pc.load_policy(THIS_DIR / "policy.yml")
        workflows = THIS_DIR.parents[1] / ".github" / "workflows"
        bot_workflow = pc.yaml.load(
            (workflows / "systems-pr-bot.yml").read_text(), Loader=pc.yaml.BaseLoader
        )
        bot_branches = bot_workflow["on"]["pull_request_target"]["branches"]
        # These are concrete target branches, not patterns to intersect.
        for branch in bot_branches:
            self.assertRegex(branch, r"^[\w./-]+$")
        required_names = {check.name for check in policy.required_checks}
        for path in workflows.glob("*.yml"):
            if path.name == "scoped-formatting.yml":
                continue
            # BaseLoader preserves the YAML key "on" rather than treating it
            # as a YAML 1.1 boolean. Filters and job names remain strings.
            workflow = pc.yaml.load(path.read_text(), Loader=pc.yaml.BaseLoader)
            for job_id, job in workflow.get("jobs", {}).items():
                name = job.get("name", job_id)
                if name not in required_names:
                    continue
                with self.subTest(workflow=path.name):
                    trigger = workflow["on"]["pull_request"] or {}
                    # Formatting experiments on other target branches are
                    # outside the bot's required-check contract.
                    if not any(
                        pc._matches_workflow_filter(branch, trigger.get("branches", []))
                        for branch in bot_branches
                    ):
                        continue
                    self.fail(f"{path.name} duplicates formatting check {name}")

    def test_workflows_pin_the_same_comparison_and_events(self) -> None:
        workflows = THIS_DIR.parents[1] / ".github" / "workflows"
        formatting = pc.yaml.load(
            (workflows / "scoped-formatting.yml").read_text(), Loader=pc.yaml.BaseLoader
        )
        bot = pc.yaml.load(
            (workflows / "systems-pr-bot.yml").read_text(), Loader=pc.yaml.BaseLoader
        )
        self.assertEqual(
            formatting["run-name"],
            "${{ github.event_name == 'pull_request' && format('PR #{0}: {1}@{2}', github.event.number, github.base_ref, github.event.pull_request.base.sha) || format('Push {0}', github.sha) }}",
        )
        self.assertEqual(
            pc.PullRequestContext(7, "sha", "release/v1", "base-sha").run_name,
            "PR #7: release/v1@base-sha",
        )
        self.assertEqual(
            formatting["on"]["pull_request"]["types"],
            bot["on"]["pull_request_target"]["types"],
        )
        self.assertEqual(bot["permissions"]["actions"], "read")
        enforce = next(
            step
            for step in bot["jobs"]["policy"]["steps"]
            if step.get("name") == "Enforce policy"
        )
        for key, expression in {
            "SHA": "${{ github.event.pull_request.head.sha }}",
            "BASE_REF": "${{ github.event.pull_request.base.ref }}",
            "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
            "ACTIONS_TOKEN": "${{ github.token }}",
        }.items():
            self.assertEqual(enforce["env"][key], expression)

    def test_invalid_required_check_rules_are_rejected(self) -> None:
        for rules in (
            ["pre-commit"],
            [{"name": ""}],
            [{"name": "same"}, {"name": "same"}],
            [{"name": "check", "paths": "runtimes/**"}],
            [{"name": "check", "paths": ["!runtimes/**"]}],
            [{"name": "check", "branches": [""]}],
            [{"name": "check", "pathz": ["runtimes/**"]}],
            [{"name": "check", "branches": ["[invalid"]}],
            [{"name": "check", "branches": ["!develop"]}],
            [{"name": "check", "branches": "develop"}],
            [{"name": "check", "workflow": None}],
            [{"name": "check", "workflow": "../workflow.yml"}],
            [{"name": "check", "workflow": 17}],
        ):
            with self.subTest(rules=rules), self.assertRaises(ValueError):
                pc._load_required_checks(rules)

    def test_multiline_jira_issue_patterns_loaded(self) -> None:
        """Verify multiline JIRA/ISSUE ID patterns are in the loaded policy."""
        policy_path = THIS_DIR / "policy.yml"
        if not policy_path.exists():
            self.skipTest("policy.yml not present next to tests")
        policy = pc.load_policy(policy_path)

        # Should have at least 5 issue reference patterns (inline + multiline + closing keywords + bare refs + urls)
        self.assertGreaterEqual(len(policy.description_issue_patterns), 5)

        # Verify multiline patterns work by testing them directly
        multiline_jira_pattern = None
        multiline_issue_pattern = None

        for pat in policy.description_issue_patterns:
            if pat.search("JIRA ID\nROCM-25757"):
                multiline_jira_pattern = pat
            if pat.search("ISSUE ID\nAIRUNTIME-2352"):
                multiline_issue_pattern = pat

        self.assertIsNotNone(
            multiline_jira_pattern, "Multiline JIRA ID pattern not found in policy"
        )
        self.assertIsNotNone(
            multiline_issue_pattern, "Multiline ISSUE ID pattern not found in policy"
        )

    def test_unit_test_patterns_exclude_unit_glob(self) -> None:
        """Verify 'unit/**' pattern is NOT in the loaded unit_test_patterns."""
        policy_path = THIS_DIR / "policy.yml"
        if not policy_path.exists():
            self.skipTest("policy.yml not present next to tests")
        policy = pc.load_policy(policy_path)

        # Per team lead request, 'unit/**' was removed from unit_test_patterns.
        # Test files are now recognized ONLY by basename (test_*, *_test.*, Test*).
        self.assertNotIn("unit/**", policy.unit_test_patterns)
        # Verify the allowed patterns ARE present.
        self.assertIn("test_*", policy.unit_test_patterns)
        self.assertIn("*_test.*", policy.unit_test_patterns)
        self.assertIn("*_tests.*", policy.unit_test_patterns)
        self.assertIn("*_gtest.*", policy.unit_test_patterns)
        self.assertIn("Test*", policy.unit_test_patterns)
        self.assertIn("**/test/gtest/**", policy.unit_test_patterns)


if __name__ == "__main__":
    unittest.main()
