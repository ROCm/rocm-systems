"""
Unit + integration tests for policy_check.py.

These let us iterate on policies WITHOUT pushing branches or running workflows:
  • Unit tests   — exercise individual validators / regex patterns.
  • Integration  — feed blobs of [branch, title, description, files] through
                   the higher-level ensure_* functions.

Run locally:
    python -m unittest .github/therock_pr_bot/test_policy_check_ut.py -v
    # or
    pytest .github/therock_pr_bot/test_policy_check_ut.py
"""

import io
import re
import sys
import tempfile
import unittest
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
        required_checks=["pre-commit"],
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


class RequiredCheckMainTests(unittest.TestCase):
    """Use real trusted workflow filters, with every GitHub operation mocked."""

    def run_policy(self, filename: str, runs: List[Dict[str, Any]]) -> int:
        pr = {
            "body": "Fixes #13877\n- [x] Look over the contributing guidelines.\n",
            "base": {"ref": "develop"},
            "user": {"login": "contributor"},
        }
        with (
            mock.patch.dict(
                pc.os.environ,
                {
                    "GH_TOKEN": "test-only",
                    "OWNER": "test",
                    "REPO": "test",
                    "PR_NUMBER": "1",
                    "SHA": "test-only",
                },
                clear=True,
            ),
            mock.patch.object(pc, "load_policy", return_value=make_policy()),
            mock.patch.object(pc, "get_pr", return_value=pr),
            mock.patch.object(pc, "iter_pr_files", return_value=[make_file(filename)]),
            mock.patch.object(pc, "get_check_runs", return_value=runs),
            mock.patch.object(
                pc, "gh_get", side_effect=AssertionError("unexpected API read")
            ),
            mock.patch.object(pc, "upsert_comment"),
            mock.patch.object(pc, "update_comment_if_exists"),
            mock.patch.object(pc, "add_label"),
            mock.patch.object(pc, "remove_label"),
            mock.patch.object(pc, "maybe_comment_precommit_failure"),
            redirect_stdout(io.StringIO()),
        ):
            return pc.main(["--timeout-seconds", "0", "--poll-seconds", "0"])

    def test_clr_does_not_wait_for_untriggered_precommit(self) -> None:
        self.assertEqual(
            self.run_policy("projects/clr/hipamd/src/hip_graph.cpp", []), 0
        )

    def test_covered_missing_check_still_fails(self) -> None:
        self.assertEqual(self.run_policy("emulation/rocjitsu/example.py", []), 1)

    def test_covered_failure_still_fails(self) -> None:
        runs = [{"name": "pre-commit", "conclusion": "failure"}]
        self.assertEqual(self.run_policy("emulation/rocjitsu/example.py", runs), 1)

    def test_covered_success_passes(self) -> None:
        runs = [{"name": "pre-commit", "conclusion": "success"}]
        self.assertEqual(self.run_policy("emulation/rocjitsu/example.py", runs), 0)

    def test_observed_failure_outside_scope_still_fails(self) -> None:
        runs = [{"name": "pre-commit", "conclusion": "failure"}]
        self.assertEqual(self.run_policy("projects/clr/example.cpp", runs), 1)

    def test_covered_running_check_still_waits(self) -> None:
        runs = [{"name": "pre-commit", "conclusion": None}]
        self.assertEqual(self.run_policy("emulation/rocjitsu/example.py", runs), 1)


class RequiredCheckScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workflows = Path(self.directory.name)

    def workflow(self, trigger: str, *, name: str = "format.yml") -> None:
        (self.workflows / name).write_text(
            "on:\n" + trigger + "\njobs:\n  pre-commit:\n    runs-on: ubuntu-latest\n"
        )

    def required(
        self, *paths: str, base: str = "develop", observed: tuple = ()
    ) -> List[str]:
        return pc.required_checks_for_pr(
            ["pre-commit"],
            [make_file(path) for path in paths],
            base,
            observed_checks=observed,
            workflows_dir=self.workflows,
        )

    def test_no_known_producer_remains_required(self) -> None:
        self.assertEqual(self.required("projects/clr/a.cpp"), ["pre-commit"])

    def test_uncovered_path_does_not_create_a_requirement(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        self.assertEqual(self.required("projects/clr/a.cpp"), [])
        self.assertEqual(self.required("emulation/module/a.py"), ["pre-commit"])

    def test_mixed_scope_changes_still_require_the_check(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        self.assertEqual(
            self.required("projects/clr/a.cpp", "emulation/a.py"), ["pre-commit"]
        )

    def test_ordered_negative_filters_and_reinclusion(self) -> None:
        self.workflow(
            '  pull_request:\n    paths: ["shared/**", "!**/*.md", "shared/KEEP.md"]\n'
        )
        self.assertEqual(self.required("shared/docs/README.md"), [])
        self.assertEqual(self.required("shared/KEEP.md"), ["pre-commit"])

    def test_paths_ignore_requires_any_nonignored_file(self) -> None:
        self.workflow('  pull_request:\n    paths-ignore: ["docs/**", "**/*.md"]\n')
        self.assertEqual(self.required("docs/a.txt", "README.md"), [])
        self.assertEqual(self.required("docs/a.txt", "a.cpp"), ["pre-commit"])

    def test_single_star_does_not_cross_a_directory(self) -> None:
        self.workflow('  pull_request:\n    paths: ["src/*.py"]\n')
        self.assertEqual(self.required("src/nested/a.py"), [])
        self.assertEqual(self.required("src/a.py"), ["pre-commit"])

    def test_double_star_matches_zero_or_more_directories(self) -> None:
        self.workflow('  pull_request:\n    paths: ["src/**/*.py"]\n')
        for path in ("src/a.py", "src/one/two/a.py"):
            with self.subTest(path=path):
                self.assertEqual(self.required(path), ["pre-commit"])

    def test_base_branch_filter(self) -> None:
        self.workflow('  pull_request:\n    branches: ["testbranch"]\n')
        self.assertEqual(self.required("a.py"), [])
        self.assertEqual(self.required("a.py", base="testbranch"), ["pre-commit"])
        self.assertEqual(self.required("a.py", base=""), ["pre-commit"])

    def test_branches_ignore(self) -> None:
        self.workflow('  pull_request:\n    branches-ignore: ["release/**"]\n')
        self.assertEqual(self.required("a.py", base="release/10.0"), [])
        self.assertEqual(self.required("a.py"), ["pre-commit"])

    def test_another_applicable_producer_keeps_the_check_required(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        self.workflow(
            '  pull_request:\n    paths: ["shared/cuid/**"]\n', name="cuid.yml"
        )
        self.assertEqual(self.required("shared/cuid/a.h"), ["pre-commit"])

    def test_renamed_from_covered_path_remains_required(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        files = [
            dict(make_file("projects/clr/a.py"), previous_filename="emulation/a.py")
        ]
        self.assertEqual(
            pc.required_checks_for_pr(
                ["pre-commit"], files, "develop", workflows_dir=self.workflows
            ),
            ["pre-commit"],
        )

    def test_observed_check_is_kept_even_if_all_filters_exclude_it(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        self.assertEqual(
            self.required("a.py", observed=("pre-commit",)), ["pre-commit"]
        )

    def test_external_check_without_known_producer_remains_required(self) -> None:
        self.workflow('  pull_request:\n    paths: ["emulation/**"]\n')
        self.assertEqual(
            pc.required_checks_for_pr(
                ["pre-commit", "external-security"],
                [make_file("projects/clr/a.cpp")],
                "develop",
                workflows_dir=self.workflows,
            ),
            ["external-security"],
        )

    def test_unsupported_glob_remains_required(self) -> None:
        for pattern in (
            "src/a?.py",
            "src/a+.py",
            "src/[ab].py",
            "src/a**b.py",
            "src/{a,b}.py",
            "src/@(a|b).py",
        ):
            with self.subTest(pattern=pattern):
                self.workflow(f'  pull_request:\n    paths: ["{pattern}"]\n')
                self.assertEqual(self.required("elsewhere/a.cpp"), ["pre-commit"])

    def test_indirect_or_manual_trigger_remains_required(self) -> None:
        for event in ("workflow_call", "workflow_run", "workflow_dispatch"):
            with self.subTest(event=event):
                self.workflow(f"  {event}:\n")
                self.assertEqual(self.required("a.cpp"), ["pre-commit"])

    def test_unconditional_pr_trigger_requires_the_check(self) -> None:
        (self.workflows / "format.yml").write_text(
            "on: [push, pull_request]\njobs:\n  pre-commit:\n    runs-on: ubuntu-latest\n"
        )
        self.assertEqual(self.required("a.cpp"), ["pre-commit"])

    def test_yaml_merge_does_not_hide_a_producer(self) -> None:
        (self.workflows / "format.yml").write_text(
            "defaults: &job\n  name: pre-commit\n  runs-on: ubuntu-latest\n"
            "on:\n  pull_request:\n    paths: [emulation/**]\n"
            "jobs:\n  format:\n    <<: *job\n"
        )
        self.assertEqual(self.required("a.cpp"), [])
        self.assertEqual(self.required("emulation/a.py"), ["pre-commit"])

    def test_target_event_filters_apply(self) -> None:
        self.workflow('  pull_request_target:\n    paths: ["emulation/**"]\n')
        self.assertEqual(self.required("a.cpp"), [])

    def test_malformed_workflow_does_not_relax_the_gate(self) -> None:
        (self.workflows / "format.yml").write_text("on: [\n")
        self.assertEqual(self.required("a.cpp"), ["pre-commit"])

    def test_invalid_filters_do_not_relax_the_gate(self) -> None:
        for filters in (
            "paths: []",
            'paths: ["!emulation/**"]',
            'paths: ["emulation/**"]\n    paths-ignore: ["docs/**"]',
        ):
            with self.subTest(filters=filters):
                self.workflow("  pull_request:\n    " + filters + "\n")
                self.assertEqual(self.required("a.cpp"), ["pre-commit"])

    def test_actual_trusted_workflow_scopes(self) -> None:
        cases = {
            "projects/clr/a.cpp": [],
            "projects/hip-tests/catch/unit/stream/a.cc": [],
            "emulation/mirage/a.py": ["pre-commit"],
            "shared/cuid/a.h": ["pre-commit"],
            "shared/cuid/docs/a.md": [],
            "projects/rocprofiler-compute/a.py": ["pre-commit"],
            ".pre-commit-config.yaml": ["pre-commit"],
            ".github/workflows/rocjitsu-formatting.yml": ["pre-commit"],
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(
                    pc.required_checks_for_pr(
                        ["pre-commit"], [make_file(path)], "develop"
                    ),
                    expected,
                )


class CheckRunPaginationTests(unittest.TestCase):
    def test_required_check_after_first_page_is_not_lost(self) -> None:
        pages = [
            {"check_runs": [{"name": f"build-{i}"} for i in range(100)]},
            {"check_runs": [{"name": "pre-commit", "conclusion": "failure"}]},
        ]
        with mock.patch.object(pc, "gh_get", side_effect=pages) as get:
            runs = pc.get_check_runs("test", "test", "head", "test-only")
        self.assertEqual(len(runs), 101)
        self.assertIn("page=2", get.call_args_list[1].args[0])
        self.assertEqual(
            pc.summarize_required_checks(make_policy(), runs)[1], ["pre-commit=failure"]
        )

    def test_malformed_response_is_an_error(self) -> None:
        with mock.patch.object(pc, "gh_get", return_value={"check_runs": None}):
            with self.assertRaises(RuntimeError):
                pc.get_check_runs("test", "test", "head", "test-only")


class LoadPolicyTests(unittest.TestCase):
    """Smoke-test the shipped policy.yml so config drift is caught."""

    def test_load_shipped_policy(self) -> None:
        policy_path = THIS_DIR / "policy.yml"
        if not policy_path.exists():
            self.skipTest("policy.yml not present next to tests")
        policy = pc.load_policy(policy_path)
        self.assertIn("pre-commit", policy.required_checks)
        # Title policy has been removed from policy.yml — the description
        # min-length is the meaningful text-length gate now.
        self.assertGreaterEqual(policy.description_min_length, 0)

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
