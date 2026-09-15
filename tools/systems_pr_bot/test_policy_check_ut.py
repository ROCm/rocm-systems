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

import re
import sys
import unittest
import unittest.mock
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class CheckRunPaginationTests(unittest.TestCase):
    """get_check_runs must read every page, not just the first 100."""

    def _fake_gh_get(self, pages: List[Dict[str, Any]]):
        calls: List[str] = []

        def _get(url: str, token: str) -> Dict[str, Any]:
            calls.append(url)
            # page= is 1-based; return an empty payload past the end.
            idx = int(url.rsplit("page=", 1)[1]) - 1
            return pages[idx] if idx < len(pages) else {"check_runs": []}

        return _get, calls

    def test_reads_beyond_the_first_page(self) -> None:
        # 117 runs is the real shape of a busy rocm-systems PR, and the
        # required `pre-commit` check lands on page 2.
        page1 = {
            "total_count": 117,
            "check_runs": [
                {"name": f"job-{i}", "conclusion": "success"} for i in range(100)
            ],
        }
        page2 = {
            "total_count": 117,
            "check_runs": (
                [{"name": f"job-{i}", "conclusion": "success"} for i in range(100, 116)]
                + [{"name": "pre-commit", "conclusion": "success"}]
            ),
        }
        fake, calls = self._fake_gh_get([page1, page2])
        with unittest.mock.patch.object(pc, "gh_get", fake):
            runs = pc.get_check_runs("o", "r", "deadbeef", "tok")

        self.assertEqual(len(runs), 117)
        self.assertIn("pre-commit", {r["name"] for r in runs})
        self.assertEqual(len(calls), 2)

    def test_stops_once_total_count_is_reached(self) -> None:
        page1 = {
            "total_count": 2,
            "check_runs": [
                {"name": "a", "conclusion": "success"},
                {"name": "b", "conclusion": "success"},
            ],
        }
        fake, calls = self._fake_gh_get([page1])
        with unittest.mock.patch.object(pc, "gh_get", fake):
            runs = pc.get_check_runs("o", "r", "sha", "tok")

        self.assertEqual(len(runs), 2)
        self.assertEqual(len(calls), 1, "should not request a page it does not need")

    def test_stops_on_empty_page_without_total_count(self) -> None:
        pages = [{"check_runs": [{"name": "a", "conclusion": "success"}]}]
        fake, _ = self._fake_gh_get(pages)
        with unittest.mock.patch.object(pc, "gh_get", fake):
            runs = pc.get_check_runs("o", "r", "sha", "tok")

        self.assertEqual(len(runs), 1)

    def test_a_server_that_ignores_page_raises_instead_of_looping(self) -> None:
        # Never empty, total_count always out of reach: the pre-cap loop spun
        # forever, and nothing else in the bot bounds this call.
        calls: List[str] = []

        def _get(url: str, token: str) -> Dict[str, Any]:
            calls.append(url)
            return {
                "total_count": 10**9,
                "check_runs": [{"name": "a", "conclusion": "success"}],
            }

        with unittest.mock.patch.object(pc, "gh_get", _get):
            with self.assertRaises(RuntimeError) as ctx:
                pc.get_check_runs("o", "r", "sha", "tok")

        self.assertIn("pagination exceeded", str(ctx.exception))
        self.assertEqual(len(calls), pc.CHECK_RUNS_MAX_PAGES)


class DuplicateCheckRunNameTests(unittest.TestCase):
    """Several workflows publish a job named `pre-commit`; duplicates must not mask."""

    def _policy(self) -> pc.Policy:
        # make_policy already defaults required_checks to ["pre-commit"].
        return make_policy()

    def test_a_failing_duplicate_is_not_masked_by_a_passing_one(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": "failure"},
            {"name": "pre-commit", "conclusion": "success"},
        ]
        missing, failing, _ = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, ["pre-commit=failure"])

    def test_order_does_not_matter(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": "success"},
            {"name": "pre-commit", "conclusion": "failure"},
        ]
        _, failing, _ = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual(failing, ["pre-commit=failure"])

    def test_failure_wins_over_a_still_running_duplicate(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": None},
            {"name": "pre-commit", "conclusion": "failure"},
        ]
        _, failing, _ = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual(
            failing,
            ["pre-commit=failure"],
            "a known failure should report now, not wait out the poll window",
        )

    def test_pending_wins_over_success(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": "success"},
            {"name": "pre-commit", "conclusion": None},
        ]
        missing, failing, conc = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual(missing, [])
        self.assertEqual(failing, [])
        self.assertEqual(conc["pre-commit"], "null")

    def test_all_passing_duplicates_pass(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": "success"},
            {"name": "pre-commit", "conclusion": "skipped"},
        ]
        missing, failing, _ = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual((missing, failing), ([], []))

    def test_missing_required_check_is_reported_missing(self) -> None:
        missing, failing, _ = pc.summarize_required_checks(self._policy(), [])
        self.assertEqual(missing, ["pre-commit"])
        self.assertEqual(failing, [])

    def test_skipped_still_counts_as_a_pass(self) -> None:
        runs = [{"name": "pre-commit", "conclusion": "skipped"}]
        missing, failing, _ = pc.summarize_required_checks(self._policy(), runs)
        self.assertEqual((missing, failing), ([], []))

    def test_readiness_and_the_table_agree_on_the_same_run(self) -> None:
        # The bug this guards: summarize_required_checks collapsed worst-wins
        # while main()'s readiness test collapsed last-wins, so a stale passing
        # duplicate declared the PR ready while the gating run was in flight.
        # Both must resolve to the SAME representative for every ordering.
        pending = {"name": "pre-commit", "id": 200, "conclusion": None}
        passing = {"name": "pre-commit", "id": 100, "conclusion": "success"}
        for runs in ([pending, passing], [passing, pending]):
            with self.subTest(order=[r["id"] for r in runs]):
                _, _, conc = pc.summarize_required_checks(self._policy(), runs)
                chosen = pc.effective_run_by_name(runs)["pre-commit"]
                self.assertEqual(chosen["id"], 200, "pending must beat success")
                self.assertEqual(conc["pre-commit"], "null")

    def test_build_check_results_reports_the_failing_duplicate(self) -> None:
        runs = [
            {"name": "pre-commit", "conclusion": "success"},
            {"name": "pre-commit", "conclusion": "failure"},
        ]
        rows = pc.build_check_results(self._policy(), runs)
        row = next(r for r in rows if r.name == "pre-commit")
        self.assertFalse(row.passed)


class PrecommitHelpCommentTests(unittest.TestCase):
    """The remediation comment must key off the same run the table does."""

    def _run(self, runs: List[Dict[str, Any]]) -> List[str]:
        policy = make_policy(
            precommit_failure_comment=pc.FailureComment(
                title="Pre-commit check failed", body="run it locally"
            )
        )
        posted: List[str] = []
        with unittest.mock.patch.object(
            pc, "upsert_comment", lambda *a, **k: posted.append(a[-1])
        ):
            pc.maybe_comment_precommit_failure("o", "r", 1, "tok", policy, runs)
        return posted

    def test_a_passing_duplicate_listed_first_does_not_suppress_the_comment(
        self,
    ) -> None:
        # A first-wins scan picked the success and stayed silent, while the
        # caller -- which only reaches here because the check is failing --
        # printed "pre-commit=failure". The developer got no remediation text.
        posted = self._run(
            [
                {"name": "pre-commit", "conclusion": "success"},
                {"name": "pre-commit", "conclusion": "failure"},
            ]
        )
        self.assertEqual(len(posted), 1)
        self.assertIn("Pre-commit check failed", posted[0])

    def test_all_passing_stays_silent(self) -> None:
        self.assertEqual(
            self._run([{"name": "pre-commit", "conclusion": "success"}]), []
        )


if __name__ == "__main__":
    unittest.main()
