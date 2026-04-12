"""Tests for CWE-78 command injection vulnerabilities in CI scripts.

These tests verify that shell metacharacters in user-controlled inputs
(PR branch names, SHA values) cannot escape into shell execution.

Discovered via static analysis: bandit B602 (subprocess with shell=True)
and B605 (os.system/os.popen calls) on the basic-static-analysis branch.
"""
import subprocess
import unittest
from unittest.mock import patch, MagicMock

import import_subrepo_prs


class TestRunFunctionInjection(unittest.TestCase):
    """Verify import_subrepo_prs.run() is safe against shell injection."""

    MALICIOUS_BRANCH = "feature;rm -rf / #"
    SAFE_CMD_PREFIX = "git subtree pull --prefix=projects/foo https://github.com/org/repo.git"

    @patch("import_subrepo_prs.subprocess.check_call")
    def test_shell_metacharacters_not_interpreted(self, mock_call):
        """Shell metacharacters in branch names must not be interpreted.

        When run() receives a string command, shlex.split tokenizes it — but
        critically, shell=False means ';', '|', '$()' etc. are NEVER evaluated
        by a shell.  The tokens are passed as literal argv entries to execve().
        """
        cmd = f"{self.SAFE_CMD_PREFIX} {self.MALICIOUS_BRANCH}"
        import_subrepo_prs.run(cmd)

        mock_call.assert_called_once()
        args, kwargs = mock_call.call_args

        # The critical assertion: shell must NOT be True
        self.assertFalse(
            kwargs.get("shell", False),
            "run() must not use shell=True — attacker-controlled branch names "
            "can inject arbitrary commands via PR fork branch names",
        )

        # The command must be split into a list of arguments
        called_cmd = args[0]
        self.assertIsInstance(
            called_cmd, list,
            "run() must pass an argument list, not a string",
        )

    @patch("import_subrepo_prs.subprocess.check_call")
    def test_list_preserves_metacharacters_literally(self, mock_call):
        """When passing a pre-built list, metacharacters stay in one token."""
        cmd_list = [
            "git", "subtree", "pull", "--prefix=projects/foo",
            "https://github.com/org/repo.git", self.MALICIOUS_BRANCH,
        ]
        import_subrepo_prs.run(cmd_list)

        mock_call.assert_called_once()
        args, kwargs = mock_call.call_args
        self.assertFalse(kwargs.get("shell", False))
        # The branch name with semicolon stays as a single argument
        self.assertIn(self.MALICIOUS_BRANCH, args[0])

    @patch("import_subrepo_prs.subprocess.check_call")
    def test_run_with_list_argument(self, mock_call):
        """run() should also accept a pre-split list."""
        cmd_list = ["git", "config", "user.name", "bot"]
        import_subrepo_prs.run(cmd_list)

        mock_call.assert_called_once()
        args, kwargs = mock_call.call_args
        self.assertFalse(kwargs.get("shell", False))
        self.assertEqual(args[0], cmd_list)

    @patch("import_subrepo_prs.subprocess.check_call")
    def test_or_true_pattern_handled_safely(self, mock_call):
        """The '|| true' shell pattern must not require shell=True.
        Instead, failures should be caught with try/except."""
        # This tests that the code doesn't rely on shell for error suppression.
        # The old code had: run(f"git merge --abort || true")
        # The fix should use try/except instead.
        cmd = "git merge --abort"
        try:
            import_subrepo_prs.run(cmd)
        except subprocess.CalledProcessError:
            pass  # Expected if merge --abort fails — that's the safe pattern

        mock_call.assert_called_once()
        args, kwargs = mock_call.call_args
        self.assertFalse(kwargs.get("shell", False))


class TestPrCategoryLabelInjection(unittest.TestCase):
    """Verify pr_category_label.py does not use os.popen/os.system."""

    def test_no_os_popen_or_system(self):
        """pr_category_label.py must not use os.popen() or os.system()."""
        import pr_category_label
        import inspect
        source = inspect.getsource(pr_category_label)
        self.assertNotIn(
            "os.popen",
            source,
            "os.popen() is vulnerable to command injection (CWE-78). "
            "Use subprocess.run() with argument lists instead.",
        )
        self.assertNotIn(
            "os.system",
            source,
            "os.system() is vulnerable to command injection (CWE-78). "
            "Use subprocess.run() with argument lists instead.",
        )


class TestPrDetectChangedSubtreesInjection(unittest.TestCase):
    """Verify pr_detect_changed_subtrees.py does not use os.popen/os.system."""

    def test_no_os_popen_or_system(self):
        """pr_detect_changed_subtrees.py must not use os.popen() or os.system()."""
        import pr_detect_changed_subtrees
        import inspect
        source = inspect.getsource(pr_detect_changed_subtrees)
        self.assertNotIn(
            "os.popen",
            source,
            "os.popen() is vulnerable to command injection (CWE-78). "
            "Use subprocess.run() with argument lists instead.",
        )
        self.assertNotIn(
            "os.system",
            source,
            "os.system() is vulnerable to command injection (CWE-78). "
            "Use subprocess.run() with argument lists instead.",
        )
