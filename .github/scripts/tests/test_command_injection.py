"""Tests that import_subrepo_prs.run() never invokes a shell (CWE-78).

PR-derived values (head branch/URL) and env vars flow into run(), so command
strings must be executed as argv lists without a shell -- otherwise shell
metacharacters in those values are interpreted as commands.
"""

import unittest
from unittest.mock import patch

from import_subrepo_prs import run

# description / input / expected argv passed to subprocess.check_call
CASES = [
    {
        "description": "positive: list command passed through unchanged",
        "cmd": ["git", "status"],
        "expected": ["git", "status"],
    },
    {
        "description": "positive: string command is tokenised (shlex)",
        "cmd": "git checkout develop",
        "expected": ["git", "checkout", "develop"],
    },
    {
        "description": "boundary: single-token string",
        "cmd": "status",
        "expected": ["status"],
    },
    {
        "description": "corner: quoted arg stays one token",
        "cmd": "git config user.name 'systems-assistant[bot]'",
        "expected": ["git", "config", "user.name", "systems-assistant[bot]"],
    },
    {
        "description": "corner: shell metacharacters stay inside one argv element",
        "cmd": ["git", "checkout", "-b", "evil; rm -rf / #"],
        "expected": ["git", "checkout", "-b", "evil; rm -rf / #"],
    },
]


class TestRunNeverUsesShell(unittest.TestCase):
    def test_cases(self):
        for case in CASES:
            with self.subTest(case["description"]):
                with patch("import_subrepo_prs.subprocess.check_call") as mock_call:
                    run(case["cmd"])
                    args, kwargs = mock_call.call_args
                    # command is passed as the expected argv list ...
                    self.assertEqual(args[0], case["expected"], case["description"])
                    # ... and never with shell=True
                    self.assertNotEqual(kwargs.get("shell"), True, case["description"])


if __name__ == "__main__":
    unittest.main()
