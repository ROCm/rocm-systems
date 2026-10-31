# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if SCRIPTS.name == "tests":
    SCRIPTS = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import test_runner  # noqa: E402


class TestRunnerParallelTest(unittest.TestCase):
    def test_positive_count_appends_ctest_parallel(self):
        cmd = test_runner.ctest_command(5, "/tests", "^smoke$", 14400, "skip_me")
        self.assertEqual(cmd[:2], ["ctest", "--test-dir"])
        self.assertIn("--parallel", cmd)
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "5")

    def test_zero_omits_parallel(self):
        cmd = test_runner.ctest_command(0, "/tests", "^smoke$", 14400, "skip_me")
        self.assertNotIn("--parallel", cmd)


if __name__ == "__main__":
    unittest.main()
