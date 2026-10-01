# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if SCRIPTS.name == "tests":
    SCRIPTS = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import rocjitsu_suite_runner  # noqa: E402


class SuiteRunnerTest(unittest.TestCase):
    def test_ctest_result_line(self):
        line = rocjitsu_suite_runner.ctest_result_line(1, 1, "DeviceTest", True, 1.2)
        self.assertIn("1/1 Test #1: DeviceTest", line)
        self.assertIn("Passed", line)
        self.assertIn("1.20 sec", line)

    def test_empty_gtest_filter_is_removed(self):
        env = rocjitsu_suite_runner.test_env({"GTEST_FILTER": ""}, "")
        self.assertNotIn("GTEST_FILTER", env)
        kept = rocjitsu_suite_runner.test_env({}, "rocrtst.Test_Example")
        self.assertEqual(kept["GTEST_FILTER"], "rocrtst.Test_Example")

    def test_streamed_log_uses_ctest_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            launcher = root / "rocjitsu"
            launcher.write_text(
                "#!/bin/sh\n"
                "if [ -n \"${GTEST_FILTER+x}\" ]; then echo \"filter set\"; exit 1; fi\n"
                "echo smoke-ok\n",
                encoding="utf-8",
            )
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            binary = root / "DeviceTest"
            binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
            result = rocjitsu_suite_runner.run_under_rocjitsu(
                str(launcher),
                str(root / "cfg.json"),
                [str(binary), "[smoke]"],
                "DeviceTest",
                30,
                {"GTEST_FILTER": ""},
                str(root),
            )
            log = "\n".join(result["log_lines"])
            self.assertIn("Start 1: DeviceTest", log)
            self.assertIn("1: smoke-ok", log)
            self.assertIn("Passed", log)
            self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
