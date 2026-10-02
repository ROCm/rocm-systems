# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if SCRIPTS.name == "tests":
    SCRIPTS = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import rocjitsu_suite_runner  # noqa: E402


class SuiteRunnerTest(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "Linux process-group contract")
    def test_timeout_kills_descendants_holding_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Path(tmp) / "child.pid"
            code = (
                "import subprocess,sys,time,pathlib; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"pathlib.Path({str(child)!r}).write_text(str(p.pid)); "
                "print('started', flush=True); time.sleep(60)"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", code], stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
            started = time.monotonic()
            lines, returncode, timed_out = rocjitsu_suite_runner.stream_output(proc, 1, 1)
            self.assertTrue(timed_out)
            self.assertNotEqual(returncode, 0)
            self.assertLess(time.monotonic() - started, 8)
            self.assertIn("started", lines)
            # A killed child can remain a zombie briefly until init reaps it.
            state = Path(f"/proc/{child.read_text()}/stat")
            if state.exists():
                self.assertEqual(state.read_text().split()[2], "Z")

    @unittest.skipUnless(os.name == "posix", "POSIX process-group contract")
    def test_timeout_after_stdout_is_closed(self):
        proc = subprocess.Popen(
            [sys.executable, "-c", "import os,time; os.close(1); os.close(2); time.sleep(60)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        started = time.monotonic()
        _, _, timed_out = rocjitsu_suite_runner.stream_output(proc, 1, 1)
        self.assertTrue(timed_out)
        self.assertLess(time.monotonic() - started, 8)

    def test_missing_launcher_fails_and_writes_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "suite.log"
            result = subprocess.run([
                sys.executable, str(SCRIPTS / "rocjitsu_suite_runner.py"),
                "--rocjitsu", str(Path(tmp) / "missing"), "--config", "unused",
                "--name", "missing-suite", "--cwd", tmp, "--log", str(log),
                "--", "unused",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("missing-suite", log.read_text())

    @unittest.skipUnless(os.name == "posix", "POSIX launcher fixture")
    def test_nonzero_test_exit_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            launcher = Path(tmp) / "rocjitsu"
            launcher.write_text("#!/bin/sh\necho failed-test\nexit 7\n")
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
            result = rocjitsu_suite_runner.run_under_rocjitsu(
                str(launcher), "unused", ["unused"], "failing-suite", 5,
                os.environ.copy(), tmp,
            )
            self.assertFalse(result["passed"])
            self.assertEqual(result["returncode"], 7)
            self.assertIn("***Failed", "\n".join(result["log_lines"]))

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

    @unittest.skipUnless(os.name == "posix", "POSIX launcher fixture")
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
