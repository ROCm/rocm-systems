#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stdlib-only unit tests for ``dme_integration.services``."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

# Allow running directly as well as via unittest discovery from
# ``projects/amdsmi/tests``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dme_integration import services


class PidCmdlineTest(unittest.TestCase):
    def test_current_process_cmdline_contains_python(self):
        cmdline = services._pid_cmdline(os.getpid())
        self.assertIn("python", cmdline.lower())

    def test_nonexistent_pid_returns_empty(self):
        # PID 2^22 is above the typical pid_max, so it should not exist.
        self.assertEqual(services._pid_cmdline(4194304), "")


class StopExpectedGuardTest(unittest.TestCase):
    def _spawn_sleep(self) -> tuple[subprocess.Popen, Path]:
        proc = subprocess.Popen(["sleep", "30"])
        self.addCleanup(self._reap, proc)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pid", delete=False, encoding="utf-8")
        tmp.write(str(proc.pid))
        tmp.close()
        pid_file = Path(tmp.name)
        self.addCleanup(pid_file.unlink, missing_ok=True)
        return proc, pid_file

    @staticmethod
    def _reap(proc: subprocess.Popen) -> None:
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    def test_stop_with_mismatched_expected_skips_kill(self):
        proc, pid_file = self._spawn_sleep()
        services.stop(name="sleeper", pid_file=pid_file, expected="definitely-not-this-binary")
        # Process must still be alive; the guard should have skipped the kill.
        self.assertIsNone(proc.poll())
        # The pid file is removed even when the kill is skipped.
        self.assertFalse(pid_file.exists())

    def test_stop_without_expected_kills_process(self):
        proc, pid_file = self._spawn_sleep()
        services.stop(name="sleeper", pid_file=pid_file)
        # Give the signal a moment to take effect.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.1)
        self.assertIsNotNone(proc.poll())
        self.assertFalse(pid_file.exists())


if __name__ == "__main__":
    unittest.main()
