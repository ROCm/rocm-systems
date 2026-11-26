#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Teardown checks for the emulated pod rank helper."""

import importlib.util
from pathlib import Path
import signal
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/run-rccl-pod-emulated-node.py"
SPEC = importlib.util.spec_from_file_location("run_rccl_pod_emulated_node", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RankCleanupTest(unittest.TestCase):
    def test_exited_group_does_not_prevent_peer_cleanup(self):
        class Process:
            def __init__(self, pid):
                self.pid = pid
                self.waited = False

            def poll(self):
                return None

            def wait(self, timeout):
                self.waited = True
                return 0

        processes = [Process(101), Process(102)]
        calls = []

        def kill_group(pid, signum):
            calls.append((pid, signum))
            if pid == 101:
                raise ProcessLookupError

        with mock.patch.object(MODULE.os, "killpg", side_effect=kill_group):
            MODULE.cleanup_ranks(processes)

        self.assertEqual(calls, [(101, signal.SIGTERM), (102, signal.SIGTERM)])
        self.assertTrue(all(process.waited for process in processes))


if __name__ == "__main__":
    unittest.main()
