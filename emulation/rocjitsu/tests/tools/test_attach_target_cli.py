#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Exercise rocjitsu --attach target resolution without starting a daemon."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest


class AttachTargetCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.environ.get("RJ_ROCJITSU"):
            raise unittest.SkipTest("RJ_ROCJITSU is not set")
        cls.cli = os.environ["RJ_ROCJITSU"]

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rj-attach-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.env = dict(os.environ)
        for key in ("LD_PRELOAD", "ROCJITSU_INVOCATION_DIR"):
            self.env.pop(key, None)
        self.env["ROCJITSU_RUNTIME_DIR"] = str(self.runtime)
        self.sockets = []
        self.addCleanup(self.close_sockets)

    def close_sockets(self):
        for bound in self.sockets:
            bound.close()

    def bind_daemon_socket(self, runtime_dir):
        runtime_dir.mkdir(parents=True, exist_ok=True)
        bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound.bind(str(runtime_dir / "daemon.sock"))
        bound.listen(1)
        self.sockets.append(bound)

    def run_attach(self, target):
        return subprocess.run(
            [
                self.cli,
                "--attach",
                str(target),
                "--",
                sys.executable,
                "-c",
                "import os; print(os.environ.get('ROCJITSU_INVOCATION_DIR', ''))",
            ],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_default_id_path_and_socket_targets(self):
        cases = [
            ("default", self.runtime, "default"),
            ("worker-a", self.runtime / "worker-a", "id"),
            ("9999999", self.runtime / "9999999", "numeric-id"),
            (self.root / "explicit", self.root / "explicit", "path"),
            (
                self.root / "explicit-socket" / "daemon.sock",
                self.root / "explicit-socket",
                "socket-path",
            ),
        ]
        for target, runtime_dir, label in cases:
            with self.subTest(label=label):
                self.bind_daemon_socket(runtime_dir)
                result = self.run_attach(target)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.strip(), str(runtime_dir))

    def test_rejects_non_socket_target(self):
        runtime_dir = self.root / "not-a-socket"
        runtime_dir.mkdir()
        (runtime_dir / "daemon.sock").write_text("not a socket")
        result = self.run_attach(runtime_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no daemon socket", result.stderr)


if __name__ == "__main__":
    unittest.main()
