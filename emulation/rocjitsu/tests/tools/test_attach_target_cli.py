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

    def run_attach(self, target, *extra_args, script=None, cwd=None):
        if script is None:
            script = (
                "import os; "
                "print(os.environ.get('ROCJITSU_INVOCATION_DIR', '')); "
                "print(os.environ.get('ROCJITSU_REQUIRE_SELECTED_DAEMON', ''))"
            )
        return subprocess.run(
            [
                self.cli,
                "--attach",
                str(target),
                *extra_args,
                "--",
                sys.executable,
                "-c",
                script,
            ],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
            cwd=cwd,
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
                lines = result.stdout.splitlines()
                self.assertEqual(lines, [str(runtime_dir), "1"])

    def test_clear_env_keeps_selected_runtime(self):
        self.bind_daemon_socket(self.runtime)
        self.env["RJ_SHOULD_NOT_LEAK"] = "bad"
        result = self.run_attach(
            "default",
            "--clear-env-vars",
            script=(
                "import os; "
                "print(os.environ.get('ROCJITSU_INVOCATION_DIR', '')); "
                "print(os.environ.get('ROCJITSU_REQUIRE_SELECTED_DAEMON', '')); "
                "raise SystemExit(0 if 'RJ_SHOULD_NOT_LEAK' not in os.environ else 1)"
            ),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines, [str(self.runtime), "1"])

    def test_child_xdg_override_selects_child_runtime(self):
        self.env.pop("ROCJITSU_RUNTIME_DIR", None)
        parent_xdg = self.root / "parent-xdg"
        child_xdg = self.root / "child-xdg"
        self.env["XDG_RUNTIME_DIR"] = str(parent_xdg)
        child_runtime = child_xdg / "rocjitsu"
        self.bind_daemon_socket(child_runtime)
        result = self.run_attach("default", "--env", f"XDG_RUNTIME_DIR={child_xdg}")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines, [str(child_runtime), "1"])

    def test_empty_runtime_override_falls_through_to_child_xdg(self):
        child_xdg = self.root / "empty-override-xdg"
        child_runtime = child_xdg / "rocjitsu"
        self.bind_daemon_socket(child_runtime)
        result = self.run_attach(
            "default",
            "--env",
            "ROCJITSU_RUNTIME_DIR=",
            "--env",
            f"XDG_RUNTIME_DIR={child_xdg}",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines, [str(child_runtime), "1"])

    def test_parent_relative_runtime_survives_workdir(self):
        launch_cwd = self.root / "launch-cwd"
        workdir = self.root / "workdir"
        launch_cwd.mkdir()
        workdir.mkdir()
        self.env["ROCJITSU_RUNTIME_DIR"] = "r"
        selected_runtime = launch_cwd / "r"
        self.bind_daemon_socket(selected_runtime)
        result = self.run_attach(
            "default",
            "--workdir",
            str(workdir),
            cwd=launch_cwd,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines, [str(selected_runtime), "1"])

    def test_child_relative_runtime_override_survives_workdir(self):
        launch_cwd = self.root / "child-launch-cwd"
        workdir = self.root / "child-workdir"
        launch_cwd.mkdir()
        workdir.mkdir()
        selected_runtime = launch_cwd / "r"
        self.bind_daemon_socket(selected_runtime)
        result = self.run_attach(
            "default",
            "--env",
            "ROCJITSU_RUNTIME_DIR=r",
            "--workdir",
            str(workdir),
            cwd=launch_cwd,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(lines, [str(selected_runtime), "1"])

    def test_rejects_non_socket_target(self):
        runtime_dir = self.root / "not-a-socket"
        runtime_dir.mkdir()
        (runtime_dir / "daemon.sock").write_text("not a socket")
        result = self.run_attach(runtime_dir)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no daemon socket", result.stderr)

    def test_local_launch_uses_reserved_invocation_directory(self):
        read_fd, write_fd = os.pipe()
        child_pid = os.fork()
        if child_pid == 0:
            try:
                os.close(read_fd)
                runtime_dir = self.runtime / str(os.getpid())
                self.bind_daemon_socket(runtime_dir)
                os.dup2(write_fd, 1)
                os.close(write_fd)
                os.execve(
                    self.cli,
                    [
                        self.cli,
                        "--profile",
                        "mi350x",
                        "--",
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('ROCJITSU_INVOCATION_DIR', ''))",
                    ],
                    self.env,
                )
            except BaseException:
                os._exit(127)

        os.close(write_fd)
        with os.fdopen(read_fd, "r", encoding="utf-8") as pipe:
            output = pipe.read()
        _, status = os.waitpid(child_pid, 0)
        self.assertTrue(os.WIFEXITED(status), output)
        self.assertEqual(os.WEXITSTATUS(status), 0, output)
        self.assertEqual(
            output.strip(), str(self.runtime / "invocations" / str(child_pid))
        )
        self.assertTrue((self.runtime / str(child_pid) / "daemon.sock").exists())


if __name__ == "__main__":
    unittest.main()
