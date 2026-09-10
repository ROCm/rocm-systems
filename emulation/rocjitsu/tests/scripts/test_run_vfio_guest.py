#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/run-vfio-guest.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import tempfile
import time
from typing import Any
import unittest
from unittest import mock

SCRIPT = Path(__file__).parents[2] / "scripts" / "run-vfio-guest.py"
SPEC = importlib.util.spec_from_file_location("run_vfio_guest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def write_file(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.write_bytes(payload)
    path.chmod(mode)


def fake_qemu(path: Path, delay: float = 0.05) -> None:
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "import time\n"
            "if sys.argv[1:3] == ['-accel', 'help']:\n"
            "    print('Accelerators supported in QEMU binary:')\n"
            "    print('tcg')\n"
            "else:\n"
            "    print('workload: PASS', flush=True)\n"
            f"    time.sleep({delay!r})\n"
        ).encode(),
        0o755,
    )


def fake_server(path: Path, behavior: str) -> None:
    action = {
        "clean": (
            "stopping = False\n"
            "def stop(_signal, _frame):\n"
            "    global stopping\n"
            "    stopping = True\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "while not stopping:\n"
            "    time.sleep(0.01)\n"
        ),
        "fail": "time.sleep(0.08)\nraise SystemExit(23)\n",
        "fail_on_stop": (
            "def fail(_signal, _frame):\n"
            "    raise SystemExit(23)\n"
            "signal.signal(signal.SIGTERM, fail)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        ),
        "ignore": (
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True:\n"
            "    time.sleep(1)\n"
        ),
    }[behavior]
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import signal\n"
            "import sys\n"
            "import time\n"
            "socket_path = sys.argv[sys.argv.index('--vfio-socket') + 1]\n"
            "Path(socket_path).touch()\n" + action
        ).encode(),
        0o755,
    )


def wait_for_fake_socket(path: Path, process: Any, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            raise RUNNER.GuestRunError("fake server exited before readiness")
        time.sleep(0.005)
    raise RUNNER.GuestRunError("fake server did not become ready")


class RunVfioGuestTest(unittest.TestCase):
    def run_fake_guest(
        self,
        root: Path,
        server_behavior: str,
        qemu_delay: float = 0.05,
    ) -> tuple[int, str]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu, qemu_delay)
        fake_server(rocjitsu, server_behavior)

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics), mock.patch.object(
            RUNNER, "wait_for_socket", side_effect=wait_for_fake_socket
        ):
            status = RUNNER.main(
                [
                    "--kernel",
                    str(kernel),
                    "--initramfs",
                    str(initramfs),
                    "--qemu",
                    str(qemu),
                    "--rocjitsu",
                    str(rocjitsu),
                    "--config",
                    str(config),
                    "--output",
                    str(root / "run"),
                    "--accel",
                    "tcg",
                    "--expect-log",
                    "workload: PASS",
                ]
            )
        return status, diagnostics.getvalue()

    def test_auto_acceleration_falls_back_to_tcg(self) -> None:
        self.assertEqual(
            RUNNER.select_accelerator(
                "auto", {"kvm", "tcg"}, Path("/definitely/missing/kvm")
            ),
            "tcg",
        )

    def test_timeouts_must_be_positive_and_finite(self) -> None:
        required = [
            "--kernel",
            "kernel",
            "--initramfs",
            "initramfs",
            "--qemu",
            "qemu",
            "--rocjitsu",
            "rocjitsu",
            "--config",
            "config",
            "--output",
            "output",
        ]
        for option in ("--timeout", "--socket-timeout"):
            for value in ("nan", "inf", "-inf", "0", "-1"):
                with self.subTest(
                    option=option, value=value
                ), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        RUNNER.parse_arguments([*required, option, value])

    def test_qemu_arguments_preserve_caller_policy(self) -> None:
        arguments = RUNNER.build_qemu_arguments(
            Path("/qemu"),
            Path("/guest/vmlinuz"),
            Path("/guest/initramfs.gz"),
            Path("/tmp/device.sock"),
            "tcg",
            "3G",
            "console=ttyS0 rdinit=/init workload=gemm",
        )

        self.assertEqual(arguments[arguments.index("-cpu") + 1], "qemu64,+hypervisor")
        self.assertEqual(arguments[arguments.index("-m") + 1], "3G")
        self.assertIn("memory-backend-memfd,id=mem,size=3G,share=on", arguments)
        self.assertEqual(
            arguments[arguments.index("-append") + 1],
            "console=ttyS0 rdinit=/init workload=gemm",
        )
        device = json.loads(arguments[arguments.index("-device") + 1])
        self.assertEqual(device["driver"], "vfio-user-pci")
        self.assertEqual(device["rombar"], 0)
        self.assertEqual(device["socket"]["path"], "/tmp/device.sock")

    def test_log_checks_are_generic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "guest.log"
            log.write_text("kernel booted\nworkload: PASS\n", encoding="utf-8")
            RUNNER.check_guest_log(log, ["workload: PASS"], ["workload: FAIL"])
            with self.assertRaisesRegex(RUNNER.GuestRunError, "rejected text"):
                RUNNER.check_guest_log(log, [], ["kernel booted"])

    def test_dry_run_prints_commands_without_persisting_input_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            write_file(
                qemu,
                b"#!/bin/sh\nprintf 'Accelerators supported in QEMU binary:\\ntcg\\n'\n",
                0o755,
            )
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o755)
            output = root / "run"

            diagnostics = io.StringIO()
            commands = io.StringIO()
            with contextlib.redirect_stderr(diagnostics), contextlib.redirect_stdout(
                commands
            ):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--append",
                        "console=ttyS0 test=1",
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0, diagnostics.getvalue())
            self.assertIn("--vfio-socket", commands.getvalue())
            self.assertIn("console=ttyS0 test=1", commands.getvalue())
            self.assertFalse((output / "run-manifest.json").exists())
            self.assertFalse(output.exists())

    def test_preexisting_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            write_file(qemu, b"#!/bin/sh\nexit 0\n", 0o755)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o755)
            output = root / "run"
            output.mkdir()
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("run output directory already exists", diagnostics.getvalue())

    def test_socket_must_be_owned_by_the_run_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kernel = root / "vmlinuz"
            initramfs = root / "initramfs.gz"
            config = root / "config.json"
            qemu = root / "qemu"
            rocjitsu = root / "rocjitsu"
            write_file(kernel, b"kernel")
            write_file(initramfs, b"initramfs")
            write_file(config, b"{}\n")
            write_file(qemu, b"#!/bin/sh\nexit 0\n", 0o755)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o755)
            external_socket = root / "unrelated.sock"
            external_socket.write_text("owned by another service", encoding="utf-8")
            diagnostics = io.StringIO()
            with contextlib.redirect_stderr(diagnostics):
                status = RUNNER.main(
                    [
                        "--kernel",
                        str(kernel),
                        "--initramfs",
                        str(initramfs),
                        "--qemu",
                        str(qemu),
                        "--rocjitsu",
                        str(rocjitsu),
                        "--config",
                        str(config),
                        "--output",
                        str(root / "run"),
                        "--socket",
                        str(external_socket),
                        "--dry-run",
                    ]
                )
            self.assertEqual(status, 1)
            self.assertIn("socket must be directly inside", diagnostics.getvalue())
            self.assertEqual(
                external_socket.read_text(encoding="utf-8"),
                "owned by another service",
            )

    def test_cleanup_does_not_unlink_a_replacement_socket(self) -> None:
        socket_path = Path("/run/vfio-user.sock")
        replacement = mock.Mock(st_mode=stat.S_IFSOCK, st_dev=11, st_ino=23)
        with mock.patch.object(
            Path, "lstat", return_value=replacement
        ), mock.patch.object(Path, "unlink") as unlink:
            RUNNER.remove_owned_socket(socket_path, (11, 22))
        unlink.assert_not_called()

    def test_server_failure_is_not_masked_by_successful_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "fail", qemu_delay=0.2)
            self.assertIn(
                "workload: PASS",
                (root / "run/guest.log").read_text(encoding="utf-8"),
            )

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_nonzero_server_shutdown_fails_after_qemu_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics = self.run_fake_guest(Path(temporary), "fail_on_stop")

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_clean_server_shutdown_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "clean")
            self.assertEqual(stat.S_IMODE((root / "run").stat().st_mode), 0o700)

        self.assertEqual(status, 0, diagnostics)

    def test_server_that_requires_forced_kill_fails_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            RUNNER, "PROCESS_SHUTDOWN_TIMEOUT", 0.1
        ):
            status, diagnostics = self.run_fake_guest(Path(temporary), "ignore")

        self.assertEqual(status, 1)
        self.assertIn("did not exit after SIGTERM and was killed", diagnostics)


if __name__ == "__main__":
    unittest.main()
