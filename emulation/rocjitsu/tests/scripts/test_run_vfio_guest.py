#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/run-vfio-guest.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
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


def fake_qemu(
    path: Path,
    wait_for_signal: bool = False,
    returncode: int = 0,
    output: str = "workload: PASS",
    supports_vfio: bool = True,
    notify_server: bool = True,
) -> None:
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import signal\n"
            "import sys\n"
            "if sys.argv[1:3] == ['-accel', 'help']:\n"
            "    print('Accelerators supported in QEMU binary:')\n"
            "    print('tcg')\n"
            "elif sys.argv[1:3] == ['-device', 'help']:\n"
            f"    print('name \\\"vfio-user-pci\\\"') if {supports_vfio!r} else None\n"
            "else:\n"
            "    Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
            f"    print({output!r}, flush=True)\n"
            f"    if {wait_for_signal!r}:\n"
            f"        if {notify_server!r}:\n"
            "            server_pid = int(Path(__file__).with_name('rocjitsu.pid').read_text())\n"
            "            os.kill(server_pid, signal.SIGUSR1)\n"
            "        signal.pause()\n"
            f"    raise SystemExit({returncode})\n"
        ).encode(),
        0o755,
    )


def fake_server(path: Path, behavior: str, supports_vfio: bool = True) -> None:
    actions = {
        "clean": "signal.sigwait({signal.SIGTERM})\n",
        "fail": "signal.sigwait({signal.SIGUSR1})\nraise SystemExit(23)\n",
        "exit_clean": "signal.sigwait({signal.SIGUSR1})\n",
        "fail_on_stop": ("signal.sigwait({signal.SIGTERM})\n" "raise SystemExit(23)\n"),
    }
    if behavior == "startup_fail":
        startup = "raise SystemExit(23)\n"
        action = ""
    elif behavior in {"ignore_sigterm", "stall"}:
        startup = (
            (
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                if behavior == "ignore_sigterm"
                else "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})\n"
            )
            + "socket_path = sys.argv[sys.argv.index('--vfio-socket') + 1]\n"
            "ready_fd = int(sys.argv[sys.argv.index('--vfio-ready-fd') + 1])\n"
            "listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "listener.bind(socket_path)\n"
            "listener.listen()\n"
            "Path(__file__).with_suffix('.bound').write_text('bound')\n"
        )
        action = (
            "while True:\n    signal.pause()\n"
            if behavior == "ignore_sigterm"
            else "signal.sigwait({signal.SIGTERM})\nos.close(ready_fd)\n"
        )
    else:
        startup = (
            (
                "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})\n"
                if behavior in {"fail", "exit_clean"}
                else "signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})\n"
            )
            + "socket_path = sys.argv[sys.argv.index('--vfio-socket') + 1]\n"
            "ready_fd = int(sys.argv[sys.argv.index('--vfio-ready-fd') + 1])\n"
            "listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "listener.bind(socket_path)\n"
            "listener.listen()\n"
            "os.write(ready_fd, b'\\x01')\n"
            "os.close(ready_fd)\n"
        )
        action = actions[behavior]
    write_file(
        path,
        (
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            "import signal\n"
            "import socket\n"
            "import sys\n"
            "if sys.argv[1:] == ['--check-vfio-user']:\n"
            f"    if {supports_vfio!r}:\n"
            "        print('vfio-user support enabled')\n"
            "        raise SystemExit(0)\n"
            "    print('this build has no vfio-user support', file=sys.stderr)\n"
            "    raise SystemExit(1)\n"
            "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
            + startup
            + action
        ).encode(),
        0o755,
    )


@unittest.skipUnless(sys.platform.startswith("linux"), "VFIO launcher is Linux-only")
class RunVfioGuestTest(unittest.TestCase):
    def run_fake_guest(
        self,
        root: Path,
        server_behavior: str,
        qemu_wait_for_signal: bool = False,
        qemu_returncode: int = 0,
        qemu_output: str = "workload: PASS",
        startup_timeout: float | None = None,
    ) -> tuple[int, str]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu, qemu_wait_for_signal, qemu_returncode, qemu_output)
        fake_server(rocjitsu, server_behavior)

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            arguments = [
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
            if startup_timeout is not None:
                arguments.extend(["--startup-timeout", str(startup_timeout)])
            status = RUNNER.main(arguments)
        return status, diagnostics.getvalue()

    def assert_fake_processes_reaped(self, root: Path) -> None:
        for name in ("qemu", "rocjitsu"):
            self.assert_fake_process_reaped(root, name)

    def assert_fake_process_reaped(self, root: Path, name: str) -> None:
        pid_file = (root / name).with_suffix(".pid")
        self.assertTrue(pid_file.is_file(), f"{name} never recorded its pid")
        pid = int(pid_file.read_text(encoding="utf-8"))
        with self.assertRaises(ChildProcessError, msg=f"{name} was not reaped"):
            os.waitpid(pid, os.WNOHANG)

    def assert_process_gone(self, pid: int) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.01)
        self.fail(f"process {pid} is still alive")

    def run_spawn_window_interrupt(
        self, root: Path, target: str
    ) -> tuple[int, str, bool]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu, wait_for_signal=True, notify_server=False)
        fake_server(rocjitsu, "clean")

        arguments = [
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
        real_popen = subprocess.Popen
        spawned: subprocess.Popen[str] | None = None
        interrupted = False

        def popen_with_interrupt(
            command: list[str], *popen_args: object, **popen_kwargs: object
        ) -> subprocess.Popen[str]:
            nonlocal interrupted, spawned
            process = real_popen(command, *popen_args, **popen_kwargs)
            is_target = (target == "server" and "--vfio-ready-fd" in command) or (
                target == "qemu" and "-kernel" in command
            )
            if is_target and not interrupted:
                interrupted = True
                spawned = process
                os.kill(os.getpid(), signal.SIGTERM)
            return process

        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stderr(diagnostics), mock.patch.object(
                RUNNER.subprocess, "Popen", side_effect=popen_with_interrupt
            ):
                status = RUNNER.main(arguments)
            self.assertTrue(interrupted, f"did not intercept the {target} spawn")
            self.assertIsNotNone(spawned)
            reaped = spawned.returncode is not None
        finally:
            if spawned is not None and spawned.returncode is None:
                spawned.kill()
                spawned.wait()
        return status, diagnostics.getvalue(), reaped

    def run_cleanup_window_interrupt(
        self, root: Path, target: str
    ) -> tuple[int, str, bool]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu, wait_for_signal=target == "qemu")
        fake_server(rocjitsu, "fail" if target == "qemu" else "clean")

        arguments = [
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
        interrupted = False
        real_kill = subprocess.Popen.kill
        real_send_signal = subprocess.Popen.send_signal

        def kill_with_interrupt(process: subprocess.Popen[str]) -> None:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGTERM)
            real_kill(process)

        def send_signal_with_interrupt(
            process: subprocess.Popen[str], signum: int
        ) -> None:
            nonlocal interrupted
            if signum == signal.SIGTERM and not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGTERM)
            real_send_signal(process, signum)

        patcher = mock.patch.object(
            RUNNER.subprocess.Popen,
            "kill" if target == "qemu" else "send_signal",
            new=kill_with_interrupt if target == "qemu" else send_signal_with_interrupt,
        )
        diagnostics = io.StringIO()
        try:
            with contextlib.redirect_stderr(diagnostics), patcher:
                status = RUNNER.main(arguments)
            self.assertTrue(interrupted, f"did not intercept {target} cleanup")
            self.assert_fake_processes_reaped(root)
            socket_removed = not (root / "run/vfio-user.sock").exists()
        finally:
            for executable in (qemu, rocjitsu):
                pid_file = executable.with_suffix(".pid")
                if not pid_file.is_file():
                    continue
                pid = int(pid_file.read_text(encoding="utf-8"))
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
        return status, diagnostics.getvalue(), socket_removed

    def run_readiness_close_interrupt(self, root: Path) -> tuple[int, str, bool]:
        kernel = root / "vmlinuz"
        initramfs = root / "initramfs.gz"
        config = root / "config.json"
        qemu = root / "qemu"
        rocjitsu = root / "rocjitsu"
        write_file(kernel, b"kernel")
        write_file(initramfs, b"initramfs")
        write_file(config, b"{}\n")
        fake_qemu(qemu)
        fake_server(rocjitsu, "clean")

        arguments = [
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
        real_wait_for_server_ready = RUNNER.wait_for_server_ready
        real_close = os.close
        ready_fd: int | None = None
        interrupted = False

        def wait_for_server_ready_with_capture(
            path: Path,
            process: subprocess.Popen[str],
            fd: int,
            timeout: float,
        ) -> tuple[int, int]:
            nonlocal ready_fd
            ready_fd = fd
            return real_wait_for_server_ready(path, process, fd, timeout)

        def close_with_interrupt(fd: int) -> None:
            nonlocal interrupted
            real_close(fd)
            if fd == ready_fd and not interrupted:
                interrupted = True
                os.kill(os.getpid(), signal.SIGTERM)

        diagnostics = io.StringIO()
        with contextlib.redirect_stderr(diagnostics), mock.patch.object(
            RUNNER,
            "wait_for_server_ready",
            side_effect=wait_for_server_ready_with_capture,
        ), mock.patch.object(RUNNER.os, "close", side_effect=close_with_interrupt):
            status = RUNNER.main(arguments)

        self.assertTrue(interrupted, "did not interrupt readiness descriptor close")
        self.assert_fake_process_reaped(root, "rocjitsu")
        self.assertFalse((root / "qemu.pid").exists())
        return (
            status,
            diagnostics.getvalue(),
            not (root / "run/vfio-user.sock").exists(),
        )

    def test_auto_acceleration_falls_back_to_tcg(self) -> None:
        self.assertEqual(
            RUNNER.select_accelerator(
                "auto", {"kvm", "tcg"}, Path("/definitely/missing/kvm")
            ),
            "tcg",
        )

    def test_timeout_options_are_not_supported(self) -> None:
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
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    RUNNER.parse_arguments([*required, option, "12.5"])

    def test_startup_timeout_must_be_positive_and_finite(self) -> None:
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
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    RUNNER.parse_arguments([*required, "--startup-timeout", value])

    def test_probe_timeout_must_be_positive_and_finite(self) -> None:
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
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    RUNNER.parse_arguments([*required, "--probe-timeout", value])

    def test_capability_probe_timeout_kills_and_reaps_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            write_file(
                probe,
                (
                    "#!/usr/bin/env python3\n"
                    "import os\n"
                    "from pathlib import Path\n"
                    "import signal\n"
                    "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
                    "signal.pause()\n"
                ).encode(),
                0o755,
            )

            with self.assertRaisesRegex(
                RUNNER.GuestRunError, "test probe did not complete within 0.05 seconds"
            ):
                RUNNER.run_capability_probe([str(probe)], "test probe", 0.05)
            self.assert_fake_process_reaped(root, "probe")

    def test_sigterm_in_capability_probe_spawn_window_reaps_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            write_file(
                probe,
                b"#!/usr/bin/env python3\nimport signal\nsignal.pause()\n",
                0o755,
            )
            real_popen = subprocess.Popen
            spawned: subprocess.Popen[str] | None = None

            def popen_with_interrupt(
                command: list[str], *popen_args: object, **popen_kwargs: object
            ) -> subprocess.Popen[str]:
                nonlocal spawned
                spawned = real_popen(command, *popen_args, **popen_kwargs)
                os.kill(os.getpid(), signal.SIGTERM)
                return spawned

            try:
                with mock.patch.object(
                    RUNNER.subprocess, "Popen", side_effect=popen_with_interrupt
                ), self.assertRaisesRegex(
                    RUNNER.GuestRunError, "launcher received SIGTERM"
                ):
                    RUNNER.run_capability_probe([str(probe)], "test probe", 5)
                self.assertIsNotNone(spawned)
                self.assertIsNotNone(spawned.returncode)
            finally:
                if spawned is not None and spawned.returncode is None:
                    spawned.kill()
                    spawned.wait()

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
        self.assertEqual(device["socket"], {"type": "unix", "path": "/tmp/device.sock"})

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
            fake_qemu(qemu)
            fake_server(rocjitsu, "clean")
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
            self.assertNotIn("--vfio-ready-fd", commands.getvalue())
            self.assertIn("console=ttyS0 test=1", commands.getvalue())
            self.assertFalse((output / "run-manifest.json").exists())
            self.assertFalse(output.exists())

    def test_dry_run_rejects_a_non_executable_server(self) -> None:
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
            fake_qemu(qemu)
            write_file(rocjitsu, b"#!/bin/sh\nexit 0\n", 0o644)

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
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("rocjitsu is not executable", diagnostics.getvalue())

    def test_dry_run_rejects_qemu_without_vfio_user_pci(self) -> None:
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
            fake_qemu(qemu, supports_vfio=False)
            fake_server(rocjitsu, "clean")

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
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn(
                "does not provide the vfio-user-pci device", diagnostics.getvalue()
            )

    def test_dry_run_rejects_rocjitsu_without_vfio_user_support(self) -> None:
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
            fake_qemu(qemu)
            fake_server(rocjitsu, "clean", supports_vfio=False)

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
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 1)
            self.assertIn("no usable vfio-user support", diagnostics.getvalue())

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

    def test_readiness_signal_observes_a_real_unix_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(socket_path))
                listener.listen()
                ready_read, ready_write = os.pipe()
                try:
                    os.write(ready_write, b"\x01")
                    identity = RUNNER.wait_for_server_ready(
                        socket_path,
                        mock.Mock(poll=mock.Mock(return_value=None)),
                        ready_read,
                    )
                finally:
                    os.close(ready_write)
                    os.close(ready_read)

            socket_stat = socket_path.lstat()
            self.assertEqual(identity, (socket_stat.st_dev, socket_stat.st_ino))

    def test_readiness_pipe_eof_reports_startup_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            ready_read, ready_write = os.pipe()
            os.close(ready_write)
            try:
                with self.assertRaisesRegex(
                    RUNNER.GuestRunError, "closed its readiness channel"
                ):
                    RUNNER.wait_for_server_ready(
                        socket_path,
                        mock.Mock(poll=mock.Mock(return_value=23)),
                        ready_read,
                    )
            finally:
                os.close(ready_read)

    def test_cleanup_removes_only_the_observed_real_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "vfio-user.sock"
            with socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as original, socket.socket(
                socket.AF_UNIX, socket.SOCK_STREAM
            ) as replacement:
                original.bind(str(socket_path))
                original_identity = (
                    socket_path.lstat().st_dev,
                    socket_path.lstat().st_ino,
                )
                socket_path.unlink()
                replacement.bind(str(socket_path))
                replacement_identity = (
                    socket_path.lstat().st_dev,
                    socket_path.lstat().st_ino,
                )
                RUNNER.remove_owned_socket(socket_path, original_identity)
                self.assertTrue(socket_path.exists())
                RUNNER.remove_owned_socket(socket_path, replacement_identity)
                self.assertFalse(socket_path.exists())

    def test_server_failure_is_not_masked_by_successful_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "fail", qemu_wait_for_signal=True
            )
            self.assertIn(
                "workload: PASS",
                (root / "run/guest.log").read_text(encoding="utf-8"),
            )
            self.assert_fake_processes_reaped(root)
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_server_startup_failure_does_not_start_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "startup_fail")
            self.assert_fake_process_reaped(root, "rocjitsu")
            self.assertFalse((root / "qemu.pid").exists())
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("closed its readiness channel", diagnostics)

    def test_server_startup_timeout_stops_server_without_starting_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bound_marker = root / "rocjitsu.bound"
            real_wait_for_server_ready = RUNNER.wait_for_server_ready

            def wait_after_server_bound(
                path: Path,
                process: subprocess.Popen[bytes],
                ready_fd: int,
                timeout: float,
            ) -> tuple[int, int]:
                deadline = time.monotonic() + 5
                while not bound_marker.is_file():
                    self.assertIsNone(
                        process.poll(), "fake rocjitsu exited before binding its socket"
                    )
                    if time.monotonic() >= deadline:
                        self.fail("fake rocjitsu did not bind within 5 seconds")
                    time.sleep(0.01)
                return real_wait_for_server_ready(path, process, ready_fd, timeout)

            with mock.patch.object(
                RUNNER,
                "wait_for_server_ready",
                side_effect=wait_after_server_bound,
            ):
                status, diagnostics = self.run_fake_guest(
                    root, "stall", startup_timeout=0.05
                )
            self.assertTrue(bound_marker.is_file())
            self.assert_fake_process_reaped(root, "rocjitsu")
            self.assertFalse((root / "qemu.pid").exists())
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("did not report readiness within 0.05 seconds", diagnostics)

    def test_server_startup_timeout_force_kills_unresponsive_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(RUNNER, "SERVER_SHUTDOWN_GRACE_SECONDS", 0.05):
                status, diagnostics = self.run_fake_guest(
                    root, "ignore_sigterm", startup_timeout=0.2
                )
            self.assertTrue((root / "rocjitsu.bound").is_file())
            self.assert_fake_process_reaped(root, "rocjitsu")
            self.assertFalse((root / "qemu.pid").exists())
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 1)
        self.assertIn("did not report readiness within 0.2 seconds", diagnostics)
        self.assertIn("killed it with SIGKILL", diagnostics)

    def test_clean_server_exit_while_qemu_runs_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "exit_clean", qemu_wait_for_signal=True
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited while QEMU was running (status 0)", diagnostics)

    def test_nonzero_qemu_exit_fails_and_reaps_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "clean", qemu_returncode=17)
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("QEMU exited with status 17", diagnostics)

    def test_missing_log_marker_fails_and_reaps_both_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "clean", qemu_output="workload finished"
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("guest log is missing expected text", diagnostics)

    def test_guest_failure_is_preserved_when_server_cleanup_also_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(
                root, "fail_on_stop", qemu_returncode=17
            )
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("QEMU exited with status 17", diagnostics)
        self.assertIn(
            "cleanup also failed: rocjitsu exited with status 23", diagnostics
        )

    def test_nonzero_server_shutdown_fails_after_qemu_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "fail_on_stop")
            self.assert_fake_processes_reaped(root)

        self.assertEqual(status, 1)
        self.assertIn("rocjitsu exited with status 23", diagnostics)

    def test_clean_server_shutdown_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status, diagnostics = self.run_fake_guest(root, "clean")
            self.assertEqual(stat.S_IMODE((root / "run").stat().st_mode), 0o700)
            self.assert_fake_processes_reaped(root)
            self.assertFalse((root / "run/vfio-user.sock").exists())

        self.assertEqual(status, 0, diagnostics)

    def test_sigterm_cleans_up_blocked_children_and_owned_socket(self) -> None:
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
            fake_qemu(qemu, wait_for_signal=True, notify_server=False)
            fake_server(rocjitsu, "clean")
            output = root / "run"

            launcher = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT),
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
                    "--accel",
                    "tcg",
                    "--expect-log",
                    "workload: PASS",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if (root / "qemu.pid").is_file() and (root / "rocjitsu.pid").is_file():
                    break
                time.sleep(0.01)
            else:
                launcher.kill()
                launcher.wait()
                self.fail("launcher did not start both children")

            qemu_pid = int((root / "qemu.pid").read_text(encoding="utf-8"))
            server_pid = int((root / "rocjitsu.pid").read_text(encoding="utf-8"))
            launcher.send_signal(signal.SIGTERM)
            _stdout, stderr = launcher.communicate(timeout=10)

            self.assertEqual(launcher.returncode, 1, stderr)
            self.assertIn("launcher received SIGTERM", stderr)
            self.assert_process_gone(qemu_pid)
            self.assert_process_gone(server_pid)
            self.assertFalse((output / "vfio-user.sock").exists())

    def test_sigterm_in_server_spawn_window_reaps_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics, reaped = self.run_spawn_window_interrupt(
                Path(temporary), "server"
            )

        self.assertEqual(status, 1, diagnostics)
        self.assertIn("launcher received SIGTERM", diagnostics)
        self.assertTrue(reaped, "rocjitsu escaped cleanup during its spawn window")

    def test_sigterm_in_qemu_spawn_window_reaps_qemu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics, reaped = self.run_spawn_window_interrupt(
                Path(temporary), "qemu"
            )

        self.assertEqual(status, 1, diagnostics)
        self.assertIn("launcher received SIGTERM", diagnostics)
        self.assertTrue(reaped, "QEMU escaped cleanup during its spawn window")

    def test_sigterm_after_readiness_close_reaps_server_and_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics, socket_removed = self.run_readiness_close_interrupt(
                Path(temporary)
            )

        self.assertEqual(status, 1, diagnostics)
        self.assertIn("launcher received SIGTERM", diagnostics)
        self.assertNotIn("readiness cleanup also failed", diagnostics)
        self.assertTrue(socket_removed)

    def test_sigterm_at_qemu_cleanup_entry_reaps_both_children(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics, socket_removed = self.run_cleanup_window_interrupt(
                Path(temporary), "qemu"
            )

        self.assertEqual(status, 1, diagnostics)
        self.assertIn("launcher received SIGTERM", diagnostics)
        self.assertTrue(socket_removed)

    def test_sigterm_at_server_cleanup_entry_reaps_server_and_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status, diagnostics, socket_removed = self.run_cleanup_window_interrupt(
                Path(temporary), "server"
            )

        self.assertEqual(status, 1, diagnostics)
        self.assertIn("launcher received SIGTERM", diagnostics)
        self.assertTrue(socket_removed)


if __name__ == "__main__":
    unittest.main()
