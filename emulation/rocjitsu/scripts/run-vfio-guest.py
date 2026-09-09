#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a prepared Linux guest against rocjitsu's vfio-user device."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import shlex
import stat
import subprocess
import sys
import time
from typing import Any

PROCESS_POLL_INTERVAL = 0.05
PROCESS_SHUTDOWN_TIMEOUT = 5.0
SocketIdentity = tuple[int, int]


class GuestRunError(ValueError):
    """A QEMU or vfio-user launch requirement was not satisfied."""


def positive_finite_seconds(value: str) -> float:
    """Parse a finite, positive command timeout."""

    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid timeout: {value}") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and greater than zero")
    return seconds


def require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise GuestRunError(f"{description} is not a file: {path}")
    return resolved


def available_accelerators(qemu: Path) -> set[str]:
    result = subprocess.run(
        [str(qemu), "-accel", "help"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuestRunError(f"cannot query QEMU accelerators: {detail}")
    return {
        line.strip().split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("Accelerators supported")
    }


def select_accelerator(
    requested: str, accelerators: set[str], kvm_device: Path = Path("/dev/kvm")
) -> str:
    if requested == "kvm":
        if "kvm" not in accelerators:
            raise GuestRunError("QEMU does not support KVM acceleration")
        if not os.access(kvm_device, os.R_OK | os.W_OK):
            raise GuestRunError(f"KVM is not usable: {kvm_device}")
        return "kvm"
    if requested == "tcg":
        if "tcg" not in accelerators:
            raise GuestRunError("QEMU does not support TCG acceleration")
        return "tcg"
    if requested != "auto":
        raise GuestRunError(f"unknown accelerator: {requested}")
    if "kvm" in accelerators and os.access(kvm_device, os.R_OK | os.W_OK):
        return "kvm"
    if "tcg" not in accelerators:
        raise GuestRunError("KVM is unavailable and QEMU does not support TCG")
    return "tcg"


def build_qemu_arguments(
    qemu: Path,
    kernel: Path,
    initramfs: Path,
    socket_path: Path,
    accelerator: str,
    memory: str,
    append: str,
) -> list[str]:
    cpu = "host,+hypervisor" if accelerator == "kvm" else "qemu64,+hypervisor"
    return [
        str(qemu),
        "-accel",
        accelerator,
        "-cpu",
        cpu,
        "-m",
        memory,
        "-object",
        f"memory-backend-memfd,id=mem,size={memory},share=on",
        "-machine",
        "q35,memory-backend=mem",
        "-kernel",
        str(kernel),
        "-initrd",
        str(initramfs),
        "-append",
        append,
        "-device",
        json.dumps(
            {
                "driver": "vfio-user-pci",
                "rombar": 0,
                "socket": {"path": str(socket_path), "type": "unix"},
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        "-nodefaults",
        "-no-user-config",
        "-display",
        "none",
        "-monitor",
        "none",
        "-serial",
        "stdio",
        "-no-reboot",
    ]


def wait_for_socket(
    path: Path, process: subprocess.Popen[Any], timeout: float
) -> SocketIdentity:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise GuestRunError(
                "rocjitsu exited before creating the vfio-user socket "
                f"({process.returncode})"
            )
        try:
            socket_stat = path.lstat()
            if not stat.S_ISSOCK(socket_stat.st_mode):
                raise GuestRunError(f"vfio-user socket path is not a socket: {path}")
            return socket_stat.st_dev, socket_stat.st_ino
        except FileNotFoundError:
            pass
        except OSError as error:
            raise GuestRunError(f"cannot inspect vfio-user socket: {error}") from error
        time.sleep(PROCESS_POLL_INTERVAL)
    raise GuestRunError(f"rocjitsu did not create {path} within {timeout:g} seconds")


def remove_owned_socket(path: Path, identity: SocketIdentity | None) -> None:
    """Remove only the socket observed after this invocation started its server."""

    if identity is None:
        return
    try:
        socket_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(socket_stat.st_mode):
        return
    if (socket_stat.st_dev, socket_stat.st_ino) == identity:
        path.unlink()


def stop_after_failure(process: subprocess.Popen[Any]) -> None:
    """Best-effort cleanup for a launch that has already failed."""

    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)


def finish_server(process: subprocess.Popen[Any]) -> None:
    """Stop rocjitsu if needed and require a clean, cooperative exit."""

    status = process.poll()
    if status is None:
        try:
            process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            status = process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait(timeout=PROCESS_SHUTDOWN_TIMEOUT)
            raise GuestRunError(
                "rocjitsu did not exit after SIGTERM and was killed"
            ) from error
    if status != 0:
        raise GuestRunError(f"rocjitsu exited with status {status}")


def run_qemu(
    arguments: list[str],
    server: subprocess.Popen[Any],
    guest_log: Any,
    timeout: float,
) -> int:
    """Run QEMU while continuously checking the vfio-user server."""

    guest = subprocess.Popen(
        arguments,
        stdout=guest_log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout
    try:
        while True:
            server_status = server.poll()
            if server_status is not None and server_status != 0:
                raise GuestRunError(
                    "rocjitsu exited while QEMU was running "
                    f"(status {server_status})"
                )

            guest_status = guest.poll()
            if guest_status is not None:
                return guest_status

            if time.monotonic() >= deadline:
                raise GuestRunError(f"QEMU did not exit within {timeout:g} seconds")
            time.sleep(PROCESS_POLL_INTERVAL)
    finally:
        stop_after_failure(guest)


def check_guest_log(path: Path, expected: list[str], rejected: list[str]) -> None:
    log = path.read_text(encoding="utf-8", errors="replace")
    for text in expected:
        if text not in log:
            raise GuestRunError(f"guest log is missing expected text: {text}")
    for text in rejected:
        if text in log:
            raise GuestRunError(f"guest log contains rejected text: {text}")


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--initramfs", required=True, type=Path)
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--rocjitsu", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--accel", choices=("auto", "kvm", "tcg"), default="auto")
    parser.add_argument("--memory", default="4G")
    parser.add_argument("--append", default="console=ttyS0 panic=-1")
    parser.add_argument("--timeout", type=positive_finite_seconds, default=180.0)
    parser.add_argument("--socket-timeout", type=positive_finite_seconds, default=10.0)
    parser.add_argument("--expect-log", action="append", default=[])
    parser.add_argument("--reject-log", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        kernel = require_file(args.kernel, "guest kernel")
        initramfs = require_file(args.initramfs, "guest initramfs")
        qemu = require_file(args.qemu, "QEMU")
        rocjitsu = require_file(args.rocjitsu, "rocjitsu")
        config = require_file(args.config, "rocjitsu config")
        output = args.output.resolve()
        if output.exists():
            raise GuestRunError(f"run output directory already exists: {output}")

        socket_path = (args.socket or output / "vfio-user.sock").resolve()
        if socket_path.parent != output:
            raise GuestRunError(
                "vfio-user socket must be directly inside the run output directory"
            )
        if len(str(socket_path).encode("utf-8")) >= 108:
            raise GuestRunError(f"vfio-user socket path is too long: {socket_path}")
        try:
            socket_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise GuestRunError(f"vfio-user socket path already exists: {socket_path}")

        accelerator = select_accelerator(args.accel, available_accelerators(qemu))
        qemu_arguments = build_qemu_arguments(
            qemu,
            kernel,
            initramfs,
            socket_path,
            accelerator,
            args.memory,
            args.append,
        )
        server_arguments = [
            str(rocjitsu),
            "--config",
            str(config),
            "--vfio-socket",
            str(socket_path),
        ]
        if args.dry_run:
            print(shlex.join(server_arguments))
            print(shlex.join(qemu_arguments))
            return 0

        output.mkdir(mode=0o700, parents=True)
        server_log_path = output / "server.log"
        guest_log_path = output / "guest.log"
        with server_log_path.open("wb") as server_log:
            server = subprocess.Popen(
                server_arguments,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
        run_error: GuestRunError | OSError | subprocess.SubprocessError | None = None
        socket_identity: SocketIdentity | None = None
        try:
            socket_identity = wait_for_socket(socket_path, server, args.socket_timeout)
            with guest_log_path.open("wb") as guest_log:
                guest_returncode = run_qemu(
                    qemu_arguments, server, guest_log, args.timeout
                )
            if guest_returncode != 0:
                raise GuestRunError(f"QEMU exited with status {guest_returncode}")
            check_guest_log(guest_log_path, args.expect_log, args.reject_log)
        except (GuestRunError, OSError, subprocess.SubprocessError) as error:
            run_error = error
        finally:
            try:
                finish_server(server)
            except (GuestRunError, OSError, subprocess.SubprocessError) as error:
                run_error = error
            remove_owned_socket(socket_path, socket_identity)

        if run_error is not None:
            raise run_error

        return 0
    except (GuestRunError, OSError, subprocess.SubprocessError) as error:
        print(f"vfio guest launch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
