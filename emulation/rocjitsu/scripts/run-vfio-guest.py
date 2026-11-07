#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a prepared Linux guest against rocjitsu's vfio-user device."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import shlex
import stat
import subprocess
import sys
from collections.abc import Iterator, Sequence
from typing import Any

SocketIdentity = tuple[int, int]
DEFAULT_PROBE_TIMEOUT_SECONDS = 10.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 60.0
SERVER_SHUTDOWN_GRACE_SECONDS = 5.0
MANAGED_TERMINATION_SIGNALS = frozenset((signal.SIGTERM, signal.SIGINT))


class GuestRunError(ValueError):
    """A QEMU or vfio-user launch requirement was not satisfied."""


@contextlib.contextmanager
def managed_termination_signals() -> Iterator[None]:
    """Turn ordinary launcher termination into cleanup-aware unwinding."""

    previous_handlers: dict[signal.Signals, Any] = {}
    received_signal: signal.Signals | None = None

    def interrupt(signum: int, _frame: Any) -> None:
        nonlocal received_signal
        caught = signal.Signals(signum)
        if received_signal is not None:
            return
        received_signal = caught
        raise GuestRunError(f"launcher received {caught.name}")

    try:
        for caught in MANAGED_TERMINATION_SIGNALS:
            previous_handlers[caught] = signal.signal(caught, interrupt)
        yield
    finally:
        for caught, handler in previous_handlers.items():
            signal.signal(caught, handler)


@contextlib.contextmanager
def block_termination_signals() -> Iterator[set[signal.Signals]]:
    """Defer launcher termination until a spawned child has a cleanup owner."""

    previous_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, MANAGED_TERMINATION_SIGNALS
    )
    try:
        yield previous_mask
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def require_file(path: Path, description: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise GuestRunError(f"{description} is not a file: {path}")
    return resolved


def require_executable(path: Path, description: str) -> Path:
    """Resolve a regular file and require that the launcher can execute it."""

    resolved = require_file(path, description)
    if not os.access(resolved, os.X_OK):
        raise GuestRunError(f"{description} is not executable: {path}")
    return resolved


def run_capability_probe(
    arguments: list[str], description: str, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run one bounded probe and reap it on timeout or launcher termination."""

    process: subprocess.Popen[str] | None = None
    try:
        with managed_termination_signals():
            try:
                with block_termination_signals() as child_signal_mask:

                    def restore_child_signal_mask() -> None:
                        signal.pthread_sigmask(signal.SIG_SETMASK, child_signal_mask)

                    process = subprocess.Popen(
                        arguments,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=restore_child_signal_mask,
                    )
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    raise GuestRunError(
                        f"{description} did not complete within {timeout:g} seconds"
                    ) from error
                return subprocess.CompletedProcess(
                    arguments, process.returncode, stdout, stderr
                )
            finally:
                with block_termination_signals():
                    if process is not None:
                        if process.poll() is None:
                            process.kill()
                        process.wait()
                        if process.stdout is not None:
                            process.stdout.close()
                        if process.stderr is not None:
                            process.stderr.close()
    except OSError as error:
        raise GuestRunError(f"cannot run {description}: {error}") from error


def available_accelerators(qemu: Path, timeout: float) -> set[str]:
    result = run_capability_probe(
        [str(qemu), "-accel", "help"], "QEMU accelerator probe", timeout
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuestRunError(f"cannot query QEMU accelerators: {detail}")
    return {
        line.strip().split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("Accelerators supported")
    }


def require_qemu_vfio_user(qemu: Path, timeout: float) -> None:
    """Require the QEMU device model used by this launcher."""

    result = run_capability_probe(
        [str(qemu), "-device", "help"], "QEMU device probe", timeout
    )
    detail = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise GuestRunError(f"cannot query QEMU devices: {detail.strip()}")
    if "vfio-user-pci" not in detail:
        raise GuestRunError("QEMU does not provide the vfio-user-pci device")


def require_rocjitsu_vfio_user(rocjitsu: Path, timeout: float) -> None:
    """Require a rocjitsu binary built with its VFIO-user front end."""

    result = run_capability_probe(
        [str(rocjitsu), "--check-vfio-user"], "rocjitsu VFIO-user probe", timeout
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise GuestRunError(f"rocjitsu has no usable vfio-user support: {detail}")


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
    socket_paths: Sequence[Path],
    accelerator: str,
    memory: str,
    append: str,
) -> list[str]:
    cpu = "host,+hypervisor" if accelerator == "kvm" else "qemu64,+hypervisor"
    arguments = [
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
    ]
    for ordinal, socket_path in enumerate(socket_paths):
        device: dict[str, Any] = {
            "driver": "vfio-user-pci",
            "rombar": 0,
            "socket": {"type": "unix", "path": str(socket_path)},
        }
        if len(socket_paths) > 1:
            # Keep low slots and the Q35 chipset's upper slots available.
            if ordinal + 4 >= 0x1C:
                raise GuestRunError("too many VFIO-user devices for Q35 PCI slots")
            device.update(
                {
                    "bus": "pcie.0",
                    "addr": f"0x{ordinal + 4:x}",
                    "id": f"rocjitsu-gpu-{ordinal}",
                }
            )
        arguments.extend(
            [
                "-device",
                json.dumps(device, separators=(",", ":"), sort_keys=True),
            ]
        )
    arguments.extend(
        [
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
    )
    return arguments


def start_vfio_server(
    arguments: list[str], server_log: Any, child_signal_mask: set[signal.Signals]
) -> tuple[subprocess.Popen[Any], int]:
    """Start rocjitsu and return its explicit readiness channel."""

    def restore_child_signal_mask() -> None:
        signal.pthread_sigmask(signal.SIG_SETMASK, child_signal_mask)

    ready_read, ready_write = os.pipe()
    try:
        process = subprocess.Popen(
            [*arguments, "--vfio-ready-fd", str(ready_write)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            pass_fds=(ready_write,),
            preexec_fn=restore_child_signal_mask,
        )
    except BaseException:
        os.close(ready_read)
        raise
    finally:
        os.close(ready_write)
    return process, ready_read


def wait_for_server_ready(
    paths: Sequence[Path],
    process: subprocess.Popen[Any],
    ready_fd: int,
    timeout: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> dict[Path, SocketIdentity]:
    """Wait for rocjitsu's single readiness signal and inspect every socket."""

    with selectors.DefaultSelector() as selector:
        selector.register(ready_fd, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise GuestRunError(
                f"rocjitsu did not report readiness within {timeout:g} seconds"
            )
        try:
            ready = os.read(ready_fd, 1)
        except OSError as error:
            raise GuestRunError(
                f"cannot read rocjitsu readiness signal: {error}"
            ) from error
    if ready != b"\x01":
        status = process.poll()
        detail = f" (status {status})" if status is not None else ""
        raise GuestRunError(
            "rocjitsu closed its readiness channel before the vfio-user "
            f"socket was ready{detail}"
        )

    identities: dict[Path, SocketIdentity] = {}
    for path in paths:
        try:
            socket_stat = path.lstat()
        except FileNotFoundError as error:
            raise GuestRunError(
                f"rocjitsu reported readiness without creating {path}"
            ) from error
        except OSError as error:
            raise GuestRunError(f"cannot inspect vfio-user socket: {error}") from error
        if not stat.S_ISSOCK(socket_stat.st_mode):
            raise GuestRunError(f"vfio-user socket path is not a socket: {path}")
        identities[path] = socket_stat.st_dev, socket_stat.st_ino
    return identities


def validate_socket_paths(output: Path, requested: Sequence[Path]) -> list[Path]:
    """Require distinct, absent socket files directly in the private output."""

    paths: list[Path] = []
    lexical_paths: set[Path] = set()
    resolved_paths: set[Path] = set()
    for requested_path in requested:
        if ".." in requested_path.parts:
            raise GuestRunError(
                "vfio-user socket must be directly inside the run output "
                f"directory: {requested_path}"
            )
        lexical_path = Path(os.path.abspath(requested_path))
        path = requested_path.resolve()
        if path.parent != output:
            raise GuestRunError(
                "vfio-user socket must be directly inside the run output directory"
            )
        if lexical_path in lexical_paths or path in resolved_paths:
            raise GuestRunError(f"duplicate vfio-user socket path: {requested_path}")
        if len(os.fsencode(path)) >= 108:
            raise GuestRunError(f"vfio-user socket path is too long: {path}")
        try:
            lexical_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise GuestRunError(f"vfio-user socket path already exists: {lexical_path}")
        lexical_paths.add(lexical_path)
        resolved_paths.add(path)
        paths.append(path)
    return paths


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


def socket_identity_if_present(path: Path) -> SocketIdentity | None:
    """Identify a socket left in the launcher's private output directory."""

    try:
        socket_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISSOCK(socket_stat.st_mode):
        return None
    return socket_stat.st_dev, socket_stat.st_ino


def stop_after_failure(process: subprocess.Popen[Any]) -> None:
    """Best-effort cleanup for a launch that has already failed."""

    if process.poll() is None:
        process.kill()
    process.wait()


def finish_server(
    process: subprocess.Popen[Any],
    shutdown_grace: float | None = None,
    stop_requested: bool = False,
) -> None:
    """Stop rocjitsu, force-killing it if cooperative shutdown stalls."""

    status = process.poll()
    if status is None:
        if shutdown_grace is None:
            shutdown_grace = SERVER_SHUTDOWN_GRACE_SECONDS
        if not stop_requested:
            try:
                process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            status = process.wait(timeout=shutdown_grace)
        except subprocess.TimeoutExpired as error:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait()
            raise GuestRunError(
                "rocjitsu did not exit within "
                f"{shutdown_grace:g} seconds after SIGTERM; killed it with SIGKILL"
            ) from error
    if status != 0:
        raise GuestRunError(f"rocjitsu exited with status {status}")


def run_qemu(
    arguments: list[str],
    server: subprocess.Popen[Any],
    guest_log: Any,
) -> int:
    """Run QEMU until it exits or the vfio-user server terminates."""

    guest: subprocess.Popen[Any] | None = None
    guest_pidfd = -1
    server_pidfd = -1
    try:
        with block_termination_signals() as child_signal_mask:

            def restore_child_signal_mask() -> None:
                signal.pthread_sigmask(signal.SIG_SETMASK, child_signal_mask)

            guest = subprocess.Popen(
                arguments,
                stdout=guest_log,
                stderr=subprocess.STDOUT,
                preexec_fn=restore_child_signal_mask,
            )
        guest_pidfd = os.pidfd_open(guest.pid)
        server_pidfd = os.pidfd_open(server.pid)
        with selectors.DefaultSelector() as selector:
            selector.register(guest_pidfd, selectors.EVENT_READ, "guest")
            selector.register(server_pidfd, selectors.EVENT_READ, "server")
            ready = {key.data for key, _events in selector.select()}

            if "guest" in ready:
                return guest.wait()

            server_status = server.wait()
            guest_status = guest.poll()
            if guest_status is not None:
                return guest_status
            raise GuestRunError(
                "rocjitsu exited while QEMU was running " f"(status {server_status})"
            )
    finally:
        # Once cleanup owns the child, do not let a first termination signal
        # interrupt the kill/reap sequence. The pending signal is delivered
        # after the child is gone and then unwinds the caller normally.
        with block_termination_signals():
            if guest_pidfd >= 0:
                os.close(guest_pidfd)
            if server_pidfd >= 0:
                os.close(server_pidfd)
            if guest is not None:
                stop_after_failure(guest)


def check_guest_log(path: Path, expected: list[str], rejected: list[str]) -> None:
    log = path.read_text(encoding="utf-8", errors="replace")
    for text in expected:
        if text not in log:
            raise GuestRunError(f"guest log is missing expected text: {text}")
    for text in rejected:
        if text in log:
            raise GuestRunError(f"guest log contains rejected text: {text}")


def positive_seconds(value: str) -> float:
    """Parse a finite positive timeout for argparse."""

    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return seconds


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, type=Path)
    parser.add_argument("--initramfs", required=True, type=Path)
    parser.add_argument("--qemu", required=True, type=Path)
    parser.add_argument("--rocjitsu", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--socket",
        type=Path,
        action="append",
        default=[],
        help="VFIO-user socket for one local GPU; repeat in GPU ordinal order",
    )
    parser.add_argument("--node", help="local node ID in a system manifest")
    parser.add_argument("--accel", choices=("auto", "kvm", "tcg"), default="auto")
    parser.add_argument("--memory", default="4G")
    parser.add_argument("--append", default="console=ttyS0 panic=-1")
    parser.add_argument(
        "--probe-timeout",
        type=positive_seconds,
        default=DEFAULT_PROBE_TIMEOUT_SECONDS,
        help="seconds allowed for each QEMU and rocjitsu capability probe",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_seconds,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
        help="seconds to wait for rocjitsu to create and announce its VFIO-user socket",
    )
    parser.add_argument("--expect-log", action="append", default=[])
    parser.add_argument("--reject-log", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        kernel = require_file(args.kernel, "guest kernel")
        initramfs = require_file(args.initramfs, "guest initramfs")
        qemu = require_executable(args.qemu, "QEMU")
        rocjitsu = require_executable(args.rocjitsu, "rocjitsu")
        config = require_file(args.config, "rocjitsu config")
        output = args.output.resolve()
        if output.exists():
            raise GuestRunError(f"run output directory already exists: {output}")

        socket_paths = validate_socket_paths(
            output, args.socket or [output / "vfio-user.sock"]
        )

        accelerator = select_accelerator(
            args.accel, available_accelerators(qemu, args.probe_timeout)
        )
        require_qemu_vfio_user(qemu, args.probe_timeout)
        require_rocjitsu_vfio_user(rocjitsu, args.probe_timeout)
        qemu_arguments = build_qemu_arguments(
            qemu,
            kernel,
            initramfs,
            socket_paths,
            accelerator,
            args.memory,
            args.append,
        )
        server_arguments = [
            str(rocjitsu),
            "--config",
            str(config),
        ]
        if args.node is not None:
            server_arguments.extend(["--node", args.node])
        for socket_path in socket_paths:
            server_arguments.extend(["--vfio-socket", str(socket_path)])
        if args.dry_run:
            print(shlex.join(server_arguments))
            print(shlex.join(qemu_arguments))
            return 0

        with managed_termination_signals():
            output.mkdir(mode=0o700, parents=True)
            server_log_path = output / "server.log"
            guest_log_path = output / "guest.log"
            server: subprocess.Popen[Any] | None = None
            ready_fd = -1
            run_error: GuestRunError | OSError | subprocess.SubprocessError | None = (
                None
            )
            socket_identities: dict[Path, SocketIdentity] = {}
            server_stop_requested = False
            try:
                with block_termination_signals() as child_signal_mask:
                    with server_log_path.open("wb") as server_log:
                        server, ready_fd = start_vfio_server(
                            server_arguments, server_log, child_signal_mask
                        )
                try:
                    socket_identities = wait_for_server_ready(
                        socket_paths, server, ready_fd, args.startup_timeout
                    )
                finally:
                    with block_termination_signals():
                        owned_fd, ready_fd = ready_fd, -1
                        os.close(owned_fd)
                with guest_log_path.open("wb") as guest_log:
                    guest_returncode = run_qemu(qemu_arguments, server, guest_log)
                if guest_returncode != 0:
                    raise GuestRunError(f"QEMU exited with status {guest_returncode}")
                # QEMU can close its PCI sockets before process exit. Tell the
                # node its guest is done as soon as that exit is observed;
                # reading a large guest log must not consume the group's
                # disconnect grace period before shutdown is requested.
                if server.poll() is None:
                    try:
                        server.send_signal(signal.SIGTERM)
                        server_stop_requested = True
                    except ProcessLookupError:
                        pass
                check_guest_log(guest_log_path, args.expect_log, args.reject_log)
            except (GuestRunError, OSError, subprocess.SubprocessError) as error:
                run_error = error
            finally:
                try:
                    # Defer a first SIGINT/SIGTERM until every owned resource
                    # has been closed or reaped. Otherwise an asynchronous
                    # exception here can skip server termination and unlink a
                    # socket that a surviving server still owns.
                    with block_termination_signals():
                        if ready_fd >= 0:
                            owned_fd, ready_fd = ready_fd, -1
                            try:
                                os.close(owned_fd)
                            except OSError as error:
                                if run_error is None:
                                    run_error = error
                                else:
                                    run_error = GuestRunError(
                                        f"{run_error}; readiness cleanup also failed: {error}"
                                    )
                        if server is not None:
                            try:
                                finish_server(
                                    server, stop_requested=server_stop_requested
                                )
                            except (
                                GuestRunError,
                                OSError,
                                subprocess.SubprocessError,
                            ) as error:
                                if run_error is None:
                                    run_error = error
                                else:
                                    run_error = GuestRunError(
                                        f"{run_error}; cleanup also failed: {error}"
                                    )
                            for socket_path in socket_paths:
                                try:
                                    identity = socket_identities.get(socket_path)
                                    if identity is None and server.poll() is not None:
                                        identity = socket_identity_if_present(
                                            socket_path
                                        )
                                    remove_owned_socket(socket_path, identity)
                                except OSError as error:
                                    if run_error is None:
                                        run_error = error
                                    else:
                                        run_error = GuestRunError(
                                            f"{run_error}; socket cleanup also failed: {error}"
                                        )
                except GuestRunError as error:
                    # Restoring the signal mask delivers any termination signal
                    # deferred by the cleanup block. Preserve an earlier guest
                    # failure while still reporting that signal.
                    if run_error is None:
                        run_error = error
                    elif str(run_error) != str(error):
                        run_error = GuestRunError(
                            f"{run_error}; cleanup also received: {error}"
                        )

            if run_error is not None:
                raise run_error

        return 0
    except (GuestRunError, OSError, subprocess.SubprocessError) as error:
        print(f"vfio guest launch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
