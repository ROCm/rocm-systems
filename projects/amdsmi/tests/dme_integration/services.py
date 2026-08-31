#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Start, supervise, and stop GPU Agent + Device Metrics Exporter.

Replaces three inline ``run:`` blocks (start gpuagent, start DME,
cleanup) with a single ``services`` helper that:

* Launches each process with stdout/stderr redirected to a log file.
* Waits for a TCP port to become listenable before declaring readiness
  (replacing the ``sleep 5; kill -0`` flake-prone check).
* Supports a ``stop`` mode that reads the PID file written at start time
  and terminates the service.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from ._common import _process_alive, configure_logging, gh_error

logger = logging.getLogger("dme.services")

_DEFAULT_READY_TIMEOUT_S = 30.0
_READY_POLL_INTERVAL_S = 0.5


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    """Return True when ``host:port`` accepts a TCP connection within ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(_READY_POLL_INTERVAL_S)
    return False


def _pid_cmdline(pid: int) -> str:
    """Return /proc/<pid>/cmdline as a string (NUL-separated), "" on any OSError."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return ""


def _tail_log(log_file: Path, lines: int = 50) -> None:
    try:
        text = log_file.read_text(errors="replace").splitlines()[-lines:]
        for line in text:
            print(line, file=sys.stderr)
    except OSError:
        pass


def start(
    *,
    name: str,
    binary: Path,
    log_file: Path,
    pid_file: Path,
    ready_host: str,
    ready_port: int,
    ready_timeout: float = _DEFAULT_READY_TIMEOUT_S,
    extra_env: dict[str, str] | None = None,
    args: list[str] | None = None,
) -> int:
    """Start a service detached, write its PID, and wait for readiness."""
    if not binary.is_file():
        raise FileNotFoundError(f"{name} binary not found: {binary}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [str(binary), *(args or [])]
    logger.info("starting %s: %s", name, " ".join(cmd))
    with log_file.open("wb") as log_handle:
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(proc.pid))
    logger.info("%s pid=%s log=%s", name, proc.pid, log_file)

    # ready_port == 0 means "no known port; just confirm the process stays
    # alive for ready_timeout seconds". Use a real port whenever possible.
    if ready_port == 0:
        time.sleep(ready_timeout)
        if not _process_alive(proc.pid):
            gh_error(f"{name} exited within {ready_timeout}s (see {log_file})")
            _tail_log(log_file)
            raise SystemExit(1)
        logger.info("%s alive after %ss (no port readiness configured)", name, ready_timeout)
        return proc.pid

    if not _wait_for_port(ready_host, ready_port, ready_timeout):
        if not _process_alive(proc.pid):
            gh_error(f"{name} exited before becoming ready (see {log_file})")
        else:
            gh_error(f"{name} did not open {ready_host}:{ready_port} within {ready_timeout}s")
        _tail_log(log_file)
        raise SystemExit(1)

    logger.info("%s ready on %s:%s", name, ready_host, ready_port)
    return proc.pid


def check_alive(
    *, name: str, pid_file: Path, delay: float = 2.0, log_file: Path | None = None
) -> bool:
    """Wait ``delay`` seconds, then verify the service is still running.

    Returns True if the process is alive, False (and emits a GH error) if it
    died.  Used as a post-start health check to detect early crashes (e.g. ABI
    mismatches causing immediate segfaults or stack-smashing aborts).
    """
    if not pid_file.is_file():
        gh_error(f"{name}: no pid file at {pid_file}")
        return False
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        gh_error(f"{name}: invalid pid file {pid_file}")
        return False

    time.sleep(delay)
    if _process_alive(pid):
        logger.info("%s (pid=%s) still alive after %.1fs", name, pid, delay)
        return True

    gh_error(f"{name} (pid={pid}) died within {delay}s of start")
    if log_file is not None:
        _tail_log(log_file, lines=30)
    return False


def stop(*, name: str, pid_file: Path, expected: str | None = None) -> None:
    if not pid_file.is_file():
        logger.info("no pid file for %s at %s -- nothing to stop", name, pid_file)
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        logger.warning("invalid pid file: %s", pid_file)
        return
    # Guard against a recycled PID: on a shared runner the number may now
    # belong to an unrelated process, so only signal it if its cmdline matches.
    if expected is not None and _process_alive(pid):
        cmdline = _pid_cmdline(pid)
        if expected not in cmdline:
            logger.warning(
                "%s pid=%s cmdline does not contain %r -- skipping kill", name, pid, expected
            )
            pid_file.unlink(missing_ok=True)
            return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _process_alive(pid):
            break
        logger.info("sending %s to %s pid=%s", sig.name, name, pid)
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        time.sleep(2)
    pid_file.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="Start a service and wait for readiness")
    p_start.add_argument("--name", required=True)
    p_start.add_argument("--binary", required=True, type=Path)
    p_start.add_argument("--log-file", required=True, type=Path)
    p_start.add_argument("--pid-file", required=True, type=Path)
    p_start.add_argument("--ready-host", default="127.0.0.1")
    p_start.add_argument(
        "--ready-port",
        required=True,
        type=int,
        help="Port to wait for (TCP). Use 0 to skip and only check process liveness.",
    )
    p_start.add_argument("--ready-timeout", type=float, default=_DEFAULT_READY_TIMEOUT_S)
    p_start.add_argument(
        "--ld-library-path", help="Value to set for LD_LIBRARY_PATH on the child process."
    )

    p_check = sub.add_parser("check-alive", help="Verify a service is still running")
    p_check.add_argument("--name", required=True)
    p_check.add_argument("--pid-file", required=True, type=Path)
    p_check.add_argument("--log-file", type=Path, default=None)
    p_check.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait before checking liveness (default: 2).",
    )

    p_stop = sub.add_parser("stop", help="Stop a previously-started service")
    p_stop.add_argument("--name", required=True)
    p_stop.add_argument("--pid-file", required=True, type=Path)
    p_stop.add_argument(
        "--expected",
        default=None,
        help="Only signal the pid if its cmdline contains this substring.",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.action == "start":
        extra_env: dict[str, str] = {}
        if args.ld_library_path:
            extra_env["LD_LIBRARY_PATH"] = args.ld_library_path
        start(
            name=args.name,
            binary=args.binary,
            log_file=args.log_file,
            pid_file=args.pid_file,
            ready_host=args.ready_host,
            ready_port=args.ready_port,
            ready_timeout=args.ready_timeout,
            extra_env=extra_env,
        )
    elif args.action == "check-alive":
        alive = check_alive(
            name=args.name, pid_file=args.pid_file, delay=args.delay, log_file=args.log_file
        )
        if not alive:
            return 1
    elif args.action == "stop":
        stop(name=args.name, pid_file=args.pid_file, expected=args.expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
