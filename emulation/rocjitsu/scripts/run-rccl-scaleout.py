#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run two gfx1250 RCCL ranks on separate rocjitsu nodes over NET/Socket.

This is a same-host network control for a two-node system manifest. Each node
gets its own daemon and runtime directory; the ranks share only RCCL's
communicator rendezvous directory and the host loopback network.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 300.0
TERMINATION_GRACE_SECONDS = 2.0
KILL_WAIT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.05
MAX_LOG_BYTES = 32 * 1024 * 1024
TERMINATION_SIGNALS = frozenset((signal.SIGINT, signal.SIGTERM))
VISIBILITY_VARIABLES = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
RCCL_CASES = (
    "RcclTest.AllReduce",
    "RcclTest.Broadcast",
    "RcclTest.AllGather",
    "RcclTest.ReduceScatter",
    "RcclTest.SendRecv",
)
RCCL_ENVIRONMENT = {
    "NCCL_NET": "Socket",
    "NCCL_SOCKET_IFNAME": "lo",
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,NET",
    "HSA_NO_SCRATCH_RECLAIM": "1",
}
RUN_LINE = re.compile(r"^\[\s*RUN\s*\]\s+(RcclTest\.\S+)\s*$", re.MULTILINE)
OK_LINE = re.compile(
    r"^\[\s*OK\s*\]\s+(RcclTest\.\S+)(?:\s+\(\d+ ms\))?\s*$",
    re.MULTILINE,
)
PASSED_LINE = re.compile(r"^\[\s*PASSED\s*\]\s+(\d+) tests?\.\s*$", re.MULTILINE)
FAILED_LINE = re.compile(r"^\[\s*FAILED\s*\]", re.MULTILINE)
CHANNEL_LINE = re.compile(r"\bNCCL INFO Channel \d+/\d+\s*:.*\bvia NET/Socket/\S+")


class ScaleOutError(ValueError):
    """A two-node RCCL launch or its verification failed."""


class SignalState:
    def __init__(self) -> None:
        self.received: signal.Signals | None = None
        self.cleaning = False


@contextlib.contextmanager
def managed_signals() -> Iterator[SignalState]:
    """Convert launcher termination to an error after owned children exist."""

    state = SignalState()
    previous: dict[signal.Signals, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        state.received = signal.Signals(signum)
        if not state.cleaning:
            raise ScaleOutError(f"launcher received {state.received.name}")

    try:
        for caught in TERMINATION_SIGNALS:
            previous[caught] = signal.signal(caught, interrupted)
        yield state
    finally:
        for caught, handler in previous.items():
            signal.signal(caught, handler)


@contextlib.contextmanager
def block_termination_signals() -> Iterator[set[signal.Signals]]:
    """Keep a signal from arriving between Popen and recording its process."""

    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
    try:
        yield previous_mask
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number of seconds") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return seconds


def seed_value(value: str) -> int:
    try:
        seed = int(value, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if not 0 <= seed <= 2**31 - 1:
        raise argparse.ArgumentTypeError("must be between 0 and 2147483647")
    return seed


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rank-binary", required=True, type=Path)
    parser.add_argument(
        "--node",
        action="append",
        required=True,
        help="one node ID from the manifest; specify exactly twice",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        default=DEFAULT_TIMEOUT_SECONDS,
        type=positive_seconds,
        help="seconds allowed for both ranks to finish",
    )
    parser.add_argument("--gtest-filter", default="RcclTest.AllReduce")
    parser.add_argument("--seed", type=seed_value)
    args = parser.parse_args(arguments)
    if len(args.node) != 2 or args.node[0] == args.node[1]:
        parser.error("specify two distinct --node IDs")
    if args.gtest_filter != "RcclTest.*" and args.gtest_filter not in RCCL_CASES:
        parser.error("--gtest-filter must name one RcclTest case or RcclTest.*")
    return args


def require_file(path: Path, description: str, executable: bool = False) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ScaleOutError(f"{description} is not a file: {path}")
    if executable and not os.access(resolved, os.X_OK):
        raise ScaleOutError(f"{description} is not executable: {path}")
    return resolved


def require_one_gpu_nodes(config: Path, node_ids: Sequence[str]) -> None:
    """Check the selected nodes before launching expensive emulator work."""

    try:
        manifest = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScaleOutError(f"cannot read system manifest {config}: {error}") from error
    system = manifest.get("system") if isinstance(manifest, dict) else None
    nodes = system.get("nodes") if isinstance(system, dict) else None
    if not isinstance(nodes, list):
        raise ScaleOutError("config has no system.nodes manifest")
    for node_id in node_ids:
        selected = [
            node
            for node in nodes
            if isinstance(node, dict) and node.get("id") == node_id
        ]
        if len(selected) != 1:
            raise ScaleOutError(f"manifest must contain exactly one node {node_id!r}")
        gpus = selected[0].get("gpus")
        if not isinstance(gpus, list) or len(gpus) != 1:
            raise ScaleOutError(f"node {node_id!r} must contain exactly one GPU")


def rank_arguments(
    rocjitsu: Path,
    config: Path,
    rank_binary: Path,
    node_id: str,
    rank: int,
    shared: Path,
    gtest_filter: str,
    seed: int | None,
) -> list[str]:
    arguments = [
        str(rocjitsu),
        "--daemon",
        "--config",
        str(config),
        "--node",
        node_id,
        "--",
        str(rank_binary),
        f"--rank={rank}",
        "--world-size=2",
        f"--shared-dir={shared}",
        "--expected-devices=1",
        f"--gtest_filter={gtest_filter}",
    ]
    if seed is not None:
        arguments.append(f"--seed={seed}")
    return arguments


def rank_environment(runtime: Path, xdg_runtime: Path) -> dict[str, str]:
    environment = os.environ.copy()
    # GTest sharding, shuffling, and repeat settings can make the two ranks
    # execute different case sequences or only a subset of the requested cases.
    for variable in tuple(environment):
        if variable in VISIBILITY_VARIABLES or variable.startswith("GTEST_"):
            environment.pop(variable)
    environment.update(RCCL_ENVIRONMENT)
    environment["ROCJITSU_RUNTIME_DIR"] = str(runtime)
    environment["XDG_RUNTIME_DIR"] = str(xdg_runtime)
    return environment


def start_rank(
    arguments: list[str],
    environment: dict[str, str],
    log_path: Path,
    processes: list[subprocess.Popen[bytes]],
) -> subprocess.Popen[bytes]:
    """Spawn a rank in a separate group, restoring signals masked by the parent."""

    with block_termination_signals() as child_signal_mask:

        def restore_child_signal_mask() -> None:
            signal.pthread_sigmask(signal.SIG_SETMASK, child_signal_mask)

        with log_path.open("wb") as log:
            process = subprocess.Popen(
                arguments,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
                preexec_fn=restore_child_signal_mask,
            )
            processes.append(process)
            return process


def signal_process_group(
    process: subprocess.Popen[bytes], signum: signal.Signals
) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def process_group_exists(group_id: int) -> bool:
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    return True


def cleanup_process_groups(processes: Sequence[subprocess.Popen[bytes]]) -> None:
    """Stop descendants as well as rank launchers and reap every launcher."""

    for process in processes:
        signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while True:
        for process in processes:
            process.poll()  # Reap exited leaders so an empty group can disappear.
        if not any(process_group_exists(process.pid) for process in processes):
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))
    # An exited launcher can leave a child in its process group. Send SIGKILL
    # to every group, including groups whose launcher already exited.
    for process in processes:
        signal_process_group(process, signal.SIGKILL)
    for process in processes:
        try:
            process.wait(timeout=KILL_WAIT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise ScaleOutError(
                f"rank launcher {process.pid} could not be reaped"
            ) from error


def wait_for_ranks(
    processes: Sequence[subprocess.Popen[bytes]], deadline: float, timeout: float
) -> None:
    while True:
        # A raising signal handler can interrupt Popen after it acquires its
        # waitpid lock but before it installs the finally that releases it.
        with block_termination_signals():
            statuses = [process.poll() for process in processes]
        for rank, status in enumerate(statuses):
            if status is not None and status != 0:
                raise ScaleOutError(f"rank {rank} exited with status {status}")
        if all(status is not None for status in statuses):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ScaleOutError(f"ranks did not finish within {timeout:g} seconds")
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))


def verify_rank_log(log_path: Path, rank: int, expected_cases: Sequence[str]) -> None:
    if log_path.stat().st_size > MAX_LOG_BYTES:
        raise ScaleOutError(f"rank {rank} log exceeds {MAX_LOG_BYTES} bytes")
    log = log_path.read_text(encoding="utf-8", errors="replace")
    ran = RUN_LINE.findall(log)
    passed = OK_LINE.findall(log)
    summaries = PASSED_LINE.findall(log)
    if (
        sorted(ran) != sorted(expected_cases)
        or ran != passed
        or len(summaries) != 1
        or int(summaries[0]) != len(expected_cases)
        or FAILED_LINE.search(log)
    ):
        raise ScaleOutError(f"rank {rank} log lacks a passing RcclTest result")
    if not CHANNEL_LINE.search(log):
        raise ScaleOutError(f"rank {rank} log lacks a NET/Socket channel selection")


def check_daemon_socket_path(runtime: Path) -> None:
    # The CLI appends its PID before binding daemon.sock. Use the longest
    # positive pid_t decimal form so the check remains valid after PID reuse or
    # a larger kernel pid_max. Linux sockaddr_un.sun_path has 108 bytes,
    # including the terminating NUL for a pathname socket.
    socket_path = runtime / "2147483647" / "daemon.sock"
    if len(os.fsencode(socket_path)) >= 108:
        raise ScaleOutError(
            f"daemon socket path would exceed Linux's 107-byte limit: {socket_path}"
        )


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    if not sys.platform.startswith("linux"):
        raise ScaleOutError("process-group cleanup requires Linux")
    rocjitsu = require_file(args.rocjitsu, "rocjitsu", executable=True)
    config = require_file(args.config, "system manifest")
    rank_binary = require_file(args.rank_binary, "RCCL rank binary", executable=True)
    require_one_gpu_nodes(config, args.node)
    output = args.output.resolve()
    if output.exists():
        raise ScaleOutError(f"run output directory already exists: {output}")
    for rank in range(2):
        check_daemon_socket_path(output / f"runtime{rank}")

    processes: list[subprocess.Popen[bytes]] = []
    log_paths = (output / "rank0.log", output / "rank1.log")
    with managed_signals() as signal_state:
        output.mkdir(mode=0o700, parents=True)
        shared = output / "shared"
        shared.mkdir(mode=0o700)
        deadline = time.monotonic() + args.timeout
        try:
            for rank, node_id in enumerate(args.node):
                runtime = output / f"runtime{rank}"
                xdg_runtime = output / f"xdg{rank}"
                runtime.mkdir(mode=0o700)
                xdg_runtime.mkdir(mode=0o700)
                arguments = rank_arguments(
                    rocjitsu,
                    config,
                    rank_binary,
                    node_id,
                    rank,
                    shared,
                    args.gtest_filter,
                    args.seed,
                )
                environment = rank_environment(runtime, xdg_runtime)
                start_rank(arguments, environment, log_paths[rank], processes)
            wait_for_ranks(processes, deadline, args.timeout)
            expected_cases = (
                RCCL_CASES
                if args.gtest_filter == "RcclTest.*"
                else (args.gtest_filter,)
            )
            for rank, log_path in enumerate(log_paths):
                verify_rank_log(log_path, rank, expected_cases)
            if time.monotonic() > deadline:
                raise ScaleOutError(
                    f"ranks did not finish within {args.timeout:g} seconds"
                )
        finally:
            signal_state.cleaning = True
            try:
                with block_termination_signals():
                    cleanup_process_groups(processes)
            finally:
                signal_state.cleaning = False
        if signal_state.received is not None:
            raise ScaleOutError(f"launcher received {signal_state.received.name}")
    return log_paths


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    output = args.output.resolve()
    log_paths = (output / "rank0.log", output / "rank1.log")
    try:
        run(args)
    except (ScaleOutError, OSError, subprocess.SubprocessError) as error:
        print(f"RCCL scale-out run failed: {error}", file=sys.stderr)
        if output.is_dir():
            print(f"rank logs: {log_paths[0]} {log_paths[1]}", file=sys.stderr)
        return 1
    print("RCCL scale-out NET/Socket smoke passed")
    print(f"rank logs: {log_paths[0]} {log_paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
