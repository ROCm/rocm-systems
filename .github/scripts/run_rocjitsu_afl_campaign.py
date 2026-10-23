#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run parallel AFL++ workers for one bounded rocjitsu campaign."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import BinaryIO

PROGRESS_INTERVAL_SECONDS = 30
PROCESS_POLL_INTERVAL_SECONDS = 0.25
WORKER_SHUTDOWN_GRACE_SECONDS = 60
WORKER_STOP_TIMEOUT_SECONDS = 5


class CampaignInterrupted(Exception):
    """Record the signal that interrupted a campaign."""

    def __init__(self, signum: int) -> None:
        self.signum = signum


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def available_cpu_count() -> int:
    """Return the CPUs available to this process, respecting its affinity mask."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def final_log_lines(path: Path, limit: int = 40) -> str:
    """Read the final lines of a worker log for a failure diagnostic."""
    try:
        return "".join(
            path.read_text(errors="replace").splitlines(keepends=True)[-limit:]
        )
    except OSError as error:
        return f"could not read {path}: {error}\n"


def read_fuzzer_stat(path: Path, key: str) -> float:
    """Read one numeric value from an AFL++ fuzzer_stats file."""
    try:
        for line in path.read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name.strip() == key:
                try:
                    return float(value)
                except ValueError:
                    return 0
    except OSError:
        pass
    return 0


def aggregate_progress(output_dir: Path) -> tuple[int, float]:
    """Sum input counts and rates from every worker that has published stats."""
    stats = output_dir.glob("*/fuzzer_stats") if output_dir.is_dir() else ()
    paths = list(stats)
    executions = round(sum(read_fuzzer_stat(path, "execs_done") for path in paths))
    rate = sum(read_fuzzer_stat(path, "execs_per_sec") for path in paths)
    return executions, rate


def report_progress(
    output_dir: Path,
    started_at: float,
    active_workers: int,
    workers: int,
    *,
    complete: bool = False,
) -> None:
    """Print one line of aggregate progress for the live Actions log."""
    executions, rate = aggregate_progress(output_dir)
    elapsed_seconds = round(time.monotonic() - started_at)
    phase = "complete" if complete else "progress"
    active = "" if complete else f"; {active_workers}/{workers} workers active"
    print(
        f"AFL++ campaign {phase}: {elapsed_seconds}s elapsed{active}; "
        f"{executions} inputs executed ({rate:.2f} inputs/s)",
        flush=True,
    )


def process_group_exists(process_group: int) -> bool:
    """Return whether a process group still has a live member."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def signal_process_groups(
    process_groups: set[int], worker_signal: signal.Signals
) -> None:
    """Signal every retained process group that still exists."""
    for process_group in process_groups:
        try:
            os.killpg(process_group, worker_signal)
        except ProcessLookupError:
            pass


def child_processes(process_id: int) -> list[int]:
    """Read the direct children of a Linux process from procfs."""
    try:
        children = Path(f"/proc/{process_id}/task/{process_id}/children").read_text()
    except OSError:
        return []
    return [int(child) for child in children.split()]


def worker_process_groups(process: subprocess.Popen[bytes]) -> set[int]:
    """Capture the process groups in one AFL++ worker's descendant tree."""
    pending = [process.pid]
    visited: set[int] = set()
    process_groups: set[int] = set()
    runner_process_group = os.getpgrp()
    while pending:
        process_id = pending.pop()
        if process_id in visited:
            continue
        visited.add(process_id)
        pending.extend(child_processes(process_id))
        try:
            process_group = os.getpgid(process_id)
        except ProcessLookupError:
            continue
        if process_group != runner_process_group:
            process_groups.add(process_group)
    return process_groups


def stop_workers(
    processes: list[subprocess.Popen[bytes]],
    timeout_seconds: float = WORKER_STOP_TIMEOUT_SECONDS,
    retained_groups: list[set[int]] | None = None,
) -> None:
    """Stop captured worker and target process groups, then reap direct children."""
    if retained_groups is None:
        retained_groups = [set() for _ in processes]
    for process, process_groups in zip(processes, retained_groups, strict=True):
        process_groups.update(worker_process_groups(process))
    running_groups = {
        process_group
        for process_groups in retained_groups
        for process_group in process_groups
        if process_group_exists(process_group)
    }
    signal_process_groups(running_groups, signal.SIGINT)
    deadline = time.monotonic() + timeout_seconds
    while (
        any(process_group_exists(process_group) for process_group in running_groups)
        and time.monotonic() < deadline
    ):
        time.sleep(
            min(
                PROCESS_POLL_INTERVAL_SECONDS,
                max(0, deadline - time.monotonic()),
            )
        )
    signal_process_groups(
        {
            process_group
            for process_group in running_groups
            if process_group_exists(process_group)
        },
        signal.SIGKILL,
    )
    for process in processes:
        if process.poll() is None:
            process.wait()


def run_campaign(
    binary: Path,
    seed_dir: Path,
    output_dir: Path,
    duration_seconds: int,
    workers: int,
    *,
    shutdown_grace_seconds: int = WORKER_SHUTDOWN_GRACE_SECONDS,
    worker_stop_timeout_seconds: float = WORKER_STOP_TIMEOUT_SECONDS,
) -> int:
    """Launch a synchronized AFL++ campaign and report whether every worker passed."""
    available_cpus = available_cpu_count()
    if available_cpus < workers:
        print(
            f"campaign requires {workers} CPUs, but the runner exposes {available_cpus}",
            file=sys.stderr,
        )
        return 1

    log_dir = Path(f"{output_dir}-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[bytes]] = []
    instances: list[str] = []
    log_files: list[BinaryIO] = []
    retained_groups: list[set[int]] = []

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        raise CampaignInterrupted(signum)

    handled_signals = (signal.SIGINT, signal.SIGTERM)
    previous_handlers = {
        signum: signal.signal(signum, handle_signal) for signum in handled_signals
    }
    try:
        for worker in range(workers):
            instance = f"worker-{worker:02d}"
            log_file = (log_dir / f"{instance}.log").open("wb")
            log_files.append(log_file)
            command = [
                "afl-fuzz",
                "-V",
                str(duration_seconds),
                "-i",
                os.fspath(seed_dir),
                "-o",
                os.fspath(output_dir),
                "-S",
                instance,
                "-m",
                "none",
                "--",
                os.fspath(binary),
                "--afl",
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            instances.append(instance)
            retained_groups.append({processes[-1].pid})

        print(
            f"AFL++ campaign started: {workers} workers for {duration_seconds}s",
            flush=True,
        )
        started_at = time.monotonic()
        next_progress = started_at + PROGRESS_INTERVAL_SECONDS
        # AFL++ checks -V while updating its statistics, so a long stage can run
        # past that limit. Ask every live process group to stop at our deadline.
        campaign_deadline = started_at + duration_seconds
        shutdown_deadline = started_at + duration_seconds + shutdown_grace_seconds
        shutdown_requested = False
        unresponsive_workers: set[str] = set()
        while True:
            for process in processes:
                process.poll()
            if shutdown_requested:
                for process, process_groups in zip(
                    processes, retained_groups, strict=True
                ):
                    process_groups.update(worker_process_groups(process))
            active_instances = [
                instance
                for instance, process_groups in zip(
                    instances, retained_groups, strict=True
                )
                if any(process_group_exists(group) for group in process_groups)
            ]
            if not active_instances:
                break
            current_time = time.monotonic()
            if current_time >= next_progress:
                report_progress(
                    output_dir,
                    started_at,
                    len(active_instances),
                    workers,
                    complete=False,
                )
                next_progress += PROGRESS_INTERVAL_SECONDS
            if not shutdown_requested and current_time >= campaign_deadline:
                print(
                    "AFL++ campaign duration reached; requesting graceful shutdown "
                    f"from {len(active_instances)} worker(s)",
                    flush=True,
                )
                for process, instance, process_groups in zip(
                    processes, instances, retained_groups, strict=True
                ):
                    if instance in active_instances:
                        process_groups.update(worker_process_groups(process))
                        signal_process_groups({process.pid}, signal.SIGINT)
                shutdown_requested = True
            if shutdown_requested and current_time >= shutdown_deadline:
                unresponsive_workers = set(active_instances)
                print(
                    f"AFL++ campaign reached its {shutdown_grace_seconds}s shutdown "
                    f"grace with {len(unresponsive_workers)} unresponsive worker(s); "
                    "force-stopping them and failing the campaign",
                    flush=True,
                )
                stop_workers(
                    processes,
                    worker_stop_timeout_seconds,
                    retained_groups,
                )
                break
            time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
        report_progress(output_dir, started_at, 0, workers, complete=True)

        status = 0
        for process, instance in zip(processes, instances, strict=True):
            stats = output_dir / instance / "fuzzer_stats"
            if instance in unresponsive_workers:
                print(
                    f"AFL++ instance {instance} did not stop within the shutdown "
                    "grace; final log lines follow:",
                    file=sys.stderr,
                )
                print(
                    final_log_lines(log_dir / f"{instance}.log"),
                    end="",
                    file=sys.stderr,
                )
                if not stats.is_file():
                    print(
                        f"AFL++ instance {instance} did not publish fuzzer_stats",
                        file=sys.stderr,
                    )
                status = 1
                continue
            if process.returncode != 0:
                print(
                    f"AFL++ instance {instance} failed; final log lines follow:",
                    file=sys.stderr,
                )
                print(
                    final_log_lines(log_dir / f"{instance}.log"),
                    end="",
                    file=sys.stderr,
                )
                status = 1
            elif not stats.is_file():
                print(
                    f"AFL++ instance {instance} did not publish fuzzer_stats; "
                    "final log lines follow:",
                    file=sys.stderr,
                )
                print(
                    final_log_lines(log_dir / f"{instance}.log"),
                    end="",
                    file=sys.stderr,
                )
                status = 1
            elif read_fuzzer_stat(stats, "execs_done") <= 0:
                print(
                    f"AFL++ instance {instance} did not execute any inputs; "
                    "final log lines follow:",
                    file=sys.stderr,
                )
                print(
                    final_log_lines(log_dir / f"{instance}.log"),
                    end="",
                    file=sys.stderr,
                )
                status = 1
        return status
    except CampaignInterrupted as interruption:
        print(f"campaign interrupted by signal {interruption.signum}", file=sys.stderr)
        return 128 + interruption.signum
    except OSError as error:
        print(f"could not launch AFL++ campaign: {error}", file=sys.stderr)
        return 1
    finally:
        stop_workers(processes, worker_stop_timeout_seconds, retained_groups)
        for log_file in log_files:
            log_file.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("seed_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("duration_seconds", type=positive_integer)
    parser.add_argument("workers", type=positive_integer)
    arguments = parser.parse_args(argv)
    if not arguments.binary.is_file() or not os.access(arguments.binary, os.X_OK):
        parser.error(f"binary is not executable: {arguments.binary}")
    if not arguments.seed_dir.is_dir():
        parser.error(f"seed directory does not exist: {arguments.seed_dir}")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    return run_campaign(
        arguments.binary,
        arguments.seed_dir,
        arguments.output_dir,
        arguments.duration_seconds,
        arguments.workers,
    )


if __name__ == "__main__":
    sys.exit(main())
