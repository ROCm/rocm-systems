#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a bounded GPU-target raw-execution campaign and retain failing inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

MAX_STDERR_CHARACTERS = 64 * 1024
MAX_SEED = (1 << 64) - 1
SEED_LIMIT = MAX_SEED + 1

STAT_KEYS = (
    "executed",
    "async_executed",
    "async_capable_inputs",
    "async_enabled_inputs",
    "concurrent_inputs",
    "issuer_executions",
    "waves_halted",
    "rounds",
    "budget_exhausted",
    "inputs",
    "inputs_executed",
    "inputs_halted",
    "inputs_stopped",
)
TARGETS = (
    "gfx908",
    "gfx90a",
    "gfx942",
    "gfx950",
    "gfx1010",
    "gfx1030",
    "gfx1100",
    "gfx1151",
    "gfx1200",
    "gfx1201",
    "gfx1250",
    "gfx1251",
)


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=("auto", *TARGETS),
        default="auto",
        help="pin one GPU target; by default byte zero selects the target for each input",
    )
    parser.add_argument(
        "--duration", type=int, default=3600, help="wall time in seconds"
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, available_cpus()),
        help="parallel isolated fuzz processes (default: min(8, available CPUs))",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="fuzz threads in each worker process (default: 1)",
    )
    parser.add_argument(
        "--async",
        dest="async_mode",
        choices=("auto", "on", "off"),
        default="auto",
    )
    parser.add_argument("--async-helpers", type=int, default=1)
    parser.add_argument(
        "--issuers", default="auto", help="auto or 1..8 issuers per input"
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")


def parse_stats(stdout: str) -> dict[str, int]:
    lines = [line for line in stdout.splitlines() if line]
    if not lines:
        raise RuntimeError("fuzz executable produced no statistics")
    record = json.loads(lines[-1])
    return {key: int(record[key]) for key in STAT_KEYS}


def save_crash(
    output: Path,
    current: Path,
    binary: Path,
    seed: int,
    returncode: int,
    stderr: str,
    target: str,
    exact: bool,
    async_mode: str,
    async_helpers: int,
    issuers: str,
    timed_out: bool = False,
    stopped: bool = False,
    stop_reason: str | None = None,
) -> tuple[str, bool]:
    if not current.is_file():
        raise RuntimeError("fuzz executable failed without retaining current.bin")
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    directory = output / ("stopped" if stopped else "crashes")
    directory.mkdir(exist_ok=True)
    crash = directory / f"{digest}.bin"
    metadata_path = crash.with_suffix(".json")
    duplicate = crash.exists() and metadata_path.exists()
    if not duplicate:
        temporary_crash = crash.with_suffix(crash.suffix + ".tmp")
        shutil.copyfile(current, temporary_crash)
        metadata = {
            "first_seen_seed": seed,
            "returncode": returncode,
            "sha256": digest,
            "target": target,
            "exact_attribution": exact,
            "timed_out": timed_out,
            "stop_reason": stop_reason,
            "replay": [
                str(binary),
                "--input",
                str(crash),
                "--target",
                target,
                "--async",
                async_mode,
                "--async-helpers",
                str(async_helpers),
                "--issuers",
                issuers,
            ],
            "replay_note": (
                "Repeat timing-sensitive async or concurrent candidates; a single replay "
                "may not reproduce their original scheduling."
            ),
            "stderr": stderr[-MAX_STDERR_CHARACTERS:],
        }
        if exact:
            metadata["attributed_seed"] = seed
        write_json(metadata_path, metadata)
        temporary_crash.replace(crash)
    elif exact:
        metadata = json.loads(metadata_path.read_text())
        if not metadata["exact_attribution"]:
            metadata["exact_attribution"] = True
            metadata["attributed_seed"] = seed
            metadata["returncode"] = returncode
            metadata["stderr"] = stderr[-MAX_STDERR_CHARACTERS:]
            metadata["timed_out"] = bool(metadata.get("timed_out", False) and timed_out)
            write_json(metadata_path, metadata)
    return digest, duplicate


def retained_inputs(work: Path) -> list[tuple[int, Path, bool]]:
    """Return distinct exact or signal-time inputs retained by a fuzz worker."""

    def complete_pair(seed_path: Path, input_path: Path) -> tuple[int, Path] | None:
        if not seed_path.is_file() or not input_path.is_file():
            return None
        try:
            seed = int(seed_path.read_text())
            if input_path.stat().st_size == 0:
                return None
        except (OSError, ValueError):
            return None
        return seed, input_path

    retained: list[tuple[int, Path, bool]] = []
    seen: set[tuple[int, bytes]] = set()

    def retain(seed_path: Path, input_path: Path, exact: bool) -> None:
        candidate = complete_pair(seed_path, input_path)
        if candidate is None:
            return
        seed, path = candidate
        try:
            identity = (seed, hashlib.sha256(path.read_bytes()).digest())
        except OSError:
            return
        if identity in seen:
            return
        seen.add(identity)
        retained.append((seed, path, exact))

    retain(work / "failure.seed", work / "failure.bin", True)
    directories = [work, *sorted(work.glob("thread-*"))]
    for directory in directories:
        retain(directory / "current.seed", directory / "current.bin", False)
    return retained


def clear_active_inputs(work: Path) -> None:
    """Remove files from the preceding launch before recording new active inputs."""
    for name in ("failure.seed", "failure.bin", "current.seed", "current.bin"):
        (work / name).unlink(missing_ok=True)
    for directory in work.glob("thread-*"):
        if directory.is_dir():
            shutil.rmtree(directory)


def main() -> int:
    args = parse_args()
    if (
        args.duration <= 0
        or args.batch_size <= 0
        or args.workers <= 0
        or args.threads_per_worker <= 0
        or args.async_helpers < 0
        or args.async_helpers > 128
        or args.start_seed < 0
        or args.start_seed > MAX_SEED
        or args.batch_size > MAX_SEED
    ):
        raise SystemExit(
            "duration, batch size, and workers must be positive; async helpers must be 0..128; "
            "seed and batch size must fit in uint64"
        )
    if args.issuers != "auto":
        try:
            issuer_count = int(args.issuers)
        except ValueError as error:
            raise SystemExit("issuers must be auto or 1..8") from error
        if issuer_count < 1 or issuer_count > 8:
            raise SystemExit("issuers must be auto or 1..8")
        args.issuers = str(issuer_count)
    if args.async_mode == "on" and args.async_helpers == 0:
        raise SystemExit("--async on requires at least one async helper")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"fuzz executable does not exist: {binary}")

    output = args.output.resolve()
    work_root = output / "work"
    crashes = output / "crashes"
    work_root.mkdir(parents=True, exist_ok=True)
    crashes.mkdir(exist_ok=True)
    state_path = output / "campaign-state.json"
    events_path = output / "events.jsonl"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        required = {
            "batch_size",
            "campaign_started",
            "async_helpers",
            "async_mode",
            "issuers",
            "recovery_workers",
            "target",
            "threads_per_worker",
            "totals",
            "worker_ranges",
            "workers",
        }
        missing = sorted(required - state.keys())
        if missing:
            raise SystemExit(
                f"campaign state is missing required keys: {', '.join(missing)}"
            )
        if state["target"] != args.target:
            raise SystemExit("an existing campaign must resume with the same target")
        if state["async_mode"] != args.async_mode:
            raise SystemExit(
                "an existing campaign must resume with the same async mode"
            )
        if int(state["async_helpers"]) != args.async_helpers:
            raise SystemExit(
                "an existing campaign must resume with the same async helpers"
            )
        if state["issuers"] != args.issuers:
            raise SystemExit(
                "an existing campaign must resume with the same issuer mode"
            )
        if int(state["threads_per_worker"]) != args.threads_per_worker:
            raise SystemExit(
                "an existing campaign must resume with the same threads per worker"
            )
        totals = {key: int(state["totals"].get(key, 0)) for key in STAT_KEYS}
        campaign_started = state["campaign_started"]
        if (
            int(state["workers"]) != args.workers
            or int(state["batch_size"]) != args.batch_size
        ):
            raise SystemExit(
                "an existing parallel campaign must resume with the same workers and batch size"
            )
        worker_ranges = []
        for item in state["worker_ranges"]:
            next_seed = int(item["next_seed"])
            range_end = int(item["range_end"])
            if next_seed < 0 or range_end < next_seed:
                raise SystemExit("campaign state has an invalid worker seed range")
            worker_ranges.append(
                {
                    "next_seed": min(next_seed, SEED_LIMIT),
                    "range_end": min(range_end, SEED_LIMIT),
                }
            )
        recovery_workers = {int(worker) for worker in state["recovery_workers"]}
        if any(worker < 0 or worker >= args.workers for worker in recovery_workers):
            raise SystemExit("campaign state has an invalid recovery worker")
    else:
        totals = {key: 0 for key in STAT_KEYS}
        campaign_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        worker_ranges = [
            {
                "next_seed": min(
                    args.start_seed + worker * args.batch_size, SEED_LIMIT
                ),
                "range_end": min(
                    args.start_seed + (worker + 1) * args.batch_size, SEED_LIMIT
                ),
            }
            for worker in range(args.workers)
        ]
        recovery_workers: set[int] = set()

    environment = os.environ.copy()
    environment.setdefault("ASAN_OPTIONS", "abort_on_error=1:detect_leaks=0")
    environment.setdefault("UBSAN_OPTIONS", "halt_on_error=1:print_stacktrace=1")
    run_started = time.monotonic()
    deadline = run_started + args.duration
    last_report = run_started
    failures = 0
    run_inputs = 0

    def checkpoint() -> None:
        state = {
            "binary": str(binary),
            "async_helpers": args.async_helpers,
            "async_mode": args.async_mode,
            "batch_size": args.batch_size,
            "campaign_started": campaign_started,
            "next_seed": min(item["next_seed"] for item in worker_ranges),
            "totals": totals,
            "target": args.target,
            "issuers": args.issuers,
            "threads_per_worker": args.threads_per_worker,
            "retained_inputs": len(list(crashes.glob("*.bin"))),
            "stopped_inputs": len(list((output / "stopped").glob("*.bin"))),
            "recovery_workers": sorted(recovery_workers),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "worker_ranges": worker_ranges,
            "workers": args.workers,
        }
        write_json(state_path, state)

    processes: dict[
        int, tuple[subprocess.Popen[bytes], int, int, int, float, Path, Path, Path]
    ] = {}

    def launch(worker: int) -> bool:
        item = worker_ranges[worker]
        batch_seed = item["next_seed"]
        count = item["range_end"] - batch_seed
        if count <= 0:
            return False
        work = work_root / f"worker-{worker}"
        work.mkdir(exist_ok=True)
        clear_active_inputs(work)
        launch_threads = (
            1 if worker in recovery_workers else min(args.threads_per_worker, count)
        )
        stdout_path = work / "stdout.log"
        stderr_path = work / "stderr.log"
        command = [
            str(binary),
            "--random",
            str(count),
            "--seed",
            str(batch_seed),
            "--target",
            args.target,
            "--threads",
            str(launch_threads),
            "--async",
            args.async_mode,
            "--async-helpers",
            str(args.async_helpers),
            "--issuers",
            args.issuers,
            "--output",
            str(work),
        ]
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            with stdout_path.open("wb") as stdout_stream, stderr_path.open(
                "wb"
            ) as stderr_stream:
                process = subprocess.Popen(
                    command,
                    stdout=stdout_stream,
                    stderr=stderr_stream,
                    env=environment,
                )
            processes[worker] = (
                process,
                batch_seed,
                count,
                launch_threads,
                time.monotonic(),
                work,
                stdout_path,
                stderr_path,
            )
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        return True

    def advance_worker(worker: int) -> None:
        item = worker_ranges[worker]
        next_seed = item["range_end"] + (args.workers - 1) * args.batch_size
        item["next_seed"] = min(next_seed, SEED_LIMIT)
        item["range_end"] = min(next_seed + args.batch_size, SEED_LIMIT)

    def add_stats(stats: dict[str, int]) -> None:
        nonlocal run_inputs
        for key in STAT_KEYS:
            totals[key] += stats[key]
        run_inputs += stats["inputs"]

    def complete_process(worker: int) -> None:
        nonlocal failures
        (
            process,
            batch_seed,
            count,
            launch_threads,
            started,
            work,
            stdout_path,
            stderr_path,
        ) = processes[worker]
        process.wait()
        stdout = stdout_path.read_text(errors="replace")
        stderr = stderr_path.read_text(errors="replace")
        elapsed = time.monotonic() - started
        if process.returncode == 0:
            stats = parse_stats(stdout)
            if stats["inputs"] != count:
                raise RuntimeError(
                    "fuzz executable reported an incomplete successful batch"
                )
            add_stats(stats)
            advance_worker(worker)
            recovery_workers.discard(worker)
            append_event(
                events_path,
                {
                    "type": "batch",
                    "worker": worker,
                    "seed": batch_seed,
                    "cases": count,
                    "elapsed_seconds": elapsed,
                    "stats": stats,
                },
            )
        else:
            candidates = retained_inputs(work)
            if not candidates:
                raise RuntimeError(
                    "fuzz executable failed without retaining an active input"
                )
            for failed_seed, current, retained_exact in candidates:
                exact = retained_exact or (launch_threads == 1 and len(candidates) == 1)
                if failed_seed < batch_seed or failed_seed >= batch_seed + count:
                    raise RuntimeError("retained seed is outside the active batch")
                digest, duplicate = save_crash(
                    output,
                    current,
                    binary,
                    failed_seed,
                    process.returncode,
                    stderr,
                    args.target,
                    exact,
                    args.async_mode,
                    args.async_helpers,
                    args.issuers,
                )
                append_event(
                    events_path,
                    {
                        "type": "crash-candidate",
                        "worker": worker,
                        "seed": failed_seed,
                        "sha256": digest,
                        "duplicate": duplicate,
                        "exact_attribution": exact,
                        "returncode": process.returncode,
                        "elapsed_seconds": elapsed,
                    },
                )
                print(
                    f"saved crash candidate seed={failed_seed} sha256={digest} "
                    f"duplicate={duplicate}",
                    flush=True,
                )
            failures += 1
            if launch_threads > 1:
                # A fatal signal can leave one active file per fuzz thread. Preserve
                # all candidates, then replay the complete batch serially so no seed
                # is skipped and a deterministic failure receives exact attribution.
                worker_ranges[worker]["next_seed"] = batch_seed
                recovery_workers.add(worker)
            else:
                resume_seed = candidates[0][0] + 1
                if resume_seed >= worker_ranges[worker]["range_end"]:
                    advance_worker(worker)
                    recovery_workers.discard(worker)
                else:
                    worker_ranges[worker]["next_seed"] = resume_seed
                    recovery_workers.add(worker)

        processes.pop(worker)
        checkpoint()

    def stop_processes() -> set[int]:
        requested_stop: set[int] = set()
        for worker, (process, *_) in processes.items():
            if process.poll() is None:
                try:
                    process.terminate()
                    requested_stop.add(worker)
                except ProcessLookupError:
                    pass
        for worker, (process, *_) in processes.items():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                    requested_stop.add(worker)
                except ProcessLookupError:
                    pass
                process.wait()
        stop_returncodes = {-signal.SIGTERM, -signal.SIGKILL}
        return {
            worker
            for worker in requested_stop
            if processes[worker][0].returncode in stop_returncodes
        }

    def retain_stopped_processes(reason: str) -> None:
        stopped_workers = stop_processes()
        if reason != "stopped":
            for worker in list(processes):
                if worker not in stopped_workers:
                    complete_process(worker)
        else:
            stopped_workers = set(processes)
        for worker in stopped_workers:
            task = processes[worker]
            (
                process,
                batch_seed,
                count,
                _launch_threads,
                started,
                work,
                _stdout_path,
                stderr_path,
            ) = task
            candidates = retained_inputs(work)
            if not candidates:
                processes.pop(worker)
                continue
            stderr = stderr_path.read_text(errors="replace")
            for failed_seed, current, exact in candidates:
                if failed_seed < batch_seed or failed_seed >= batch_seed + count:
                    raise RuntimeError("retained seed is outside the stopped batch")
                digest, duplicate = save_crash(
                    output,
                    current,
                    binary,
                    failed_seed,
                    process.returncode,
                    stderr,
                    args.target,
                    exact,
                    args.async_mode,
                    args.async_helpers,
                    args.issuers,
                    timed_out=reason == "deadline",
                    stopped=True,
                    stop_reason=reason,
                )
                append_event(
                    events_path,
                    {
                        "type": f"{reason}-candidate",
                        "worker": worker,
                        "seed": failed_seed,
                        "sha256": digest,
                        "duplicate": duplicate,
                        "exact_attribution": exact,
                        "returncode": process.returncode,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
            processes.pop(worker)

    try:
        for worker in range(args.workers):
            launch(worker)
        while processes:
            completed = [
                worker
                for worker, task in processes.items()
                if task[0].poll() is not None
            ]
            if not completed:
                if time.monotonic() >= deadline:
                    retain_stopped_processes("deadline")
                    checkpoint()
                    break
                time.sleep(0.01)
                continue
            for worker in completed:
                complete_process(worker)
                now = time.monotonic()
                if now - last_report >= 60:
                    rate = run_inputs / max(now - run_started, 0.001)
                    print(
                        f"progress inputs={totals['inputs']} executed_inputs="
                        f"{totals['inputs_executed']} rate={rate:.1f}/s workers={args.workers} "
                        f"threads={args.workers * args.threads_per_worker} "
                        f"retained_inputs={len(list(crashes.glob('*.bin')))}",
                        flush=True,
                    )
                    last_report = now
                if now < deadline:
                    launch(worker)
    except KeyboardInterrupt:
        retain_stopped_processes("interrupt")
        checkpoint()
        print("campaign interrupted; checkpoint saved", file=sys.stderr)
        return 130
    except BaseException:
        retain_stopped_processes("stopped")
        checkpoint()
        raise

    checkpoint()
    summary = {
        "async_helpers": args.async_helpers,
        "async_mode": args.async_mode,
        "elapsed_seconds": time.monotonic() - run_started,
        "failures_this_run": failures,
        "next_seed": min(item["next_seed"] for item in worker_ranges),
        "totals": totals,
        "target": args.target,
        "issuers": args.issuers,
        "threads_per_worker": args.threads_per_worker,
        "retained_inputs": len(list(crashes.glob("*.bin"))),
        "stopped_inputs": len(list((output / "stopped").glob("*.bin"))),
        "recovery_workers": sorted(recovery_workers),
        "workers": args.workers,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
