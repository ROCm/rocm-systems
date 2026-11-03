# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run commands under Popen + wait4 and collect per-run measurements."""

from __future__ import annotations

import gc
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator, Sequence

RECORD = struct.Struct("<IIQQ")
EVENT_WORK_BEGIN = 1
EVENT_WORK_END = 2


@dataclass
class RunConfig:
    runs: int = 100
    warmup: int = 3
    timeout: float | None = None
    pause_ms: float = 10.0
    cpu_affinity: str | None = None
    mem_sample_ms: float | None = None
    clean: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    name: str
    index: int
    exit: int | None = None
    wall_ns: int = 0
    user_s: float = 0.0
    sys_s: float = 0.0
    max_rss_kb: int = 0
    ctx_vol: int = 0
    ctx_invol: int = 0
    minflt: int = 0
    majflt: int = 0
    t0_t1_ns: int | None = None
    t1_t2_ns: int | None = None
    t2_t3_ns: int | None = None
    cpu_t0_t1_ns: int | None = None
    cpu_t1_t2_ns: int | None = None
    cpu_t2_t3_ns: int | None = None
    checksum: str | None = None
    output_bytes: int = 0
    sampled_rss_kb: int | None = None
    error: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.exit == 0


def parse_markers(data: bytes) -> dict[int, list[tuple[int, int]]]:
    """Return event id -> list of (wall ns, process CPU ns) from raw pipe bytes."""
    usable = len(data) - len(data) % RECORD.size
    events: dict[int, list[tuple[int, int]]] = {}
    for event_id, _, ns, cpu_ns in RECORD.iter_unpack(data[:usable]):
        events.setdefault(event_id, []).append((ns, cpu_ns))
    return events


def read_pipe(fd: int) -> bytes:
    os.set_blocking(fd, False)
    chunks = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


METRIC_LINE = re.compile(
    r"^metric ([a-z0-9_]+) ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*$"
)


def parse_output(path: str) -> tuple[str | None, dict[str, float]]:
    """Last `checksum: <value>` and all `metric <name> <value>` lines the application printed."""
    checksum: str | None = None
    metrics: dict[str, float] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("checksum:"):
                checksum = line.split(":", 1)[1].strip()
            elif match := METRIC_LINE.match(line):
                metrics[match.group(1)] = float(match.group(2))
    return checksum, metrics


def path_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        total += sum(
            os.path.getsize(os.path.join(root, n))
            for n in files
            if os.path.isfile(os.path.join(root, n))
        )
    return total


def clean_paths(paths: Sequence[str]) -> int:
    """Remove paths and return the bytes they held."""
    freed = 0
    for path in paths:
        if not os.path.lexists(path):
            continue
        freed += path_size(path)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    return freed


def _sample_rss(
    pid: int, interval_s: float, stop: threading.Event, peak: list[int]
) -> None:
    while not stop.wait(interval_s):
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        peak[0] = max(peak[0], int(line.split()[1]))
                        break
        except OSError:
            return


def build_argv(command: str, cpu_affinity: str | None) -> list[str]:
    argv = shlex.split(command)
    return ["taskset", "-c", cpu_affinity] + argv if cpu_affinity else argv


def run_once(
    name: str,
    index: int,
    argv: Sequence[str],
    cfg: RunConfig,
    env: dict[str, str] | None = None,
) -> RunResult:
    """env: variables added on top of os.environ for this run only."""
    result = RunResult(name=name, index=index)
    read_fd, write_fd = os.pipe()
    env = {**os.environ, **(env or {}), "BENCH_FD": str(write_fd)}
    out_fd, out_path = tempfile.mkstemp(prefix="bench_out_")
    stop = threading.Event()
    peak = [0]
    sampler = None
    proc = None
    timer = None
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        t0 = time.monotonic_ns()
        try:
            proc = subprocess.Popen(
                argv,
                stdout=out_fd,
                stderr=subprocess.STDOUT,
                env=env,
                pass_fds=(write_fd,),
                start_new_session=True,
            )
        except OSError as exc:
            result.error = f"spawn failed: {exc}"
            os.unlink(out_path)
            return result
        os.close(write_fd)
        write_fd = -1
        timed_out = threading.Event()
        if cfg.timeout is not None:

            def expire() -> None:
                timed_out.set()
                # whole group: the tool may have started the workload as its own child
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            timer = threading.Timer(cfg.timeout, expire)
            timer.start()
        if cfg.mem_sample_ms:
            sampler = threading.Thread(
                target=_sample_rss,
                daemon=True,
                args=(proc.pid, cfg.mem_sample_ms / 1000, stop, peak),
            )
            sampler.start()
        _, status, ru = os.wait4(proc.pid, 0)
        t3 = time.monotonic_ns()
        if timer:
            timer.cancel()
    except BaseException:
        # The child has its own session, so Ctrl-C never reaches it: kill it here
        # or it keeps running (and using the GPU) after bench has exited.
        if timer:
            timer.cancel()
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                os.waitpid(proc.pid, 0)
            except (ProcessLookupError, ChildProcessError):
                pass
        os.unlink(out_path)
        raise
    finally:
        if gc_was_enabled:
            gc.enable()
        stop.set()
        if sampler:
            sampler.join()
        if write_fd != -1:
            os.close(write_fd)
        markers = read_pipe(read_fd)
        os.close(read_fd)
        os.close(out_fd)
    # wait4 reaped the child behind Popen's back; without this Popen warns "still running"
    proc.returncode = os.waitstatus_to_exitcode(status)
    fill_result(result, proc.returncode, t0, t3, ru, markers)
    if sampler:
        result.sampled_rss_kb = peak[0]
    if timed_out.is_set():
        result.error = f"timeout after {cfg.timeout}s"
    elif proc.returncode != 0:
        result.error = f"exit code {proc.returncode}"
    result.checksum, result.metrics = parse_output(out_path)
    os.unlink(out_path)
    return result


def fill_result(
    result: RunResult, code: int, t0: int, t3: int, ru, markers: bytes
) -> None:
    result.exit = code
    result.wall_ns = t3 - t0
    result.user_s, result.sys_s = ru.ru_utime, ru.ru_stime
    result.max_rss_kb = ru.ru_maxrss
    result.ctx_vol, result.ctx_invol = ru.ru_nvcsw, ru.ru_nivcsw
    result.minflt, result.majflt = ru.ru_minflt, ru.ru_majflt
    events = parse_markers(markers)
    begin = min(events.get(EVENT_WORK_BEGIN, []), default=None)
    end = max(events.get(EVENT_WORK_END, []), default=None)
    total_cpu_ns = round((result.user_s + result.sys_s) * 1e9)
    if begin is not None:
        result.t0_t1_ns = begin[0] - t0
        result.cpu_t0_t1_ns = begin[1]
    if begin is not None and end is not None:
        result.t1_t2_ns = end[0] - begin[0]
        result.cpu_t1_t2_ns = end[1] - begin[1]
    if end is not None:
        result.t2_t3_ns = t3 - end[0]
        result.cpu_t2_t3_ns = max(0, total_cpu_ns - end[1])


def run_command(
    name: str, command: str, cfg: RunConfig, env: dict[str, str] | None = None
) -> Iterator[RunResult]:
    """Yield one RunResult per measured run; warm-up runs are executed silently."""
    argv = build_argv(command, cfg.cpu_affinity)
    for i in range(-cfg.warmup, cfg.runs):
        result = run_once(name, i, argv, cfg, env)
        output_bytes = clean_paths(cfg.clean)
        if i >= 0:
            result.output_bytes = output_bytes
            yield result
        if cfg.pause_ms:
            time.sleep(cfg.pause_ms / 1000)
