# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Console log, detailed-log tee, and the subprocess helper used by every phase."""

from __future__ import annotations

import datetime as _dt
import subprocess
import sys
import time
from pathlib import Path
from typing import NoReturn

from .context import RunContext

_ctx: RunContext | None = None


def begin_run(ctx: RunContext) -> None:
    """Install ``ctx`` as the run whose log and summary ``die()`` will update."""
    global _ctx
    _ctx = ctx
    ctx.run_start = time.monotonic()
    ctx.step_n = 0
    ctx.step_title = None
    ctx.step_start = 0.0
    ctx.step_times = []


def current() -> RunContext:
    if _ctx is None:
        raise RuntimeError("nightly run has not been started")
    return _ctx


def fmt_dur(secs: float) -> str:
    secs = int(round(secs))
    if secs < 60:
        return f"{secs}s"
    minutes, seconds = divmod(secs, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def emit(text: str, *, end: str = "\n", stream=None) -> None:
    """Write ``text`` to the console and, if open, to the detailed log file."""
    out = stream or sys.stdout
    out.write(text + end)
    out.flush()
    handle = None if _ctx is None else _ctx.detail_fh
    if handle is not None:
        handle.write(text + end)
        handle.flush()


def open_detailed_log(path: Path) -> None:
    """Open the detailed log file; all subsequent output is teed into it."""
    ctx = current()
    ctx.detail_fh = open(path, "a", encoding="utf-8")  # noqa: SIM115
    ctx.detail_fh.write(
        f"\n{'#' * 72}\n# rocprof-sys nightly tarball test run\n"
        f"# started: {_dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"{'#' * 72}\n"
    )
    ctx.detail_fh.flush()


def log(msg: str) -> None:
    emit(f"[nightly-test] {msg}")


def _close_step() -> None:
    """Record and print the duration of the step that is currently open."""
    ctx = current()
    if ctx.step_title is not None:
        dur = time.monotonic() - ctx.step_start
        ctx.step_times.append((ctx.step_title, dur))
        emit(f"[nightly-test] step completed in {fmt_dur(dur)}")
        ctx.step_title = None


def step(title: str) -> None:
    ctx = current()
    _close_step()
    ctx.step_n += 1
    emit(f"\n{'=' * 72}\n[{ctx.step_n}] {title}\n{'=' * 72}")
    ctx.step_title = title
    ctx.step_start = time.monotonic()


def die(msg: str, code: int = 1) -> NoReturn:
    emit(f"\n[nightly-test][ERROR] {msg}", stream=sys.stderr)
    if _ctx is not None and _ctx.summary_log is not None:
        _ctx.facts.setdefault("result", "ABORTED")
        _ctx.facts["error"] = msg.splitlines()[0]
        try:
            from .reporting import write_summary

            write_summary(_ctx.summary_log, _ctx.facts)
        except Exception:  # noqa: BLE001
            pass
    sys.exit(code)


def step_times() -> list[tuple[str, float]]:
    return [] if _ctx is None else _ctx.step_times


def run_start() -> float:
    return 0.0 if _ctx is None else _ctx.run_start


def run(cmd, *, cwd=None, env=None, check=True, quiet=False):
    """Run a subprocess, streaming (and teeing) its output. Returns returncode."""
    printable = " ".join(str(c) for c in cmd)
    if not quiet:
        log(f"$ {printable}" + (f"   (cwd={cwd})" if cwd else ""))
    # context manager so the stdout pipe is closed deterministically, not at GC
    with subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            emit(line, end="")
        rc = proc.wait()
    if check and rc != 0:
        die(f"command failed (exit {rc}): {printable}", rc)
    return rc
