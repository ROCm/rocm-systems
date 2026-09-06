#!/usr/bin/env python3
"""Submit a SLURM batch job, wait for it, and verify it succeeded.

Does **not** use `sbatch --wait`. GitHub Actions cancel-in-progress sends
SIGINT/SIGTERM (and eventually SIGKILL) to the step; `sbatch --wait` then dies
without cancelling the allocation, and the compute nodes stay occupied until
the wall clock expires. Submit with `--parsable`, remember the job id, and
`scancel` it on INT/TERM/HUP so a superseded PR run releases the nodes.

A bare `sbatch --wait` also exits 0 when the scheduler kills a job (TIMEOUT,
OOM, node failure), so this still cross-checks the terminal state via `sacct`.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Callable

# Non-terminal sacct states (and a missing row); keep polling while we see these.
# This is the complement of the terminal set (BOOT_FAIL, CANCELLED, COMPLETED,
# DEADLINE, FAILED, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, TIMEOUT) -- it is the
# wait predicate now that `sbatch --wait` no longer guarantees terminality, so a
# state missing from here is silently read as "job finished" while it is still
# holding the reservation. Held/suspended states are the ones that used to be
# absent; keep this list exhaustive.
NON_TERMINAL_STATES = frozenset(
    {
        "",  # sacct has no accounting row (yet)
        "COMPLETING",
        "CONFIGURING",
        "PENDING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "RESIZING",
        "RESV_DEL_HOLD",
        "REVOKED",
        "RUNNING",
        "SIGNALING",
        "SPECIAL_EXIT",
        "STAGE_OUT",
        "STOPPED",
        "SUSPENDED",
    }
)

# How long to keep polling when sacct never produces a row for a job id that
# sbatch accepted. Past this the accounting DB is unusable, so we cannot verify
# success anyway -- give up loudly and scancel rather than hold nodes until the
# GitHub job timeout.
MISSING_ROW_TIMEOUT = 600.0


@dataclass
class JobResult:
    """Terminal accounting info for a SLURM job, as reported by sacct."""

    state: str
    exit_code: str


def log(*args: object) -> None:
    print(*args)
    sys.stdout.flush()


def parse_parsable_job_id(stdout: str) -> str:
    """Extract the job id from `sbatch --parsable` stdout (`<id>` or `<id>;<cluster>`)."""
    text = stdout.strip()
    if not text:
        return ""
    return text.splitlines()[-1].split(";")[0].strip()


def scancel_job(job_id: str) -> None:
    """Best-effort cancel. Missing scancel/job is not fatal; the wait loop will still end."""
    if not job_id:
        return
    # Cancel first, log after. This runs from a signal handler, and if the
    # signal lands while the main thread is inside print/flush, CPython raises
    # RuntimeError ("reentrant call") out of the handler -- which would abort it
    # before the one thing it exists to do.
    try:
        subprocess.run(
            ["scancel", job_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        log(f"WARNING: scancel not found; job {job_id} may keep running")
        return
    log(f"==> scancel {job_id}")


def submit_and_wait(
    script: Path,
    export: str,
    chdir: Path | None,
    partition: str | None,
    reservation: str | None = None,
    wait_poll_interval: float = 15.0,
) -> tuple[int, str]:
    """Submit with `sbatch --parsable`, wait, scancel on INT/TERM/HUP.

    Returns (returncode, job_id). returncode is 0 only if we did not cancel
    the job from this process; sacct is still the source of truth for success.
    """
    cmd = ["sbatch", "--parsable", f"--export={export}"]
    if partition:
        cmd.append(f"--partition={partition}")
    if reservation:
        cmd.append(f"--reservation={reservation}")
    cmd.append(str(script))
    log(f"==> {' '.join(cmd)}")

    cancel_requested = False
    job_id = ""

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        nonlocal cancel_requested
        cancel_requested = True
        scancel_job(job_id)
        log(f"==> caught signal {signum}; cancelled Slurm job {job_id or '<pending>'}")

    previous: dict[int, object] = {}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[sig] = signal.signal(sig, _on_signal)

    try:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(chdir) if chdir else None,
                text=True,
                capture_output=True,
            )
        except FileNotFoundError as e:
            raise RuntimeError("sbatch not found on PATH") from e

        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        sys.stdout.flush()

        job_id = parse_parsable_job_id(proc.stdout)
        if job_id and chdir is not None:
            # So a later `if: cancelled()` step can scancel if this process is
            # SIGKILL'd before the handler runs (GHA signals the step PID).
            (chdir / "slurm-job-id").write_text(f"{job_id}\n")
            log(f"==> wrote {chdir / 'slurm-job-id'}")
        if proc.returncode != 0:
            return proc.returncode, job_id
        if not job_id:
            log("WARNING: sbatch succeeded but printed no job id")
            return proc.returncode, job_id
        if cancel_requested:
            scancel_job(job_id)
            return 1, job_id

        log(f"==> waiting for job {job_id} (scancel on INT/TERM/HUP)")
        wait_rc = wait_for_job(job_id, wait_poll_interval, lambda: cancel_requested)
        return wait_rc, job_id
    except KeyboardInterrupt:
        cancel_requested = True
        scancel_job(job_id)
        return 130, job_id
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def query_job(job_id: str, retries: int, interval: float) -> JobResult:
    """Poll `sacct` until the job reaches a terminal state or retries run out."""
    state = ""
    exit_code = ""
    for _ in range(retries):
        try:
            out = subprocess.check_output(
                ["sacct", "-j", job_id, "-X", "-n", "-P", "--format=State,ExitCode"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            # sacct can transiently fail; treat as "no data yet" and retry.
            out = ""
        except FileNotFoundError as e:
            raise RuntimeError("sacct not found on PATH") from e

        lines = out.splitlines()
        if lines:
            fields = lines[0].split("|")
            # State can be e.g. "CANCELLED by 1234"; keep the leading token.
            state = fields[0].split()[0] if fields[0].strip() else ""
            exit_code = fields[1] if len(fields) > 1 else ""

        if state not in NON_TERMINAL_STATES:
            break
        time.sleep(interval)

    return JobResult(state=state, exit_code=exit_code)


def wait_for_job(
    job_id: str,
    poll_interval: float,
    cancelled: Callable[[], bool],
    missing_row_timeout: float = MISSING_ROW_TIMEOUT,
) -> int:
    """Block until sacct reports a terminal state, or until a cancel flag is set.

    Returns 0 if the job reached a terminal state on its own, 1 if we scancelled
    it or gave up waiting.

    A real PENDING/RUNNING state is waited on without a deadline -- the queue is
    allowed to be slow, and the GitHub job timeout is the backstop. What *is*
    bounded is a *continuously* empty state: sbatch handed us this id, so an
    accounting row should appear within seconds, and `""` is a member of
    NON_TERMINAL_STATES, so a permanently broken sacct would otherwise spin here
    forever. The clock resets on every non-empty answer, so a transient sacct
    outage (query_job maps CalledProcessError to `""`) mid-run does not count
    against a job that is already reporting state.
    """
    last_state_seen = time.monotonic()
    while True:
        if cancelled():
            scancel_job(job_id)
            return 1
        result = query_job(job_id, retries=1, interval=0)
        if result.state:
            if result.state not in NON_TERMINAL_STATES:
                return 1 if cancelled() else 0
            last_state_seen = time.monotonic()
        elif time.monotonic() - last_state_seen >= missing_row_timeout:
            # Cancel: sacct is the only success oracle we have, so this run is
            # already lost -- do not also leave the job holding the reservation.
            scancel_job(job_id)
            log(
                f"ERROR: sacct reported no state for job {job_id} for "
                f"{missing_row_timeout:.0f}s; giving up and cancelling it"
            )
            return 1
        deadline = time.monotonic() + poll_interval
        while time.monotonic() < deadline:
            if cancelled():
                scancel_job(job_id)
                return 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))


def evaluate(sbatch_rc: int, job_id: str, result: JobResult) -> int:
    """Decide the overall exit code from the wait rc and sacct result."""
    if sbatch_rc != 0:
        log(f"ERROR: slurm job did not complete cleanly (rc={sbatch_rc})")
        return sbatch_rc if sbatch_rc > 0 else 1

    if not job_id:
        log("WARNING: no job id from sbatch; trusting rc=0")
        return 0

    log(
        f"sacct: state={result.state or '<unavailable>'} "
        f"exit_code={result.exit_code or '<unavailable>'}"
    )

    # Only an explicit non-COMPLETED terminal state is a failure; an empty
    # state means sacct had nothing, so we fall back to the rc=0 above.
    if result.state and result.state != "COMPLETED":
        log(
            f"ERROR: job {job_id} terminal state={result.state} "
            f"(exit {result.exit_code})"
        )
        return 1

    if result.exit_code and result.exit_code.split(":")[0] != "0":
        log(f"ERROR: job {job_id} reported ExitCode={result.exit_code}")
        return 1

    log(f"job {job_id} succeeded (state={result.state or '<no sacct>'})")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job, wait for a terminal sacct state, "
        "scancel on INT/TERM/HUP, and verify the result via sacct."
    )
    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the sbatch script to submit",
    )
    parser.add_argument(
        "--export",
        type=str,
        default="ALL",
        help="Value for sbatch --export (e.g. 'ALL,FOO,BAR'). Default: ALL",
    )
    parser.add_argument(
        "--chdir",
        type=Path,
        default=None,
        help="Directory to run sbatch from (created if missing); controls where "
        "%%x-%%j.out/.err logs land",
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="",
        help="SLURM partition; passed as sbatch --partition to override the "
        "script's #SBATCH directive (per-cluster). Empty = use the script default.",
    )
    parser.add_argument(
        "--reservation",
        type=str,
        default="",
        help="SLURM reservation; passed as sbatch --reservation to pin the job "
        "to dedicated nodes (per-cluster). Empty = no reservation.",
    )
    parser.add_argument(
        "--poll-retries",
        type=int,
        default=10,
        help="How many times to poll sacct for a terminal state (default: 10)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between sacct polls (default: 3)",
    )
    parser.add_argument(
        "--wait-poll-interval",
        type=float,
        default=15.0,
        help="Seconds between sacct polls while waiting for the job (default: 15)",
    )
    args = parser.parse_args(argv)

    if not args.script.exists():
        parser.error(f"sbatch script not found: {args.script}")
    if args.chdir:
        args.chdir.mkdir(parents=True, exist_ok=True)

    sbatch_rc, job_id = submit_and_wait(
        args.script,
        args.export,
        args.chdir,
        args.partition,
        args.reservation,
        wait_poll_interval=args.wait_poll_interval,
    )
    log(f"slurm wait rc={sbatch_rc}, job_id={job_id}")

    result = JobResult(state="", exit_code="")
    if job_id:
        result = query_job(job_id, args.poll_retries, args.poll_interval)

    return evaluate(sbatch_rc, job_id, result)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
