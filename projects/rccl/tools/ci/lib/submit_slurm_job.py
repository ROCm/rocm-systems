#!/usr/bin/env python3
"""Submit a SLURM batch job, wait for it, and verify it succeeded.

`sbatch --wait` exits 0 even when the scheduler kills a job (TIMEOUT, OOM, node
failure), so this cross-checks the terminal state via `sacct`.
"""

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Non-terminal sacct states (and a missing row); keep polling while we see these.
NON_TERMINAL_STATES = frozenset(
    {"", "RUNNING", "PENDING", "REQUEUED", "COMPLETING", "RESIZING"}
)


@dataclass
class JobResult:
    """Terminal accounting info for a SLURM job, as reported by sacct."""

    state: str
    exit_code: str


def log(*args: object) -> None:
    print(*args)
    sys.stdout.flush()


def submit_and_wait(
    script: Path,
    export: str,
    chdir: Path | None,
    partition: str | None,
    reservation: str | None = None,
) -> tuple[int, str]:
    """Run `sbatch --parsable --wait` and return (returncode, job_id).

    `--parsable` makes stdout just the job id (optionally `<id>;<cluster>`);
    `--wait` blocks until the job reaches a terminal state. A non-empty
    `partition` is passed as `--partition`, overriding the script's
    `#SBATCH --partition` directive so one script runs on any cluster. A
    non-empty `reservation` is passed as `--reservation` to pin the job to a
    named SLURM reservation (e.g. dedicated CI nodes).
    """
    cmd = ["sbatch", "--parsable", "--wait", f"--export={export}"]
    if partition:
        cmd.append(f"--partition={partition}")
    if reservation:
        cmd.append(f"--reservation={reservation}")
    cmd.append(str(script))
    log(f"==> {' '.join(cmd)}")
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

    stdout = proc.stdout.strip()
    job_id = stdout.splitlines()[-1].split(";")[0] if stdout else ""
    return proc.returncode, job_id


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


def evaluate(sbatch_rc: int, job_id: str, result: JobResult) -> int:
    """Decide the overall exit code from the sbatch rc and sacct result."""
    if sbatch_rc != 0:
        log(f"ERROR: sbatch reported failure (rc={sbatch_rc})")
        return sbatch_rc

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
        description="Submit a SLURM job with sbatch --wait and verify it via sacct."
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
    args = parser.parse_args(argv)

    if not args.script.exists():
        parser.error(f"sbatch script not found: {args.script}")
    if args.chdir:
        args.chdir.mkdir(parents=True, exist_ok=True)

    sbatch_rc, job_id = submit_and_wait(
        args.script, args.export, args.chdir, args.partition, args.reservation
    )
    log(f"sbatch --wait rc={sbatch_rc}, job_id={job_id}")

    result = JobResult(state="", exit_code="")
    if sbatch_rc == 0 and job_id:
        result = query_job(job_id, args.poll_retries, args.poll_interval)

    return evaluate(sbatch_rc, job_id, result)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
