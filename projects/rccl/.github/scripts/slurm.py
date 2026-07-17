"""Reusable SLURM allocation layer.

Scheduler-specific but workload-, mnctl-, and RCCL-agnostic: turns a SlurmSpec
into a set of compute nodes plus an MPI-style hostfile, independent of what runs
on them. The rest of the pipeline only consumes a hostfile, so anything that can
produce one (a different scheduler, a static node list, ...) can replace this
module without touching the orchestrator.

Selectable per-run allocation modes (ALLOC_MODE):
  auto     - salloc a fresh allocation unless one is already visible
  salloc   - always salloc a fresh allocation (re-exec self inside it); "new" alias
  sbatch   - submit a batch job (sbatch --wait) and stream its output
  existing - reuse the current/newest allocation (no new alloc)
  inherit  - internal: set on the re-exec'd child so it does not re-allocate

Boundary crossing: the caller passes an *entrypoint* (a script re-exec'd inside
the allocation) and a dict of *carry-over* env vars to propagate to that child.
This module owns the ALLOC_MODE variable and forces it to "inherit" for the
child, so the second process attaches to the allocation instead of making a new
one.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Env var (owned by this layer) that selects the allocation mode. Set to INHERIT
# on the re-exec'd child so it reuses the allocation rather than re-allocating.
ALLOC_MODE_ENV = "ALLOC_MODE"
INHERIT = "inherit"

# Modes are named after the SLURM command they use; "new" is a legacy alias.
_ALIAS = {"new": "salloc"}
_ALLOCATING = ("salloc", "sbatch")


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class SlurmSpec:
    """Everything needed to request (or attach to) a SLURM allocation.

    `gres` uses a three-way convention: None means "default to gpu:<gpus_per_node>",
    "" means "omit --gres entirely" (some clusters reject `gpu:N`), and any other
    string is passed through verbatim.
    """

    nodes: int
    partition: str
    account: str = ""
    reservation: str = ""
    gpus_per_node: int = 8
    time_limit: str = "04:00:00"
    alloc_mode: str = "auto"
    gres: Optional[str] = None
    salloc_extra: str = ""
    job_name: str = "job"
    run_id: str = "run"
    run_dir: str = "~/.mnctl"           # where the sbatch script/output land

    def __post_init__(self) -> None:
        self.alloc_mode = _ALIAS.get(self.alloc_mode, self.alloc_mode)

    @classmethod
    def from_env(cls, **overrides) -> "SlurmSpec":
        """Build a spec from the environment (NODES/PARTITION/...); `overrides`
        win over env values. `gres` follows the None/""/value convention above:
        unset GRES -> None (default), GRES="" -> omit, GRES=... -> verbatim."""
        gres = os.environ["GRES"] if "GRES" in os.environ else None
        env_spec = dict(
            nodes=int(_env("NODES", "2")),
            partition=_env("PARTITION", ""),
            account=_env("ACCOUNT", ""),
            reservation=_env("RESERVATION", ""),
            gpus_per_node=int(_env("GPUS_PER_NODE", "8")),
            time_limit=_env("TIME_LIMIT", "04:00:00"),
            alloc_mode=_env(ALLOC_MODE_ENV, "auto"),
            gres=gres,
            salloc_extra=_env("SALLOC_EXTRA", ""),
        )
        env_spec.update(overrides)
        return cls(**env_spec)

    def gres_string(self) -> str:
        return f"gpu:{self.gpus_per_node}" if self.gres is None else self.gres

    def alloc_flags(self) -> List[str]:
        """SLURM flags shared by salloc and sbatch."""
        flags = ["-N", str(self.nodes), "-p", self.partition]
        if self.account:                    # some clusters (e.g. ruby) use no -A
            flags += ["-A", self.account]
        gres = self.gres_string()
        if gres:
            flags.append(f"--gres={gres}")
        flags += [f"--ntasks-per-node={self.gpus_per_node}", "-t", self.time_limit]
        if self.reservation:
            flags.append(f"--reservation={self.reservation}")
        if self.salloc_extra:
            flags += shlex.split(self.salloc_extra)
        return flags


def _running_job() -> str:
    """Newest running job id for $USER, or "" if none."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-t", "R", "-h", "--sort=-i", "-o", "%i"],
                         capture_output=True, text=True)
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""


def need_allocation(spec: SlurmSpec) -> bool:
    """True if we must create/enter an allocation for this run."""
    return spec.alloc_mode in _ALLOCATING or (
        spec.alloc_mode == "auto"
        and not os.environ.get("SLURM_JOB_NODELIST")
        and not _running_job()
    )


def allocate(spec: SlurmSpec, entrypoint: str, carry_vars: dict) -> None:
    """Acquire a SLURM allocation per `spec.alloc_mode`.

    - salloc / auto(new): re-exec this process under `salloc` (never returns).
    - sbatch: submit a batch job, stream its output, then exit with its code.
    - existing / inherit: no-op (reuse the current/newest allocation).

    `carry_vars` are propagated to the child with ALLOC_MODE forced to "inherit".
    """
    if not need_allocation(spec):
        return
    if spec.alloc_mode == "sbatch":
        _submit_sbatch(spec, entrypoint, carry_vars)   # blocks, then exits
    else:
        _salloc_reexec(spec, entrypoint, carry_vars)   # replaces this process


def _child_env(carry_vars: dict) -> dict:
    env = dict(os.environ)
    env.update(carry_vars)
    env[ALLOC_MODE_ENV] = INHERIT
    return env


def _salloc_reexec(spec: SlurmSpec, entrypoint: str, carry_vars: dict) -> None:
    log(f"Allocating {spec.nodes} node(s) via salloc on partition={spec.partition} "
        f"account={spec.account or '(none)'} ...")
    inner = f"exec python3 {shlex.quote(entrypoint)}"
    salloc = ["salloc", *spec.alloc_flags(), "bash", "-c", inner]
    os.execvpe("salloc", salloc, _child_env(carry_vars))  # replaces this process


def _submit_sbatch(spec: SlurmSpec, entrypoint: str, carry_vars: dict) -> None:
    rundir = os.path.expanduser(spec.run_dir)
    os.makedirs(rundir, exist_ok=True)
    script_path = os.path.join(rundir, f"{spec.run_id}.sbatch.sh")
    logfile = os.path.join(rundir, f"{spec.run_id}.sbatch.out")
    # Re-export carry-over values and force ALLOC_MODE=inherit so the batch job
    # (which inherits the submit env via --export=ALL) does not submit again.
    exports = {**carry_vars, ALLOC_MODE_ENV: INHERIT}
    export_lines = "\n".join(f"export {k}={shlex.quote(v)}" for k, v in exports.items())
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n"
                f"{export_lines}\n"
                f"exec python3 {shlex.quote(entrypoint)}\n")
    os.chmod(script_path, 0o755)
    sbatch = ["sbatch", "--wait", f"--output={logfile}",
              f"--job-name={spec.job_name}", *spec.alloc_flags(), script_path]
    log(f"Submitting sbatch job ({spec.nodes} node(s), partition={spec.partition}, "
        f"account={spec.account or '(none)'}); output -> {logfile}")
    log("$ " + " ".join(shlex.quote(a) for a in sbatch))
    open(logfile, "w").close()               # ensure it exists so tail can follow
    tail = subprocess.Popen(["tail", "-n", "+1", "-F", logfile])
    try:
        rc = subprocess.run(sbatch).returncode
    finally:
        time.sleep(1)                        # let tail flush the final lines
        tail.terminate()
    sys.exit(rc)


def resolve_nodelist() -> Tuple[List[str], str]:
    """Return (hosts, head) for the current or newest running allocation."""
    nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
    if not nodelist:
        out = subprocess.run(["squeue", "-u", os.environ["USER"], "-t", "R", "-h", "-o", "%i %N"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
        if not out:
            sys.exit("ERROR: no running SLURM allocation found")
        jobid, nodelist = max((line.split(maxsplit=1) for line in out), key=lambda t: int(t[0]))
        log(f"Attaching to existing job {jobid}")
    names = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                           capture_output=True, text=True, check=True).stdout.split()
    return names, names[0]


def write_hostfile(path: str, hosts: List[str], gpus_per_node: int) -> None:
    """Write an MPI-style hostfile (`<node> slots=<gpus>`)."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for h in hosts:
            f.write(f"{h} slots={gpus_per_node}\n")
