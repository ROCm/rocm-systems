"""Workload-agnostic multi-node GPU orchestration (Python spike).

Parallel, stdlib-only reimplementation of lib/orchestrate.sh for head-to-head
comparison. Allocates SLURM GPU nodes, writes an MPI hostfile, logs in to the
registry, launches a multi-node ROCm container via mnctl, hands off to a Payload
for the in-container work, then collects artifacts and tears everything down.

Design notes vs. the bash version:
  * Inputs are a typed RunConfig (no `${VAR:-default}` sprawl, no export list).
  * Commands run via subprocess with list args where possible; remote/in-container
    steps are explicit shell strings assembled with shlex.quote (no ssh->docker->
    bash triple-quoting gymnastics, and no `set -e` return-code footguns).
  * mnctl is still driven through its stable `python3 -m mnctl` CLI (a later step
    could import mnctl directly for deeper reuse).
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, fields
from typing import List, Optional

DEFAULT_ROCM_IMAGE = (
    "registry-sc-harbor.amd.com/framework/therock-release:"
    "47_gfx94X_7.13.0rc2_ubuntu24.04_py3.12_pytorch_release-2.11_96bfee1"
)
SSH_OPTS = ["-o", "StrictHostKeyChecking=no"]


def log(msg: str) -> None:
    print(f">>> {msg}", flush=True)


def _sanitize_id(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^A-Za-z0-9._-]", "-", text)).strip("-")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class RunConfig:
    """All run inputs, resolved from the environment with sensible defaults."""

    workload: str = ""
    rocm_image: str = field(default_factory=lambda: _env("ROCM_IMAGE", DEFAULT_ROCM_IMAGE))
    dockerfile: str = field(default_factory=lambda: _env("DOCKERFILE", "Dockerfile.Multinode.Ubuntu"))
    gpu_arch: str = field(default_factory=lambda: _env("GPU_ARCH", "gfx942"))
    nic_type: str = field(default_factory=lambda: _env("NIC_TYPE", "mellanox"))
    nodes: int = field(default_factory=lambda: int(_env("NODES", "2")))

    partition: str = field(default_factory=lambda: _env("PARTITION", "rccl"))
    account: str = field(default_factory=lambda: _env("ACCOUNT", "rccl"))
    gpus_per_node: int = field(default_factory=lambda: int(_env("GPUS_PER_NODE", "8")))
    time_limit: str = field(default_factory=lambda: _env("TIME_LIMIT", "04:00:00"))
    reservation: str = field(default_factory=lambda: _env("RESERVATION", ""))
    alloc_mode: str = field(default_factory=lambda: _env("ALLOC_MODE", "auto"))

    rccl_dir: str = ""
    mnctl_dir: str = ""
    run_id: str = ""
    container: str = ""
    hostfile: str = ""
    force_rebuild: bool = field(default_factory=lambda: _env("FORCE_REBUILD", "0") == "1")

    shared_fs_root: str = field(default_factory=lambda: _env("SHARED_FS_ROOT", os.path.expanduser("~")))
    shared_dir: str = ""
    builds_dir: str = ""

    artifacts_pointer: str = ""          # host path
    artifacts_pointer_ctr: str = ""      # container path
    artifact_env_var: str = "RESULT_ARTIFACT_DIR"

    # Populated after node resolution.
    hosts: List[str] = field(default_factory=list)
    head: str = ""

    @property
    def np(self) -> int:
        return self.nodes * self.gpus_per_node

    def resolve(self, entrypoint: str, workload_tag: str) -> None:
        # entrypoint = .github/scripts/run_workload.py -> RCCL_DIR is 2 levels up
        # (scripts -> .github -> projects/rccl).
        self.rccl_dir = self.rccl_dir or _env("RCCL_DIR") or os.path.abspath(
            os.path.join(os.path.dirname(entrypoint), "..", "..")
        )
        self.mnctl_dir = self.mnctl_dir or _env("MNCTL_DIR") or os.path.abspath(
            os.path.join(self.rccl_dir, "..", "..", "rccl-utils", "MultiNodeDocker")
        )
        branch = _env("RCCL_BRANCH") or self._git_branch()
        pr = _env("PR_NUMBER")
        # ROCm version is parsed from the image tag; if it can't be derived, skip it
        # entirely (keeps the name shorter) rather than inserting a placeholder.
        parts = [self.gpu_arch]
        rocm_ver = self._rocm_version()
        if rocm_ver:
            parts.append(rocm_ver)
        parts.append(branch or "nobranch")
        base = _sanitize_id("-".join(parts))
        self.run_id = _env("RUN_ID") or (base + (f"-pr{pr}" if pr else ""))

        self.container = _env("MNCTL_CONTAINER_NAME") or f"rccl-{workload_tag}-{self.gpu_arch}"
        self.hostfile = _env("HOSTFILE") or os.path.expanduser(f"~/.mnctl/{self.run_id}.hostfile")
        self.shared_dir = _env("SHARED_DIR") or f"{self.shared_fs_root}/.docker-shared/{self.container}"
        self.builds_dir = _env("BUILDS_DIR") or f"{self.shared_fs_root}/.docker-builds/{self.container}"
        self.artifacts_pointer = os.path.join(self.rccl_dir, f".artifacts_{self.run_id}")
        self.artifacts_pointer_ctr = f"/work/rccl/.artifacts_{self.run_id}"

    def _git_branch(self) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", self.rccl_dir, "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, check=True,
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    def _rocm_version(self) -> str:
        m = re.search(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9]*", self.rocm_image)
        return m.group(0) if m else ""

    def child_env(self) -> dict:
        """Env for the salloc re-exec (mirror the bash export list)."""
        env = dict(os.environ)
        env.update(
            WORKLOAD=self.workload, ALLOC_MODE="inherit", RUN_ID=self.run_id,
            ROCM_IMAGE=self.rocm_image, DOCKERFILE=self.dockerfile, GPU_ARCH=self.gpu_arch,
            NIC_TYPE=self.nic_type, NODES=str(self.nodes), PARTITION=self.partition,
            ACCOUNT=self.account, GPUS_PER_NODE=str(self.gpus_per_node),
            TIME_LIMIT=self.time_limit, RESERVATION=self.reservation,
            CONTAINER=self.container, HOSTFILE=self.hostfile, RCCL_DIR=self.rccl_dir,
            MNCTL_DIR=self.mnctl_dir, SHARED_FS_ROOT=self.shared_fs_root,
        )
        return env


class Orchestrator:
    def __init__(self, cfg: RunConfig, payload: "Payload", entrypoint: str):
        self.cfg = cfg
        self.payload = payload
        self.entrypoint = entrypoint
        self._cleaned = False
        self._collected = False

    # --- command helpers ----------------------------------------------------
    def _run(self, args: List[str], check: bool = True, quiet: bool = False,
             capture: bool = False) -> subprocess.CompletedProcess:
        if not quiet:
            log("$ " + " ".join(shlex.quote(a) for a in args))
        return subprocess.run(args, check=check, text=True,
                              capture_output=capture)

    def ssh(self, host: str, remote: str, check: bool = True,
            capture: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
        return self._run(["ssh", *SSH_OPTS, host, remote], check=check,
                        capture=capture, quiet=quiet)

    def ssh_head(self, remote: str, **kw) -> subprocess.CompletedProcess:
        return self.ssh(self.cfg.head, remote, **kw)

    # --- flow ---------------------------------------------------------------
    def core_validate(self) -> None:
        if not (self.cfg.mnctl_dir and os.path.isdir(os.path.join(self.cfg.mnctl_dir, "mnctl"))):
            sys.exit(f"ERROR: mnctl not found at {self.cfg.mnctl_dir!r}; set MNCTL_DIR.")

    def maybe_salloc_reexec(self) -> None:
        c = self.cfg
        need = c.alloc_mode == "new" or (
            c.alloc_mode == "auto" and not os.environ.get("SLURM_JOB_NODELIST")
            and not self._running_job()
        )
        if not need:
            return
        log(f"Allocating {c.nodes} node(s) on partition={c.partition} account={c.account} ...")
        inner = f"exec python3 {shlex.quote(self.entrypoint)}"
        salloc = ["salloc", "-N", str(c.nodes), "-p", c.partition]
        if c.account:                       # some clusters (e.g. ruby) use no -A
            salloc += ["-A", c.account]
        salloc += [
            f"--gres=gpu:{c.gpus_per_node}", f"--ntasks-per-node={c.gpus_per_node}",
            "-t", c.time_limit,
        ]
        if c.reservation:
            salloc.append(f"--reservation={c.reservation}")
        salloc += ["bash", "-c", inner]
        os.execvpe("salloc", salloc, c.child_env())  # replaces this process

    def _running_job(self) -> str:
        out = subprocess.run(["squeue", "-u", os.environ["USER"], "-t", "R", "-h", "-o", "%i"],
                            capture_output=True, text=True)
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""

    def resolve_nodelist(self) -> None:
        c = self.cfg
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            out = subprocess.run(["squeue", "-u", os.environ["USER"], "-t", "R", "-h", "-o", "%i %N"],
                                capture_output=True, text=True).stdout.strip().splitlines()
            if not out:
                sys.exit("ERROR: no running SLURM allocation found")
            jobid, nodelist = sorted(out, reverse=True)[0].split(maxsplit=1)
            log(f"Attaching to existing job {jobid}")
        names = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                              capture_output=True, text=True, check=True).stdout.split()
        c.hosts, c.head = names, names[0]
        log(f"Nodes: {' '.join(c.hosts)}  (head={c.head})")

    def write_hostfile(self) -> None:
        c = self.cfg
        for d in (c.shared_dir, c.builds_dir, os.path.dirname(c.hostfile)):
            os.makedirs(d, exist_ok=True)
        with open(c.hostfile, "w") as f:
            for h in c.hosts:
                f.write(f"{h} slots={c.gpus_per_node}\n")
        log(f"Run ID: {c.run_id}")
        log(f"Hostfile: {c.hostfile}")
        print(open(c.hostfile).read(), end="")

    def registry_login(self) -> None:
        user, token = _env("REGISTRY_USER"), _env("REGISTRY_TOKEN")
        if not (user and token):
            return
        host = _env("REGISTRY_HOST") or self.cfg.rocm_image.split("/", 1)[0]
        log(f"Logging in to {host} as {user} on all nodes ...")
        for h in self.cfg.hosts:
            p = subprocess.run(["ssh", *SSH_OPTS, h,
                               f"docker login {shlex.quote(host)} -u {shlex.quote(user)} --password-stdin"],
                              input=token, text=True, capture_output=True)
            if p.returncode != 0:
                sys.exit(f"ERROR: docker login failed on {h}")

    def preflight_pull(self) -> None:
        log(f"Preflight: verifying registry access to {self.cfg.rocm_image} on all nodes ...")
        failed = False
        for h in self.cfg.hosts:
            p = self.ssh(h, f"docker pull {shlex.quote(self.cfg.rocm_image)}",
                        check=False, capture=True, quiet=True)
            if p.returncode != 0:
                log(f"ERROR: image pull failed on {h}:\n" + (p.stderr or "").strip()[-500:])
                failed = True
        if failed:
            sys.exit("Registry access failed. Run 'docker login' on the affected node(s) and retry.")
        log("Preflight OK: base image reachable on all nodes.")

    def stage_mnctl(self) -> None:
        c = self.cfg
        if self.ssh_head(f"test -d {shlex.quote(c.mnctl_dir + '/mnctl')}",
                        check=False, quiet=True).returncode == 0:
            return
        staged = f"{c.shared_fs_root}/.mnctl-src/MultiNodeDocker"
        log(f"mnctl not visible on {c.head}; staging {c.mnctl_dir} -> {staged} (shared FS) ...")
        os.makedirs(os.path.dirname(staged), exist_ok=True)
        self._run(["rsync", "-a", "--delete", c.mnctl_dir + "/", staged + "/"])
        c.mnctl_dir = staged

    def launch_containers(self) -> None:
        c = self.cfg
        mounts = [f"--volume {shlex.quote(c.rccl_dir + ':/work/rccl')}"] + self.payload.mounts(self)
        rebuild = "--rebuild" if c.force_rebuild else ""
        log("Launching containers via mnctl ...")
        cmd = (
            f"cd {shlex.quote(c.mnctl_dir)} && python3 -m mnctl --launch-all --ssh "
            f"--hostfile {shlex.quote(c.hostfile)} --rocm-image {shlex.quote(c.rocm_image)} "
            f"--dockerfile {shlex.quote(c.dockerfile)} --gpu-targets {shlex.quote(c.gpu_arch)} "
            f"--nic-type {shlex.quote(c.nic_type)} --name {shlex.quote(c.container)} "
            f"--shared-dir {shlex.quote(c.shared_dir)} --builds-dir {shlex.quote(c.builds_dir)} "
            f"--shared-fs yes {rebuild} " + " ".join(mounts)
        )
        self.ssh_head(cmd)

    def collect_artifacts(self) -> None:
        if self._collected:
            return
        self._collected = True
        c = self.cfg
        art = ""
        if os.path.isfile(c.artifacts_pointer) and os.path.getsize(c.artifacts_pointer):
            art = os.path.basename(open(c.artifacts_pointer).readline().strip())
        if not art:
            import glob
            matches = sorted(glob.glob(os.path.join(c.rccl_dir, self.payload.artifact_glob(self))))
            art = os.path.basename(matches[-1]) if matches else ""
        if not art:
            log("WARNING: no artifacts dir found (run may have failed before creating it).")
            return
        uid, gid = os.getuid(), os.getgid()
        log(f"Fixing artifacts ownership to {uid}:{gid} in the container ...")
        self.ssh_head(
            f"docker exec {shlex.quote(c.container)} bash -lc "
            + shlex.quote(f"chown -R {uid}:{gid} /work/rccl/{art}"),
            check=False, quiet=True,
        )
        log(f"Artifacts: {c.rccl_dir}/{art}")
        gh = os.environ.get("GITHUB_ENV")
        if gh:
            with open(gh, "a") as f:
                f.write(f"{c.artifact_env_var}={art}\n")

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.collect_artifacts()
        c = self.cfg
        if not c.head:
            return
        log("Tearing down containers ...")
        self.ssh_head(f"cd {shlex.quote(c.mnctl_dir)} && python3 -m mnctl --stop-all "
                     f"--hostfile {shlex.quote(c.hostfile)}", check=False, quiet=True)
        # Safety-net removal; silence docker's container-name echo (redirect on the
        # remote so it never reaches our stdout).
        for h in c.hosts:
            self.ssh(h, f"docker rm -f {shlex.quote(c.container)} >/dev/null 2>&1 || true",
                    check=False, quiet=True)
        log(f"Removed container {c.container} on {len(c.hosts)} node(s).")

    def run(self) -> None:
        c = self.cfg
        self.core_validate()
        self.payload.validate(self)
        self.maybe_salloc_reexec()   # may replace this process
        self.resolve_nodelist()
        self.write_hostfile()
        self.payload.prelaunch(self)
        try:
            self.registry_login()
            self.preflight_pull()
            self.stage_mnctl()
            self.launch_containers()
            self.payload.prepare(self)
            if os.path.exists(c.artifacts_pointer):
                os.remove(c.artifacts_pointer)
            self.payload.run(self)
            self.collect_artifacts()
        finally:
            self.cleanup()
