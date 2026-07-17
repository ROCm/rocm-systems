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

import slurm

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

    # Extra host->container bind mounts (comma/space-separated "host:ctr[:ro]").
    # Used e.g. on clusters where ROCm can't be installed in-container and the
    # host /opt/rocm is mounted instead (see ci_targets.yml ruby target).
    extra_volumes: str = field(default_factory=lambda: _env("EXTRA_VOLUMES", ""))

    rccl_dir: str = ""
    mnctl_dir: str = ""
    # Source strategy for rccl + rccl-tests:
    #   "clone" (default): sparse+shallow clone the monorepo at the coverage
    #     branch into src_dir on the login node, then mount that -> portable,
    #     no dependence on a host working copy / its perms / uid.
    #   "mount": legacy behavior -- bind-mount the existing host checkout
    #     (rccl_dir) as-is (used by CI, where actions/checkout pins the ref).
    src_mode: str = field(default_factory=lambda: _env("RCCL_SRC_MODE", "clone"))
    repo_url: str = field(default_factory=lambda: _env(
        "RCCL_REPO_URL", "https://github.com/ROCm/rocm-systems.git"))
    src_dir: str = ""
    run_id: str = ""
    container: str = ""
    hostfile: str = ""
    # Forces `mnctl --rebuild` (full `docker build --no-cache`).
    force_rebuild: bool = field(default_factory=lambda: _env("MNCTL_REBUILD", "0") == "1")

shared_fs_root: str = field(default_factory=lambda: os.path.expanduser(_env("SHARED_FS_ROOT") or "~"))
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
        # Modes are named after the SLURM command they use: "salloc" / "sbatch".
        # "new" is kept as a backward-compat alias for "salloc".
        self.alloc_mode = {"new": "salloc"}.get(self.alloc_mode, self.alloc_mode)

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

        # Managed clone location (clone mode). Must live on a filesystem visible
        # to every node (same requirement the bind-mount already had), so anchor
        # it to the shared-FS root when set, else the home dir.
        base = self.shared_fs_root or os.path.expanduser("~")
        self.src_dir = _env("RCCL_SRC_DIR") or os.path.join(
            base, ".docker-src", self.container)

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

    def reexec_vars(self) -> dict:
        """Resolved values propagated across the salloc/sbatch boundary so the
        second process (compute node) uses identical settings. ALLOC_MODE is owned
        by the slurm layer, which forces it to "inherit" for the child."""
        return dict(
            WORKLOAD=self.workload, RUN_ID=self.run_id,
            ROCM_IMAGE=self.rocm_image, DOCKERFILE=self.dockerfile, GPU_ARCH=self.gpu_arch,
            NIC_TYPE=self.nic_type, NODES=str(self.nodes), PARTITION=self.partition,
            ACCOUNT=self.account, GPUS_PER_NODE=str(self.gpus_per_node),
            TIME_LIMIT=self.time_limit, RESERVATION=self.reservation,
            MNCTL_CONTAINER_NAME=self.container, HOSTFILE=self.hostfile, RCCL_DIR=self.rccl_dir,
            MNCTL_DIR=self.mnctl_dir, SHARED_FS_ROOT=self.shared_fs_root,
            EXTRA_VOLUMES=self.extra_volumes,
        )

    def slurm_spec(self) -> "slurm.SlurmSpec":
        """Map the resolved SLURM fields onto the reusable slurm layer's spec.

        GRES follows the layer's None/""/value convention: unset -> default
        gpu:<N>, empty -> omit --gres, else verbatim."""
        gres = os.environ["GRES"] if "GRES" in os.environ else None
        return slurm.SlurmSpec(
            nodes=self.nodes, partition=self.partition, account=self.account,
            reservation=self.reservation, gpus_per_node=self.gpus_per_node,
            time_limit=self.time_limit, alloc_mode=self.alloc_mode,
            gres=gres, salloc_extra=_env("SALLOC_EXTRA", ""),
            job_name=f"rccl-{self.workload}-{self.gpu_arch}",
            run_id=self.run_id,
            run_dir=os.path.dirname(self.hostfile) or "~/.mnctl",
        )


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

    def maybe_allocate(self) -> None:
        """Acquire a SLURM allocation via the reusable slurm layer (selectable per
        run through ALLOC_MODE). May replace this process (salloc) or submit and
        wait (sbatch); a no-op for existing/inherit."""
        slurm.allocate(self.cfg.slurm_spec(), self.entrypoint, self.cfg.reexec_vars())

    def resolve_nodelist(self) -> None:
        c = self.cfg
        c.hosts, c.head = slurm.resolve_nodelist()
        log(f"Nodes: {' '.join(c.hosts)}  (head={c.head})")

    def write_hostfile(self) -> None:
        c = self.cfg
        for d in (c.shared_dir, c.builds_dir):
            os.makedirs(d, exist_ok=True)
        slurm.write_hostfile(c.hostfile, c.hosts, c.gpus_per_node)
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

    def _resolve_branch(self) -> str:
        return _env("RCCL_BRANCH") or self.cfg._git_branch() or "develop"

    def prepare_sources(self) -> None:
        """Sparse + shallow clone of the monorepo at the coverage branch.

        rccl and rccl-tests are subdirectories of one big monorepo
        (ROCm/rocm-systems), so a single sparse checkout of just those two paths
        gives both projects at the *same commit* (monorepo coherence for free)
        without pulling the whole tree.  The clone lands in src_dir on the login
        node and is bind-mounted into the containers, which makes the run
        identical on every cluster/Dockerfile: no dependence on a host working
        copy, its branch, its NFS perms, or the container UID remap.

        No-ops in "mount" mode (legacy bind-mount of an existing checkout).
        """
        c = self.cfg
        if c.src_mode != "clone":
            return
        # In CI, actions/checkout already provides the exact ref (incl. PR merge
        # refs and fork branches not on origin), so prefer it unless the caller
        # explicitly asked for clone via RCCL_SRC_MODE.
        if _env("GITHUB_ACTIONS") == "true" and not _env("RCCL_SRC_MODE"):
            log("CI detected; using actions/checkout sources (mount). "
                "Set RCCL_SRC_MODE=clone to override.")
            c.src_mode = "mount"
            return

        branch = self._resolve_branch()
        src = c.src_dir
        log(f"Preparing sources: sparse+shallow {c.repo_url}@{branch} -> {src}")

        if not os.path.isdir(os.path.join(src, ".git")):
            os.makedirs(src, exist_ok=True)
            self._run(["git", "clone", "--filter=blob:none", "--no-checkout",
                      "--depth", "1", "--branch", branch, c.repo_url, src])
            self._run(["git", "-C", src, "sparse-checkout", "set",
                      "projects/rccl", "projects/rccl-tests"])
            self._run(["git", "-C", src, "checkout", branch])
        else:
            # Managed clone (not a user's working copy): reset hard to the branch
            # tip so every run builds the latest pushed commit deterministically.
            self._run(["git", "-C", src, "fetch", "--depth", "1", "origin", branch])
            self._run(["git", "-C", src, "checkout", "-B", branch, "FETCH_HEAD"])

        # Repoint the mounted sources at the fresh clone. mnctl_dir is left alone
        # (the sparse clone has no rccl-utils; mnctl comes from MNCTL_DIR / repo).
        c.rccl_dir = os.path.join(src, "projects", "rccl")
        c.artifacts_pointer = os.path.join(c.rccl_dir, f".artifacts_{c.run_id}")
        rccl_tests = os.path.join(src, "projects", "rccl-tests")
        if hasattr(self.payload, "rccl_tests_dir"):
            self.payload.rccl_tests_dir = rccl_tests
        log(f"Sources ready: {c.rccl_dir} + {rccl_tests}")

    def launch_containers(self) -> None:
        c = self.cfg
        mounts = [f"--volume {shlex.quote(c.rccl_dir + ':/work/rccl')}"] + self.payload.mounts(self)
        # Extra bind mounts (e.g. host /opt/rocm on clusters without in-container ROCm).
        for vol in c.extra_volumes.replace(",", " ").split():
            mounts.append(f"--volume {shlex.quote(vol)}")
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
        self.prepare_sources()  # clone mode: sparse checkout before validate/build
        self.core_validate()
        self.payload.validate(self)
        self.maybe_allocate()   # may replace this process or submit+wait via sbatch
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
