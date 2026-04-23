"""Docker container runtime implementation.

Implements ``ContainerRuntime`` for the Docker CLI, including full device
passthrough, InfiniBand library bind-mounting, and idempotent container
management.
"""

import glob
import os
import re
import subprocess
import time
from typing import Dict, List, Optional

from .runtime import ContainerRuntime
from .utils import error, log, log_verbose, Timer


# ---------------------------------------------------------------------------
# Helpers (module-level, not part of the class interface)
# ---------------------------------------------------------------------------
def _get_render_gid():
    # type: () -> str
    try:
        import grp
        return str(grp.getgrnam("render").gr_gid)
    except KeyError:
        return "109"


def _bind_mount_rdma_libs(args):
    # type: (List[str]) -> None
    """Bind-mount host MLNX_OFED / rdma-core libraries into the container."""
    ib_lib_dir = "/usr/lib/x86_64-linux-gnu"
    for lib in ("libibverbs", "libmlx5", "libmlx4", "libefa"):
        for f in sorted(glob.glob("{}/{}.so*".format(ib_lib_dir, lib))):
            if os.path.exists(f):
                args += ["-v", "{}:{}:ro".format(f, f)]

    ib_provider = "{}/libibverbs".format(ib_lib_dir)
    if os.path.isdir(ib_provider):
        args += ["-v", "{}:{}:ro".format(ib_provider, ib_provider)]
    if os.path.isdir("/etc/libibverbs.d"):
        args += ["-v", "/etc/libibverbs.d:/etc/libibverbs.d:ro"]

    log_verbose("RDMA libs bind-mounted from host")


def _docker_output(cmd):
    # type: (List[str]) -> str
    """Run a docker command and return stripped stdout."""
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _resolve_dockerfile_base(dockerfile_path):
    # type: (str) -> Optional[str]
    """Parse a Dockerfile to resolve the actual base image from ARG defaults.

    Handles patterns like:
        ARG ROCM_IMAGE=some/image:tag
        FROM ${ROCM_IMAGE}
    and:
        ARG ROCM_IMAGE_NAME=some/image
        ARG ROCM_IMAGE_TAG=1.0
        FROM "${ROCM_IMAGE_NAME}:${ROCM_IMAGE_TAG}"
    """
    try:
        with open(dockerfile_path, "r") as f:
            lines = f.readlines()
    except (IOError, OSError):
        return None

    args = {}  # type: Dict[str, str]
    from_line = None

    for line in lines:
        stripped = line.strip()
        m = re.match(r"^ARG\s+(\w+)=(.+)$", stripped)
        if m:
            args[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        if stripped.upper().startswith("FROM "):
            from_line = stripped[5:].strip().strip('"').strip("'")
            break

    if from_line is None:
        return None

    def _sub(match):
        # type: (re.Match) -> str
        name = match.group(1)
        return args.get(name, match.group(0))

    resolved = re.sub(r"\$\{(\w+)\}", _sub, from_line)
    resolved = re.sub(r"\$(\w+)", _sub, resolved)
    return resolved if "$" not in resolved else None


# ---------------------------------------------------------------------------
# Host -> container environment-variable mapping
# ---------------------------------------------------------------------------
def _container_env_pairs(cfg, render_gid):
    # type: (object, str) -> List[tuple]
    """Build the (NAME, value) pairs injected as ``-e`` flags on ``docker run``.

    This is the **single source of truth** for how mnctl's host-side
    settings reach the container's entrypoint and post-setup scripts.
    The naming convention is intentional:

      * On the **host**, all mnctl-controlled settings are namespaced
        ``MNCTL_*`` (e.g. ``MNCTL_GPUS``, ``MNCTL_NIC_TYPE``) so they
        cannot collide with the user's existing environment.
      * Inside the **container**, the prefix is dropped because these
        variables are consumed by user-maintained shell scripts
        (entrypoint, post-setup) where short, conventional names are
        more ergonomic. The container side is "private" to mnctl, so
        collision is not a concern.

    Mapping reference (host -> container)::

        MNCTL_GPUS         -> GPUS
        MNCTL_VERBOSE      -> VERBOSE             (only set when truthy)
        MNCTL_NIC_TYPE     -> NIC_TYPE
        MNCTL_GPU_TARGETS  -> GPU_TARGETS         (only set when non-empty)
        (derived)          -> HOST_UID, HOST_GID, RENDER_GID
        (derived)          -> FORCE_POST_SETUP    (when --rebuild/--replace)

    To add a new pair: append it here AND document it in the epilog of
    ``__main__.py`` and (for end-user-visible settings) in
    ``post-setup/`` script docs.
    """
    pairs = [
        ("GPUS", cfg.gpus),
        ("HOST_UID", str(os.getuid())),
        ("HOST_GID", str(os.getgid())),
        ("RENDER_GID", render_gid),
        ("NIC_TYPE", cfg.nic_type),
    ]
    if cfg.verbose:
        pairs.append(("VERBOSE", "1"))
    if cfg.gpu_targets:
        pairs.append(("GPU_TARGETS", cfg.gpu_targets))
    if cfg.force_rebuild or cfg.force_replace:
        pairs.append(("FORCE_POST_SETUP", "1"))
    return pairs


# ---------------------------------------------------------------------------
# DockerRuntime
# ---------------------------------------------------------------------------
class DockerRuntime(ContainerRuntime):
    """Docker CLI-based container runtime."""

    @property
    def name(self):
        # type: () -> str
        return "docker"

    # --- Image management ---

    def image_exists(self, tag=None):
        # type: (Optional[str]) -> bool
        """Return True if *tag* (or ``cfg.image_tag``) is present locally."""
        target = tag or self.cfg.image_tag
        result = subprocess.run(
            ["docker", "image", "inspect", target],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.returncode == 0

    def build_image(self):
        # type: () -> None
        dockerfile_path = os.path.join(
            self.cfg.script_dir, self.cfg.dockerfile
        )
        self._sync_base_from_dockerfile(dockerfile_path)

        if not self.cfg.force_rebuild and self.image_exists():
            log(
                "=== Image {} already exists "
                "(use --rebuild to force) ===".format(self.cfg.image_tag)
            )
            if self.cfg.verbose:
                details = self.image_details()
                if details:
                    log_verbose("Image details: {}".format(details))
            log("")
            return

        with Timer("Image build"):
            log("=== Building image ===")
            log("  Base       : {}".format(self.cfg.rocm_image))
            log("  Dockerfile : {}".format(self.cfg.dockerfile))
            log("  Tag        : {}".format(self.cfg.image_tag))
            log("")

            self._ensure_base_image(self.cfg.rocm_image)

            cmd = [
                "docker", "build",
                "--build-arg", "SSH_PORT={}".format(self.cfg.ssh.port),
                "-t", self.cfg.image_tag,
                "-f", dockerfile_path,
            ]

            if self.cfg.force_rebuild:
                cmd.append("--no-cache")

            if self.cfg.rocm_image_explicit:
                cmd += [
                    "--build-arg",
                    "ROCM_IMAGE={}".format(self.cfg.rocm_image),
                ]

            if self.cfg.gpu_targets:
                cmd += [
                    "--build-arg",
                    "GPU_TARGETS={}".format(self.cfg.gpu_targets),
                ]

            if self.cfg.verbose:
                cmd.append("--progress=plain")
                log_verbose(
                    "docker build {}".format(
                        " ".join(cmd + [self.cfg.script_dir])
                    )
                )

            cmd.append(self.cfg.script_dir)
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                log("")
                error("Image build failed (exit {})".format(e.returncode))
                log("  Check the Docker build output above for details.")
                log("  Common fixes:")
                log("    - Ensure Docker daemon is running")
                log("    - Check Dockerfile syntax and base image availability")
                log("    - Retry: python3 -m mnctl --rebuild")
                raise SystemExit(1)

        log("")
        log("  Built: {}".format(self.cfg.image_tag))
        if self.cfg.verbose:
            log_verbose(
                "Image ID: {}".format(
                    _docker_output([
                        "docker", "image", "inspect", self.cfg.image_tag,
                        "--format", "{{.Id}}",
                    ])[:12]
                )
            )
        log("")

    def image_details(self):
        # type: () -> Optional[str]
        text = _docker_output([
            "docker", "image", "inspect", self.cfg.image_tag,
            "--format", "{{.Id}} created={{.Created}} size={{.Size}}",
        ])
        return text or None

    # --- Container lifecycle ---

    def launch(self):
        # type: () -> None
        cfg = self.cfg

        existing = _docker_output(
            ["docker", "ps", "-a", "--format", "{{.Names}}"]
        ).splitlines()

        if cfg.container_name in existing:
            state = _docker_output([
                "docker", "inspect", cfg.container_name,
                "--format", "{{.State.Status}}",
            ])

            if cfg.force_rebuild or cfg.force_replace:
                log(
                    "=== Replacing container '{}' ({}) ===".format(
                        cfg.container_name,
                        "--rebuild" if cfg.force_rebuild else "--replace",
                    )
                )
                self._remove_container()
            elif state == "running":
                log(
                    "=== Container '{}' is already running "
                    "(use --rebuild to replace) ===".format(cfg.container_name)
                )
                log_verbose(
                    "Container ID: {}".format(
                        _docker_output([
                            "docker", "inspect", cfg.container_name,
                            "--format", "{{.Id}}",
                        ])[:12]
                    )
                )
                return
            else:
                log(
                    "=== Container '{}' exists but is {}, "
                    "removing and re-launching ===".format(
                        cfg.container_name, state
                    )
                )
                self._remove_container()

        with Timer("Container launch"):
            log("=== Launching container ===")
            log("  Image     : {}".format(cfg.image_tag))
            log("  Container : {}".format(cfg.container_name))
            log("  GPUs      : {}".format(cfg.gpus))
            log_verbose("SSH port  : {}".format(cfg.ssh.port))
            log_verbose("SHM size  : {}".format(cfg.shm_size))
            log("")

            run_args = self._assemble_run_args()

            if cfg.verbose:
                log_verbose("docker run arguments:")
                for arg in run_args:
                    log_verbose("  {}".format(arg))

            cmd = ["docker", "run"] + run_args + [cfg.image_tag]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                log("")
                error(
                    "Container launch failed (exit {})".format(e.returncode)
                )
                log("  Troubleshooting:")
                log("    docker image inspect {}".format(cfg.image_tag))
                log("    docker rm -f {}".format(cfg.container_name))
                log("    python3 -m mnctl --run --rebuild")
                raise SystemExit(1)

        self._print_launch_summary()

    def get_stop_cmd(self):
        # type: () -> str
        return (
            "docker rm -f {name} 2>/dev/null "
            "&& echo 'removed' || echo 'not running'"
        ).format(name=self.cfg.container_name)

    # --- Ephemeral build tasks ---

    def run_build_task(self, script, log_path):
        # type: (str, str) -> int
        with open(log_path, "w") as lf:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", "{}:/opt/shared".format(self.cfg.shared_dir),
                    self.cfg.image_tag,
                    "bash", "-c", script,
                ],
                stdout=lf, stderr=subprocess.STDOUT,
            )
        return result.returncode

    # --- Private helpers ---

    def _sync_base_from_dockerfile(self, dockerfile_path):
        # type: (str) -> None
        """Update cfg.rocm_image from the Dockerfile when not user-specified.

        Parses ARG/FROM defaults so that ``image_tag`` and
        ``_ensure_base_image`` use the correct base for any Dockerfile.
        """
        if self.cfg.rocm_image_explicit:
            return
        parsed = _resolve_dockerfile_base(dockerfile_path)
        if parsed:
            log_verbose(
                "Resolved base image from {}: {}".format(
                    self.cfg.dockerfile, parsed
                )
            )
            self.cfg.rocm_image = parsed

    def _ensure_base_image(self, base):
        # type: (str) -> None
        """Verify the base image is available locally; pull if needed."""
        if self.image_exists(base):
            log_verbose("Base image '{}' found locally".format(base))
            return

        log("  Base image '{}' not found locally, pulling...".format(base))
        pull = subprocess.run(
            ["docker", "pull", base],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if pull.returncode == 0:
            log("  Pulled '{}'".format(base))
            return

        log("")
        error("Base image '{}' is not available".format(base))
        log("  It does not exist locally and could not be pulled.")
        log("")
        log("  The Dockerfile ({}) requires this image in its".format(
            self.cfg.dockerfile
        ))
        log("  FROM directive. You must make it available first:")
        log("")
        log("  Options:")
        log("    1. Pull from a registry:")
        log("       docker pull {}".format(base))
        log("    2. Build it locally (if you have a Dockerfile for it):")
        log("       docker build -t {} -f <Dockerfile> .".format(base))
        log("    3. Use a different base image:")
        log("       python3 -m mnctl <other-image>")
        raise SystemExit(1)

    def _remove_container(self):
        # type: () -> None
        subprocess.run(
            ["docker", "rm", "-f", self.cfg.container_name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def _wait_for_entrypoint(self, timeout=600):
        # type: (int) -> None
        """Follow container logs until the entrypoint prints '=== Ready ===' or times out."""
        cfg = self.cfg
        log("")
        log("  Waiting for entrypoint to finish (timeout {}s) ...".format(
            timeout,
        ))

        proc = subprocess.Popen(
            ["docker", "logs", "--follow", cfg.container_name],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        start = time.time()
        ready = False
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
                log("    {}".format(line))
                if "=== Ready ===" in line:
                    ready = True
                    break
                if time.time() - start > timeout:
                    log("  WARNING: entrypoint did not become ready "
                        "within {}s".format(timeout))
                    break
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        if ready:
            elapsed = int(time.time() - start)
            log("  Entrypoint ready ({}s)".format(elapsed))
        log("")

    def _assemble_run_args(self):
        # type: () -> List[str]
        """Build the full argument list for ``docker run``.

        See :func:`_container_env_pairs` (module-level) for the
        authoritative host-to-container environment-variable mapping.
        """
        cfg = self.cfg
        render_gid = _get_render_gid()

        args = [
            "-d",
            "--name", cfg.container_name,
            "--privileged",
            "--security-opt", "apparmor=unconfined",
            "--security-opt", "seccomp=unconfined",
            "--restart", "unless-stopped",
            "--group-add", "video",
            "--group-add", render_gid,
            "--cap-add", "SYS_PTRACE",
            "--network", "host",
            "--ipc", "host",
            "--shm-size", cfg.shm_size,
            "--ulimit", "memlock=-1",
        ]

        for name, value in _container_env_pairs(cfg, render_gid):
            args += ["-e", "{}={}".format(name, value)]

        if os.path.exists("/dev/kfd"):
            args += ["--device", "/dev/kfd"]
        if os.path.isdir("/dev/dri"):
            args += ["--device", "/dev/dri"]

        if os.path.isdir("/dev/infiniband"):
            args += ["--device", "/dev/infiniband:/dev/infiniband"]
            for entry in sorted(os.listdir("/dev/infiniband")):
                dev = "/dev/infiniband/{}".format(entry)
                if os.path.exists(dev):
                    args += ["--device", "{}:{}".format(dev, dev)]
            log_verbose(
                "InfiniBand devices: {}".format(
                    " ".join(os.listdir("/dev/infiniband"))
                )
            )
            if cfg.nic_type == "mellanox":
                _bind_mount_rdma_libs(args)
            else:
                log_verbose(
                    "Skipping host RDMA lib bind-mount (nic_type={})"
                    .format(cfg.nic_type)
                )
        else:
            log_verbose("No InfiniBand devices found at /dev/infiniband")

        if os.path.isfile(cfg.hostfile):
            args += ["-v", "{}:{}:ro".format(cfg.hostfile, cfg.hostfile)]

        args += [
            "-v", "{}:/opt/shared".format(cfg.shared_dir),
            "-v", "{}:/opt/builds".format(cfg.builds_dir),
            "-v", "{}:/opt/ssh-keys:ro".format(cfg.ssh.key_dir),
        ]

        if cfg.post_setup_dir:
            args += [
                "-v", "{}:/opt/post-setup:ro".format(cfg.post_setup_dir)
            ]

        for vol in cfg.extra_volumes:
            args += ["-v", vol]

        return args

    def _print_launch_summary(self):
        # type: () -> None
        cfg = self.cfg
        log("")
        log("=== Container '{}' is running ===".format(cfg.container_name))
        log("")
        log(
            "  Shell (root)  : docker exec -it {} bash".format(
                cfg.container_name
            )
        )
        log(
            "  Shell (ubuntu): docker exec -it -u ubuntu {} bash".format(
                cfg.container_name
            )
        )
        log("")
        log("  Verify SSH across nodes:")
        log("    python3 -m mnctl --verify")
        log_verbose("SSH port      : {}".format(cfg.ssh.port))
        log_verbose("Shared directories (same on all nodes):")
        log_verbose("  /opt/shared    <- {}".format(cfg.shared_dir))
        log_verbose("  /opt/builds    <- {}".format(cfg.builds_dir))
        log_verbose("  /opt/ssh-keys  <- {} (read-only)".format(
            cfg.ssh.key_dir
        ))

        self._wait_for_entrypoint()
