"""Configuration data structures for mnctl.

Defines the Action enum for mutually exclusive modes, SSHConfig for SSH-related
settings, and Config as the single source of truth for all runtime parameters.
All defaults can be overridden via MNCTL_* environment variables or CLI flags.
"""

import os
from enum import Enum
from typing import List, Optional


class Action(Enum):
    """Mutually exclusive action modes."""
    BUILD = "build"
    RUN = "run"
    VERIFY = "verify"
    LAUNCH_ALL = "launch-all"
    STOP_ALL = "stop-all"
    SETUP_DEPS = "setup-deps"


class SSHConfig(object):
    """SSH-related settings with derived key path resolution."""

    def __init__(
        self,
        key=None,               # type: Optional[str]
        keygen=False,           # type: bool
        key_dir=None,           # type: Optional[str]
        port=2224,              # type: int
    ):
        # type: (...) -> None
        self.key = key
        self.keygen = keygen
        self.key_dir = key_dir or os.path.join(
            os.path.expanduser("~"), ".docker-ssh-keys"
        )
        self.port = port

    @property
    def priv_key(self):
        # type: () -> Optional[str]
        """Resolve private key path (handles *.pub input transparently)."""
        if not self.key:
            return None
        return self.key[:-4] if self.key.endswith(".pub") else self.key

    @property
    def pub_key(self):
        # type: () -> Optional[str]
        """Resolve public key path."""
        if not self.key:
            return None
        return self.key if self.key.endswith(".pub") else self.key + ".pub"


class Config(object):
    """All configuration for mnctl.

    Merges ``MNCTL_*`` environment variables with defaults.  CLI flags
    are applied afterwards in ``__main__.py``.

    Naming convention
    -----------------
    Host-side settings consumed by mnctl are namespaced ``MNCTL_*`` to
    avoid colliding with the user's environment.  When these settings
    are forwarded into the container via ``docker run -e``, the prefix
    is dropped because the container side is private to mnctl and
    consumed by user-maintained shell scripts where short names are
    more ergonomic.

    Example: ``MNCTL_GPUS`` (host) becomes ``GPUS`` (container).

    See :func:`docker_ops._container_env_pairs` for the full mapping.
    """

    def __init__(self):
        # type: () -> None
        home = os.path.expanduser("~")

        self.action = Action.BUILD
        _env_image = os.environ.get("MNCTL_ROCM_IMAGE", "")
        self.rocm_image = _env_image or "rocm/dev-ubuntu-24.04:7.1.1-complete"
        self.container_name = os.environ.get("MNCTL_CONTAINER_NAME", "rccl-mn")
        self.shm_size = os.environ.get("MNCTL_SHM_SIZE", "64g")
        self.gpus = os.environ.get("MNCTL_GPUS", "")
        self.gpus_explicit = False
        self.rocm_image_explicit = bool(_env_image)

        self.shared_dir = os.environ.get(
            "MNCTL_SHARED_DIR", os.path.join(home, ".docker-shared")
        )
        self.builds_dir = os.environ.get(
            "MNCTL_BUILDS_DIR", os.path.join(home, ".docker-builds")
        )
        _env_hostfile = os.environ.get("MNCTL_HOSTFILE", "")
        self.hostfile = _env_hostfile or os.path.join(home, ".mnctl_hostfile")
        # True when the hostfile path came from the user (CLI or env), not
        # the built-in default. Lets SLURM auto-detection regenerate the
        # default file safely without clobbering a user-provided one.
        self.hostfile_explicit = bool(_env_hostfile)
        # Post-setup directories (list, run in order).  Two env vars are
        # accepted:
        #   * MNCTL_POST_SETUP_DIRS  - colon-separated list (canonical form)
        #   * MNCTL_POST_SETUP_DIR   - single dir (back-compat); appended last
        # On the CLI, ``--post-setup`` is repeatable; each occurrence appends.
        _env_dirs = os.environ.get("MNCTL_POST_SETUP_DIRS", "")
        _env_dir = os.environ.get("MNCTL_POST_SETUP_DIR", "")
        self.post_setup_dirs = []  # type: List[str]
        if _env_dirs:
            self.post_setup_dirs.extend(
                d for d in _env_dirs.split(":") if d
            )
        if _env_dir and _env_dir not in self.post_setup_dirs:
            self.post_setup_dirs.append(_env_dir)
        # When True, the NIC-type built-in dir is NOT auto-prepended (see
        # docker_ops._resolve_post_setup_dirs).  User supplies their own.
        self.no_builtin_nic_setup = bool(
            os.environ.get("MNCTL_NO_BUILTIN_NIC_SETUP", "")
        )
        self.host_ssh_port = int(os.environ.get("MNCTL_HOST_SSH_PORT", "22"))
        self.verbose = bool(os.environ.get("MNCTL_VERBOSE", ""))
        self.force_rebuild = False
        self.force_replace = False
        self.dry_run = False
        self.extra_volumes = []  # type: List[str]
        self.dockerfile = os.environ.get(
            "MNCTL_DOCKERFILE", "Dockerfile.Multinode.Ubuntu"
        )

        # Container runtime selection ("docker" or future "pyxis")
        self.runtime_name = "docker"
        self.runtime = None  # type: Optional[object]  # Set by __main__ via get_runtime()

        self.ssh = SSHConfig(
            key=os.environ.get("MNCTL_SSH_KEY", "") or None,
            key_dir=(
                os.environ.get("MNCTL_SSH_KEY_DIR", "")
                or os.path.join(home, ".docker-ssh-keys")
            ),
            port=int(os.environ.get("MNCTL_SSH_PORT", "2224")),
        )

        # NIC type: "mellanox", "ainic", or any custom string
        self.nic_type = os.environ.get("MNCTL_NIC_TYPE", "mellanox")

        # GPU architecture targets (e.g. "gfx942", "gfx950")
        self.gpu_targets = (
            os.environ.get("MNCTL_GPU_TARGETS", "")
            or os.environ.get("GPU_TARGETS", "")
        )

        # Shared-filesystem coordination for --setup-deps.
        #   "auto" (default): detect via `stat -f -c %T` on shared_dir
        #   "yes" / "no":     force-enable or force-disable coordination
        # When enabled, only the first node to claim the lock builds;
        # the others poll the completion marker.  See shared_fs.py.
        self.shared_fs = os.environ.get("MNCTL_SHARED_FS", "auto")

        # Stale-lock TTL: a deps lock older than this is auto-stolen.
        self.deps_lock_ttl_sec = float(
            os.environ.get("MNCTL_DEPS_LOCK_TTL_SEC", "3600")
        )

        # Follower poll timeout: give up waiting after this many seconds.
        self.deps_wait_timeout_sec = float(
            os.environ.get("MNCTL_DEPS_WAIT_TIMEOUT_SEC", "3600")
        )

        # Resolved from the Dockerfile's ARG CONTAINER_USER (set by __main__)
        self.container_user = "ubuntu"

        # Points to the docker/ directory (parent of the mnctl package)
        self.script_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

    @property
    def image_tag(self):
        # type: () -> str
        """Derive Docker image tag from the ROCm base image name."""
        if ":" in self.rocm_image:
            tag = self.rocm_image.rsplit(":", 1)[-1]
        else:
            tag = "latest"
        return "rocm-multinode:" + tag
