"""Configuration data structures for mnctl.

Defines the Action enum for mutually exclusive modes, SSHConfig for SSH-related
settings, and Config as the single source of truth for all runtime parameters.
All defaults can be overridden via MNCTL_* environment variables or CLI flags.
"""

import os
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
# Centralized so build-time, runtime, and forwarding code all share one
# source of truth.  Forwarding logic (orchestrate._build_forward_args)
# uses these to suppress redundant CLI flags whose value already matches
# the default the remote node would compute on its own.
DEFAULT_ROCM_IMAGE      = "rocm/dev-ubuntu-24.04:7.1.1-complete"
DEFAULT_CONTAINER_NAME  = "rccl-mn"
DEFAULT_SHM_SIZE        = "64g"
DEFAULT_DOCKERFILE      = "Dockerfile.Multinode.Ubuntu"
DEFAULT_NIC_TYPE        = "mellanox"
DEFAULT_HOST_SSH_PORT   = 22
DEFAULT_CONTAINER_SSH_PORT = 2224
DEFAULT_SHARED_FS_MODE  = "auto"
DEFAULT_DEPS_LOCK_TTL_SEC = 3600.0
DEFAULT_DEPS_WAIT_TIMEOUT_SEC = 3600.0
DEFAULT_CONTAINER_USER  = "ubuntu"
DEFAULT_SHARED_DIR_NAME = ".docker-shared"
DEFAULT_BUILDS_DIR_NAME = ".docker-builds"
DEFAULT_SSH_KEY_DIR_NAME = ".docker-ssh-keys"
DEFAULT_HOSTFILE_NAME   = ".mnctl_hostfile"


# ---------------------------------------------------------------------------
# Environment-variable readers
# ---------------------------------------------------------------------------
# Each helper applies a uniform empty-string-is-unset rule so callers do not
# repeat ``os.environ.get(name, "") or default`` patterns.  Keep them tiny
# and side-effect-free: they only translate types.
def _env_str(name, default=""):
    # type: (str, str) -> str
    """Return env value or *default* (empty string treated as unset)."""
    return os.environ.get(name, "") or default


def _env_bool(name):
    # type: (str) -> bool
    """Truthy iff env variable is set to a non-empty value."""
    return bool(os.environ.get(name, ""))


def _env_int(name, default):
    # type: (str, int) -> int
    """Parse env value as int, or return *default* if unset/empty."""
    val = os.environ.get(name, "")
    return int(val) if val else default


def _env_float(name, default):
    # type: (str, float) -> float
    """Parse env value as float, or return *default* if unset/empty."""
    val = os.environ.get(name, "")
    return float(val) if val else default


def _env_path(name, default):
    # type: (str, str) -> str
    """Path-typed env reader (currently identical to _env_str; kept for clarity)."""
    return os.environ.get(name, "") or default


def _env_explicit(name):
    # type: (str) -> bool
    """True when the user explicitly set *name* to a non-empty value.

    Used to distinguish "the user told us this" from "we used a built-in
    default".  CLI overlay also sets the matching ``*_explicit`` flag.
    """
    return bool(os.environ.get(name, ""))


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
        port=DEFAULT_CONTAINER_SSH_PORT,  # type: int
    ):
        # type: (...) -> None
        self.key = key
        self.keygen = keygen
        self.key_dir = key_dir or os.path.join(
            os.path.expanduser("~"), DEFAULT_SSH_KEY_DIR_NAME
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

        # --- Action / image / container identity ---
        self.action = Action.BUILD
        self.rocm_image = _env_str("MNCTL_ROCM_IMAGE", DEFAULT_ROCM_IMAGE)
        self.rocm_image_explicit = _env_explicit("MNCTL_ROCM_IMAGE")
        self.container_name = _env_str(
            "MNCTL_CONTAINER_NAME", DEFAULT_CONTAINER_NAME,
        )
        self.shm_size = _env_str("MNCTL_SHM_SIZE", DEFAULT_SHM_SIZE)
        self.gpus = _env_str("MNCTL_GPUS", "")
        self.gpus_explicit = False  # set True by CLI overlay or auto-detect

        # --- Storage layout (host paths) ---
        self.shared_dir = _env_path(
            "MNCTL_SHARED_DIR", os.path.join(home, DEFAULT_SHARED_DIR_NAME),
        )
        self.builds_dir = _env_path(
            "MNCTL_BUILDS_DIR", os.path.join(home, DEFAULT_BUILDS_DIR_NAME),
        )
        self.hostfile = _env_path(
            "MNCTL_HOSTFILE", os.path.join(home, DEFAULT_HOSTFILE_NAME),
        )
        # True when the hostfile path came from the user (CLI or env), not
        # the built-in default. Lets SLURM auto-detection regenerate the
        # default file safely without clobbering a user-provided one.
        self.hostfile_explicit = _env_explicit("MNCTL_HOSTFILE")

        # --- Post-setup directories (list, run in order) ---
        # Two env vars are accepted:
        #   * MNCTL_POST_SETUP_DIRS  - colon-separated list (canonical form)
        #   * MNCTL_POST_SETUP_DIR   - single dir (back-compat); appended last
        # On the CLI, ``--post-setup`` is repeatable; each occurrence appends.
        self.post_setup_dirs = []  # type: List[str]
        env_dirs = _env_str("MNCTL_POST_SETUP_DIRS", "")
        if env_dirs:
            self.post_setup_dirs.extend(d for d in env_dirs.split(":") if d)
        env_dir = _env_str("MNCTL_POST_SETUP_DIR", "")
        if env_dir and env_dir not in self.post_setup_dirs:
            self.post_setup_dirs.append(env_dir)
        # When True, the NIC-type built-in dir is NOT auto-prepended (see
        # docker_ops._resolve_post_setup_dirs).  User supplies their own.
        self.no_builtin_nic_setup = _env_bool("MNCTL_NO_BUILTIN_NIC_SETUP")

        # --- Networking / runtime ---
        self.host_ssh_port = _env_int(
            "MNCTL_HOST_SSH_PORT", DEFAULT_HOST_SSH_PORT,
        )
        self.verbose = _env_bool("MNCTL_VERBOSE")
        self.force_rebuild = False
        self.force_replace = False
        self.dry_run = False
        self.extra_volumes = []  # type: List[str]
        self.dockerfile = _env_str("MNCTL_DOCKERFILE", DEFAULT_DOCKERFILE)

        # Container runtime selection ("docker" or future "pyxis").
        # The runtime *instance* is created and held by __main__ (not on
        # cfg) so that Config remains pure data.
        self.runtime_name = "docker"

        self.ssh = SSHConfig(
            key=_env_str("MNCTL_SSH_KEY", "") or None,
            key_dir=_env_path(
                "MNCTL_SSH_KEY_DIR",
                os.path.join(home, DEFAULT_SSH_KEY_DIR_NAME),
            ),
            port=_env_int("MNCTL_SSH_PORT", DEFAULT_CONTAINER_SSH_PORT),
        )

        # NIC type: "mellanox", "ainic", or any custom string
        self.nic_type = _env_str("MNCTL_NIC_TYPE", DEFAULT_NIC_TYPE)

        # GPU architecture targets (e.g. "gfx942", "gfx950").  Falls back
        # to the legacy unscoped GPU_TARGETS so existing scripts that
        # already export it keep working.
        self.gpu_targets = (
            _env_str("MNCTL_GPU_TARGETS", "")
            or _env_str("GPU_TARGETS", "")
        )

        # Shared-filesystem coordination for --setup-deps.
        #   "auto" (default): detect via `stat -f -c %T` on shared_dir
        #   "yes" / "no":     force-enable or force-disable coordination
        # When enabled, only the first node to claim the lock builds;
        # the others poll the completion marker.  See shared_fs.py.
        self.shared_fs = _env_str("MNCTL_SHARED_FS", DEFAULT_SHARED_FS_MODE)
        self.deps_lock_ttl_sec = _env_float(
            "MNCTL_DEPS_LOCK_TTL_SEC", DEFAULT_DEPS_LOCK_TTL_SEC,
        )
        self.deps_wait_timeout_sec = _env_float(
            "MNCTL_DEPS_WAIT_TIMEOUT_SEC", DEFAULT_DEPS_WAIT_TIMEOUT_SEC,
        )

        # Resolved from the Dockerfile's ARG CONTAINER_USER (set by __main__)
        self.container_user = DEFAULT_CONTAINER_USER

        # Points to the docker/ directory (parent of the mnctl package)
        self.script_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )

        # Lazily filled by docker_ops._resolve_post_setup_dirs() after the
        # final values of ``post_setup_dirs`` / ``no_builtin_nic_setup`` /
        # ``nic_type`` are known.  Reset to None whenever any of those
        # change so the next access recomputes (see invalidate helper).
        self._resolved_post_setup_dirs = None  # type: Optional[List[str]]

    def invalidate_resolved_post_setup_dirs(self):
        # type: () -> None
        """Drop the cached post-setup resolution so it is recomputed."""
        self._resolved_post_setup_dirs = None

    @property
    def image_tag(self):
        # type: () -> str
        """Derive Docker image tag from the ROCm base image name."""
        if ":" in self.rocm_image:
            tag = self.rocm_image.rsplit(":", 1)[-1]
        else:
            tag = "latest"
        return "rocm-multinode:" + tag
