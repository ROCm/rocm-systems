"""CLI entry point for mnctl (Multi-Node Control).

Invoke as:
    python3 -m mnctl [OPTIONS] [ROCM_IMAGE]
    python3 run_mnctl.py [OPTIONS] [ROCM_IMAGE]
"""

import argparse
import os
import re
import subprocess
import sys

from .config import Action, Config
from .utils import error, log, log_verbose, set_verbose, expand_path, auto_detect_gpus
from .validate import validate
from .slurm import detect_slurm
from .runtime import get_runtime
from .ssh import verify_ssh
from .deps import setup_shared_deps
from .orchestrate import setup_host, launch_all, stop_all
from .preflight import run_preflight


def build_parser():
    # type: () -> argparse.ArgumentParser
    """Construct the argument parser with all flags."""
    parser = argparse.ArgumentParser(
        prog="mnctl",
        description="Build and launch multi-node ROCm Docker containers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python3 -m mnctl                                                    # build image only
  python3 -m mnctl rocm/dev-ubuntu-24.04:7.1.1-complete               # specific ROCm
  python3 -m mnctl --run                                              # build + launch
  python3 -m mnctl --run --volume /data:/data                         # extra mount
  python3 -m mnctl --launch-all --ssh                                 # multi-node, auto SSH
  python3 -m mnctl --launch-all --ssh ~/.ssh/id_rsa                   # multi-node, your keys
  python3 -m mnctl --launch-all --dockerfile Dockerfile.Multinode.ALinux3
  python3 -m mnctl --launch-all --dry-run                             # pre-flight checks
  python3 -m mnctl --verify                                           # check SSH connectivity
  python3 -m mnctl --stop-all                                         # stop everywhere

Environment Variables (override defaults without flags):
  MNCTL_ROCM_IMAGE, MNCTL_CONTAINER_NAME, MNCTL_GPUS,
  MNCTL_SSH_PORT, MNCTL_SHM_SIZE, MNCTL_SHARED_DIR,
  MNCTL_BUILDS_DIR, MNCTL_SSH_KEY_DIR, MNCTL_SSH_KEY,
  MNCTL_HOSTFILE, MNCTL_HOST_SSH_PORT, MNCTL_POST_SETUP_DIR,
  MNCTL_DOCKERFILE, MNCTL_NIC_TYPE, MNCTL_GPU_TARGETS, MNCTL_VERBOSE,
  MNCTL_SHARED_FS, MNCTL_DEPS_LOCK_TTL_SEC, MNCTL_DEPS_WAIT_TIMEOUT_SEC

  GPU_TARGETS is also accepted as a fallback for MNCTL_GPU_TARGETS.

Shared-FS coordination for --setup-deps:
  When MNCTL_SHARED_DIR is on a network filesystem (NFS, GPFS, Lustre,
  CephFS, GlusterFS, ...), only one node ("leader") performs the
  UCX/OpenMPI build; the other nodes ("followers") wait for a
  completion marker.  On a local filesystem every node builds
  independently -- there is always a working install regardless of
  storage type.  Override detection with --shared-fs {auto,yes,no} or
  MNCTL_SHARED_FS.  See mnctl/shared_fs.py for the lock primitives.

Host vs container env-var naming:
  Host-side mnctl settings are namespaced MNCTL_* to avoid collisions
  with the user's environment.  When forwarded into the container
  (via docker run -e), the prefix is dropped because the container
  side is consumed by mnctl's own scripts (entrypoint, post-setup):

      MNCTL_GPUS         -> GPUS
      MNCTL_VERBOSE      -> VERBOSE
      MNCTL_NIC_TYPE     -> NIC_TYPE
      MNCTL_GPU_TARGETS  -> GPU_TARGETS

  See docker_ops._container_env_pairs() for the authoritative mapping.

Path expansion:
  All path options support ~ and $VAR / ${VAR} expansion.
""",
    )

    parser.add_argument(
        "rocm_image", nargs="?", default=None,
        help=(
            "ROCm base image "
            "(default: MNCTL_ROCM_IMAGE env or rocm/dev-ubuntu-24.04:7.1.1-complete)"
        ),
    )

    # --- Action modes ---
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--run", dest="action", action="store_const", const=Action.RUN,
        help="Build image and launch a container (skips build if image exists)",
    )
    action.add_argument(
        "--verify", dest="action", action="store_const",
        const=Action.VERIFY,
        help="Verify SSH connectivity to all hosts in the hostfile",
    )
    action.add_argument(
        "--launch-all", dest="action", action="store_const",
        const=Action.LAUNCH_ALL,
        help="Build + launch on ALL nodes in the hostfile via SSH",
    )
    action.add_argument(
        "--stop-all", dest="action", action="store_const",
        const=Action.STOP_ALL,
        help="Stop + remove containers on ALL nodes in the hostfile",
    )
    action.add_argument(
        "--setup-deps", dest="action", action="store_const",
        const=Action.SETUP_DEPS,
        help=argparse.SUPPRESS,
    )

    # --- Visible options ---
    parser.add_argument(
        "--hostfile",
        help="MPI hostfile (default: MNCTL_HOSTFILE env or ~/.mnctl_hostfile)",
    )
    parser.add_argument(
        "--ssh", nargs="?", const="auto", default=None,
        help=(
            "Enable inter-container SSH: "
            "--ssh to auto-generate keys, --ssh KEY_PATH to use existing"
        ),
    )
    parser.add_argument(
        "--volume", "-v", dest="volumes", action="append", default=[],
        help="Extra host volume mount SRC:DST (repeatable)",
    )
    parser.add_argument(
        "--post-setup", dest="post_setup_dir",
        help="Post-setup dir with setup.sh/env.sh (optional)",
    )
    parser.add_argument(
        "--dockerfile",
        help=(
            "Dockerfile to use for image build "
            "(default: MNCTL_DOCKERFILE env or Dockerfile.Multinode.Ubuntu)"
        ),
    )
    parser.add_argument(
        "--nic-type", dest="nic_type",
        help=(
            "NIC type: mellanox, ainic, or custom "
            "(default: MNCTL_NIC_TYPE env or mellanox). "
            "Controls RDMA library bind-mounting and "
            "NIC-specific post-setup steps."
        ),
    )
    parser.add_argument(
        "--gpu-targets", dest="gpu_targets",
        help=(
            "GPU architecture targets, e.g. gfx942 or gfx950 "
            "(default: MNCTL_GPU_TARGETS or GPU_TARGETS env). "
            "Passed as --build-arg to the Dockerfile and as an "
            "env var to containers for rccl-tests builds."
        ),
    )
    parser.add_argument(
        "--runtime", dest="runtime_name",
        choices=["docker"],
        help="Container runtime (default: docker)",
    )
    parser.add_argument(
        "--shared-fs", dest="shared_fs",
        choices=["auto", "yes", "no"],
        help=(
            "Shared-filesystem mode (default: MNCTL_SHARED_FS env or 'auto'). "
            "Affects two things: (1) --setup-deps coordination -- only the "
            "leader node builds on shared storage, followers wait for a "
            "completion marker; (2) --launch-all file distribution -- items "
            "already on a shared FS skip rsync. "
            "'yes' forces shared behavior (skip rsync, leader-elect builds); "
            "'no' forces per-node behavior (always rsync, always rebuild)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and check prerequisites without executing",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Force image rebuild and replace existing containers",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Replace existing containers without rebuilding the image",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable detailed debug logging",
    )

    # --- Hidden flags (env-var overridable, used internally for forwarding) ---
    parser.add_argument("--name", dest="container_name",
                        help=argparse.SUPPRESS)
    parser.add_argument("--gpus", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--ssh-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shm-size", help=argparse.SUPPRESS)
    parser.add_argument("--shared-dir", help=argparse.SUPPRESS)
    parser.add_argument("--builds-dir", help=argparse.SUPPRESS)
    parser.add_argument("--ssh-key-dir", help=argparse.SUPPRESS)
    parser.add_argument("--host-ssh-port", type=int, help=argparse.SUPPRESS)

    return parser


def _apply_cli_args(cfg, args):
    # type: (Config, argparse.Namespace) -> None
    """Overlay CLI arguments onto the Config (only if explicitly provided)."""
    if args.action is not None:
        cfg.action = args.action
    if args.rocm_image is not None:
        cfg.rocm_image = args.rocm_image
        cfg.rocm_image_explicit = True

    # --ssh: "auto" means keygen, any other value is a key path
    if args.ssh is not None:
        if args.ssh == "auto":
            cfg.ssh.keygen = True
        else:
            cfg.ssh.key = args.ssh

    if args.hostfile is not None:
        cfg.hostfile = args.hostfile
        cfg.hostfile_explicit = True
    if args.volumes:
        cfg.extra_volumes = list(args.volumes)
    if args.post_setup_dir is not None:
        cfg.post_setup_dir = args.post_setup_dir
    if args.dockerfile is not None:
        cfg.dockerfile = args.dockerfile
    if args.nic_type is not None:
        cfg.nic_type = args.nic_type
    if args.gpu_targets is not None:
        cfg.gpu_targets = args.gpu_targets
    if args.runtime_name is not None:
        cfg.runtime_name = args.runtime_name
    if args.shared_fs is not None:
        cfg.shared_fs = args.shared_fs
    if args.dry_run:
        cfg.dry_run = True
    if args.rebuild:
        cfg.force_rebuild = True
        cfg.force_replace = True
    if args.replace:
        cfg.force_replace = True
    if args.verbose:
        cfg.verbose = True

    # Hidden flags (used internally for forwarding and env-var overrides)
    if args.container_name is not None:
        cfg.container_name = args.container_name
    if args.gpus is not None:
        cfg.gpus = str(args.gpus)
        cfg.gpus_explicit = True
    if args.ssh_port is not None:
        cfg.ssh.port = args.ssh_port
    if args.shm_size is not None:
        cfg.shm_size = args.shm_size
    if args.shared_dir is not None:
        cfg.shared_dir = args.shared_dir
    if args.builds_dir is not None:
        cfg.builds_dir = args.builds_dir
    if args.ssh_key_dir is not None:
        cfg.ssh.key_dir = args.ssh_key_dir
    if args.host_ssh_port is not None:
        cfg.host_ssh_port = args.host_ssh_port


def _resolve_container_user(cfg):
    # type: (Config) -> str
    """Parse ``ARG CONTAINER_USER=...`` from the Dockerfile."""
    dockerfile_path = os.path.join(cfg.script_dir, cfg.dockerfile)
    try:
        with open(dockerfile_path) as f:
            for line in f:
                m = re.match(r"^ARG\s+CONTAINER_USER=(.+)$", line.strip())
                if m:
                    return m.group(1).strip().strip('"').strip("'")
    except (IOError, OSError):
        pass
    return "ubuntu"


def _expand_paths(cfg):
    # type: (Config) -> None
    """Expand ~ and $VAR in all path-valued config fields."""
    cfg.shared_dir = expand_path(cfg.shared_dir)
    cfg.builds_dir = expand_path(cfg.builds_dir)
    cfg.ssh.key_dir = expand_path(cfg.ssh.key_dir)
    cfg.hostfile = expand_path(cfg.hostfile)
    if cfg.post_setup_dir:
        cfg.post_setup_dir = expand_path(cfg.post_setup_dir)
    if cfg.ssh.key:
        cfg.ssh.key = expand_path(cfg.ssh.key)
    cfg.extra_volumes = [expand_path(v) for v in cfg.extra_volumes]


def _dump_config(cfg):
    # type: (Config) -> None
    """Print all configuration values in verbose mode."""
    log("=== Verbose mode enabled ===")
    log_verbose("script_dir={}".format(cfg.script_dir))
    log_verbose("rocm_image={}".format(cfg.rocm_image))
    log_verbose("image_tag={}".format(cfg.image_tag))
    log_verbose("container_name={}".format(cfg.container_name))
    log_verbose("gpus={}".format(cfg.gpus))
    log_verbose("ssh.port={}".format(cfg.ssh.port))
    log_verbose("shm_size={}".format(cfg.shm_size))
    log_verbose("shared_dir={}".format(cfg.shared_dir))
    log_verbose("builds_dir={}".format(cfg.builds_dir))
    log_verbose("ssh.key_dir={}".format(cfg.ssh.key_dir))
    log_verbose("hostfile={}".format(cfg.hostfile))
    log_verbose("post_setup_dir={}".format(cfg.post_setup_dir))
    log_verbose("ssh.key={}".format(cfg.ssh.key or ""))
    log_verbose("ssh.keygen={}".format(cfg.ssh.keygen))
    log_verbose("action={}".format(cfg.action.value))
    log_verbose("host_ssh_port={}".format(cfg.host_ssh_port))
    log_verbose("dockerfile={}".format(cfg.dockerfile))
    log_verbose("force_rebuild={}".format(cfg.force_rebuild))
    log_verbose("force_replace={}".format(cfg.force_replace))
    log_verbose("nic_type={}".format(cfg.nic_type))
    log_verbose("gpu_targets={}".format(cfg.gpu_targets or "(dockerfile default)"))
    log_verbose("runtime={}".format(cfg.runtime_name))
    log_verbose("shared_fs={}".format(cfg.shared_fs))
    log_verbose("extra_volumes={}".format(cfg.extra_volumes))
    import platform
    log_verbose("Host kernel: {}".format(platform.release()))
    log("")


def _run():
    # type: () -> None
    """Core logic: parse args, configure, validate, dispatch."""
    parser = build_parser()
    args = parser.parse_args()

    # --- Build configuration ---
    cfg = Config()
    _apply_cli_args(cfg, args)
    _expand_paths(cfg)
    cfg.container_user = _resolve_container_user(cfg)

    # Auto-detect GPUs if not set
    if not cfg.gpus:
        cfg.gpus = str(auto_detect_gpus())

    # Enable verbose logging
    set_verbose(cfg.verbose)

    # Instantiate the container runtime
    cfg.runtime = get_runtime(cfg)

    # SLURM auto-detection (may mutate hostfile and ssh settings)
    detect_slurm(cfg)

    # Validate configuration
    validate(cfg)

    # Verbose config dump
    if cfg.verbose:
        _dump_config(cfg)

    # Dry-run: pre-flight checks only, then exit
    if cfg.dry_run:
        run_preflight(cfg)
        return

    # --- Dispatch ---
    if cfg.action == Action.VERIFY:
        verify_ssh(cfg)

    elif cfg.action == Action.STOP_ALL:
        stop_all(cfg)

    elif cfg.action == Action.SETUP_DEPS:
        setup_host(cfg)
        cfg.runtime.build_image()
        setup_shared_deps(cfg)

    elif cfg.action == Action.LAUNCH_ALL:
        setup_host(cfg)
        cfg.runtime.build_image()
        setup_shared_deps(cfg)
        launch_all(cfg)

    elif cfg.action == Action.RUN:
        setup_host(cfg)
        cfg.runtime.build_image()
        cfg.runtime.launch()

    else:
        # Default: build only
        setup_host(cfg)
        cfg.runtime.build_image()
        log("Image ready: {}".format(cfg.image_tag))
        log("")
        log("To launch a container:")
        log("  python3 -m mnctl --run")
        log("")


def main():
    # type: () -> None
    """Entry point with top-level exception handling for partial failures."""
    try:
        _run()
    except KeyboardInterrupt:
        log("")
        log("Interrupted.")
        log("  Partial setup may exist. To clean up:")
        log("    python3 -m mnctl --stop-all")
        sys.exit(130)
    except SystemExit:
        raise
    except subprocess.CalledProcessError as e:
        log("")
        error("Command failed (exit {}): {}".format(
            e.returncode,
            " ".join(str(a) for a in e.cmd) if e.cmd else "(unknown)",
        ))
        log("  Partial setup may exist. To clean up:")
        log("    python3 -m mnctl --stop-all")
        sys.exit(1)
    except Exception as e:
        log("")
        error("Unexpected error: {}".format(e))
        log("  Partial setup may exist. To clean up:")
        log("    python3 -m mnctl --stop-all")
        sys.exit(1)


if __name__ == "__main__":
    main()
