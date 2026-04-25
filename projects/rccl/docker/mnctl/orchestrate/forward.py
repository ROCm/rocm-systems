"""Build the per-node CLI argument list for forwarded ``run_mnctl.py`` invocations.

Defaults are suppressed (compared against the matching ``DEFAULT_*``
constant from :mod:`mnctl.config`) so the forwarded command line stays
compact: only user-explicit settings cross the wire.
"""

from typing import List

from ..config import Config, DEFAULT_DOCKERFILE, DEFAULT_NIC_TYPE


def build_forward_args(cfg, action="--run",
                       force_rebuild=None, force_replace=None):
    # type: (Config, str, bool, bool) -> List[str]
    """Build CLI args to forward when invoking per-node setup.

    *force_rebuild* / *force_replace* default to the values on *cfg* but
    callers may pass explicit overrides (e.g. to convert a host-level
    ``--rebuild`` into a per-node ``--replace`` for the ``--run`` phase).
    Passing overrides avoids the need to mutate *cfg* across calls.
    """
    if force_rebuild is None:
        force_rebuild = cfg.force_rebuild
    if force_replace is None:
        force_replace = cfg.force_replace

    args = [
        action,
        "--name", cfg.container_name,
        "--ssh-port", str(cfg.ssh.port),
        "--shm-size", cfg.shm_size,
        "--shared-dir", cfg.shared_dir,
        "--builds-dir", cfg.builds_dir,
        "--ssh-key-dir", cfg.ssh.key_dir,
        "--hostfile", cfg.hostfile,
    ]
    if cfg.gpus_explicit:
        args += ["--gpus", cfg.gpus]
    if cfg.dockerfile != DEFAULT_DOCKERFILE:
        args += ["--dockerfile", cfg.dockerfile]
    if force_rebuild:
        args.append("--rebuild")
    elif force_replace:
        args.append("--replace")
    if cfg.verbose:
        args.append("--verbose")
    for ps_dir in cfg.post_setup_dirs:
        args += ["--post-setup", ps_dir]
    if cfg.no_builtin_nic_setup:
        args.append("--no-builtin-nic-setup")
    if cfg.ssh.key:
        args += ["--ssh", cfg.ssh.key]
    elif cfg.ssh.keygen:
        args.append("--ssh")
    for vol in cfg.extra_volumes:
        args += ["--volume", vol]
    if cfg.nic_type != DEFAULT_NIC_TYPE:
        args += ["--nic-type", cfg.nic_type]
    if cfg.gpu_targets:
        args += ["--gpu-targets", cfg.gpu_targets]
    # Forward shared-fs override so all nodes agree on coordination mode.
    if cfg.shared_fs and cfg.shared_fs != "auto":
        args += ["--shared-fs", cfg.shared_fs]
    # --runtime before positional to avoid nargs='?' ambiguity with --ssh
    args += ["--runtime", cfg.runtime_name]
    args.append(cfg.rocm_image)
    return args
