"""Host-local setup: shared dirs + SSH keys.  Idempotent."""

import os

from ..config import Config
from ..ssh import install_ssh_keys
from ..utils import ensure_dir, log, log_verbose, Timer


def setup_host(cfg):
    # type: (Config) -> None
    """Create shared directories and install SSH keys."""
    with Timer("Host setup"):
        log("=== Host setup ===")

        for d in (cfg.shared_dir, cfg.builds_dir):
            existed = os.path.isdir(d)
            # Prefer 0o777 so other users on the same host (different UIDs
            # in the container) can write; fall back to 0o755 on filesystems
            # that reject world-writable bits (e.g. some shared FS mounts).
            ensure_dir(d, modes=(0o777, 0o755))
            log_verbose(("Exists  " if existed else "Created ") + d)

        key_dir = cfg.ssh.key_dir
        existed = os.path.isdir(key_dir)
        ensure_dir(key_dir, modes=(0o700,))
        log_verbose(
            ("Exists  " if existed else "Created ")
            + "{}{}".format(key_dir, "" if existed else " (mode 700)")
        )

        install_ssh_keys(cfg)

        if cfg.verbose:
            log_verbose("SSH key dir contents:")
            if os.path.isdir(key_dir):
                for entry in sorted(os.listdir(key_dir)):
                    log_verbose("  {}".format(entry))
            hf = cfg.hostfile
            if os.path.isfile(hf):
                with open(hf) as f:
                    n = sum(1 for _ in f)
                log_verbose(
                    "Hostfile: {} (exists, {} lines)".format(hf, n)
                )
            else:
                log_verbose("Hostfile: {} (not found)".format(hf))

    log("")
