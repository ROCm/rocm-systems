"""SLURM auto-detection: hostfile generation and SSH key configuration.

When running inside a SLURM allocation (SLURM_NODELIST is set) and no
hostfile exists yet, this module expands the node list via ``scontrol``,
writes a hostfile, and auto-configures SSH keys from ``~/.ssh/``.
"""

import os
import shutil
import subprocess

from .config import Config
from .utils import log, log_verbose, warn


def detect_slurm(cfg):
    # type: (Config) -> None
    """Auto-detect SLURM allocation; mutates *cfg* in-place.

    Behavior:
      * User-provided hostfile (``--hostfile`` / ``MNCTL_HOSTFILE``) is
        always honored as-is, even when SLURM env vars are set.
      * Otherwise, if ``SLURM_NODELIST`` is set, regenerate the default
        hostfile from the live allocation (overwriting any stale file
        from a previous run/allocation).
      * If no SLURM vars and no existing hostfile, do nothing -- the
        validator will surface a clear error.
    """
    if cfg.hostfile_explicit:
        # User passed --hostfile explicitly; trust it verbatim.
        log_verbose(
            "User-provided hostfile {}; skipping SLURM auto-detect".format(
                cfg.hostfile
            )
        )
        return

    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get(
        "SLURM_JOB_NODELIST", ""
    )
    if not nodelist:
        # No SLURM allocation; fall back to whatever hostfile exists.
        return

    if os.path.isfile(cfg.hostfile):
        log_verbose(
            "SLURM_NODELIST is set; regenerating default hostfile {}".format(
                cfg.hostfile
            )
        )

    if not shutil.which("scontrol"):
        warn(
            "SLURM allocation detected (SLURM_NODELIST={})".format(nodelist)
        )
        warn("  but 'scontrol' not found in PATH; cannot expand node list")
        warn("  Install slurm-client or create a hostfile manually")
        return

    # --- Determine slots per node ---
    slots = int(cfg.gpus) if cfg.gpus else 0

    if slots == 0:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE", "")
        if slurm_gpus:
            part = slurm_gpus.rsplit(":", 1)[-1]
            part = part.split("(")[0]
            if part.isdigit():
                slots = int(part)

    if slots == 0:
        ntasks = os.environ.get("SLURM_NTASKS_PER_NODE", "")
        if ntasks and ntasks.isdigit():
            slots = int(ntasks)

    if slots == 0:
        slots = 1
        warn(
            "Could not determine GPU/slot count from SLURM; "
            "defaulting to slots=1"
        )

    log("=== SLURM allocation detected ===")
    log("  SLURM_NODELIST : {}".format(nodelist))
    log("  SLURM_NNODES   : {}".format(
        os.environ.get("SLURM_NNODES", "unknown")
    ))
    log("  Slots per node : {}".format(slots))
    log_verbose(
        "SLURM_JOB_ID={} SLURM_GPUS_PER_NODE={} "
        "SLURM_NTASKS_PER_NODE={}".format(
            os.environ.get("SLURM_JOB_ID", ""),
            os.environ.get("SLURM_GPUS_PER_NODE", ""),
            os.environ.get("SLURM_NTASKS_PER_NODE", ""),
        )
    )

    # --- Expand nodelist via scontrol ---
    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        warn("scontrol failed to expand node list")
        return

    hosts = result.stdout.decode("utf-8", errors="replace").strip().splitlines()

    # --- Write hostfile ---
    hostfile_dir = os.path.dirname(cfg.hostfile)
    if hostfile_dir:
        os.makedirs(hostfile_dir, exist_ok=True)

    with open(cfg.hostfile, "w") as f:
        for host in hosts:
            host = host.strip()
            if host:
                f.write("{} slots={}\n".format(host, slots))

    log("  Generated hostfile: {}".format(cfg.hostfile))
    if cfg.verbose:
        log_verbose("Hostfile contents:")
        with open(cfg.hostfile) as f:
            for line in f:
                log_verbose("  {}".format(line.rstrip()))

    # --- Auto-configure SSH keys from ~/.ssh ---
    _auto_configure_ssh(cfg)

    log("")


def _auto_configure_ssh(cfg):
    # type: (Config) -> None
    """Pick up the user's existing SSH key when SLURM is detected."""
    if cfg.ssh.key or cfg.ssh.keygen:
        log_verbose("SSH key explicitly provided; skipping SLURM SSH auto-detect")
        return

    home = os.path.expanduser("~")

    for key_name in ("id_rsa", "id_ed25519"):
        key_path = os.path.join(home, ".ssh", key_name)
        if os.path.isfile(key_path):
            cfg.ssh.key = key_path
            log("  SSH key        : {} (auto-detected)".format(key_path))
            return

    cfg.ssh.keygen = True
    log("  SSH key        : (none found at ~/.ssh/id_rsa; will auto-generate)")
