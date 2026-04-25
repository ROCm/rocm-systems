"""Distribute SSH keys, tool code, hostfile, and post-setup to remote nodes.

On a shared filesystem (NFS / GPFS / Lustre / ...) the items are already
visible to every node, so distribution is a no-op per item.  Each item is
probed independently so a mixed setup (e.g. shared $HOME but local /tmp)
still copies only what is needed.  Override the auto-detection with
``--shared-fs {auto,yes,no}``.
"""

import os
import shutil
import sys
from typing import Dict, List, Tuple

from ..config import Config
from ..shared_fs import resolve_shared_fs
from ..utils import (
    host_ssh_cmd, log, log_verbose, run_parallel, ssh_opts,
)


def is_path_shared(path, override):
    # type: (str, str) -> bool
    """Return True if *path* lives on a shared filesystem visible to all nodes.

    ``override`` mirrors :data:`Config.shared_fs` (``"auto"`` / ``"yes"`` /
    ``"no"``).  An explicit ``"yes"`` short-circuits to True without probing
    the filesystem; ``"no"`` short-circuits to False so the caller always
    distributes.  In ``"auto"`` mode the *directory* containing the path is
    inspected (so that not-yet-existing destination files still resolve to
    their parent dir).
    """
    probe = path if os.path.isdir(path) else os.path.dirname(path) or path
    return resolve_shared_fs(probe, override)


def push_pubkey_to_remotes(cfg, remote_hosts, pub_key_path):
    # type: (Config, List[str], str) -> None
    """Append our public key to ``~/.ssh/authorized_keys`` on each remote host.

    Uses ``ssh-copy-id`` when available; falls back to a manual mkdir+append.
    This is the bootstrap step that enables subsequent rsync/SSH operations.
    """
    with open(pub_key_path, "r") as f:
        pub_data = f.read().strip()

    use_copy_id = shutil.which("ssh-copy-id") is not None

    # ssh-copy-id reuses our canonical SSH options but skips BatchMode
    # (it may need to prompt for a password on first contact) and the
    # ConnectTimeout option (handled internally).
    copy_id_opts = ssh_opts(
        cfg.host_ssh_port, batch=False, connect_timeout=None,
    )

    jobs = {}  # type: Dict[str, List[str]]
    for host in remote_hosts:
        if use_copy_id:
            cmd = ["ssh-copy-id", "-i", pub_key_path] + copy_id_opts + [host]
        else:
            remote_cmd = (
                "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
                "grep -qxF '{key}' ~/.ssh/authorized_keys 2>/dev/null "
                "|| echo '{key}' >> ~/.ssh/authorized_keys && "
                "chmod 600 ~/.ssh/authorized_keys"
            ).format(key=pub_data)
            cmd = host_ssh_cmd(cfg, host) + [remote_cmd]
        jobs[host] = cmd
    results = run_parallel(jobs)

    ok_count = 0
    for host in remote_hosts:
        r = results[host]
        if r.ok:
            ok_count += 1
            log_verbose("  Public key installed on {}".format(host))
        else:
            log_verbose(
                "  Key push to {} returned {}: {}".format(
                    host, r.returncode, r.stderr_text.strip()[:200]
                )
            )

    if ok_count:
        log("  SSH key bootstrapped on {}/{} remote node(s)".format(
            ok_count, len(remote_hosts),
        ))


def distribute_files(cfg, remote_hosts):
    # type: (Config, List[str]) -> None
    """Distribute SSH keys, tool code, hostfile, and post-setup to remotes.

    Uses rsync over the host SSH port.  All items are small so this
    completes quickly even for many nodes.  Items already on a shared
    filesystem are skipped automatically.
    """
    if not remote_hosts:
        return

    log("=== Distributing files to {} remote node(s) ===".format(
        len(remote_hosts),
    ))

    host_key = os.path.join(cfg.ssh.key_dir, "id_rsa")
    pub_key = host_key + ".pub"

    # Bootstrap: install our public key into each remote host's
    # ~/.ssh/authorized_keys so subsequent SSH/rsync operations work.
    # Skipped when ~/.ssh is itself on a shared FS (key already visible).
    home_ssh = os.path.join(os.path.expanduser("~"), ".ssh")
    if not os.path.isfile(pub_key):
        log_verbose("No public key at {}; skipping key bootstrap".format(pub_key))
    elif is_path_shared(home_ssh, cfg.shared_fs):
        log("  Skipping SSH key bootstrap ({} is on a shared filesystem)"
            .format(home_ssh))
    else:
        push_pubkey_to_remotes(cfg, remote_hosts, pub_key)

    # (local_path, is_dir, label)
    items = [
        (cfg.ssh.key_dir, True, "SSH keys"),
        (cfg.script_dir, True, "tool code"),
    ]
    if cfg.hostfile and os.path.isfile(cfg.hostfile):
        items.append((cfg.hostfile, False, "hostfile"))
    for i, ps_dir in enumerate(cfg.post_setup_dirs):
        if ps_dir and os.path.isdir(ps_dir):
            label = (
                "post-setup[{}]".format(i)
                if len(cfg.post_setup_dirs) > 1 else "post-setup"
            )
            items.append((ps_dir, True, label))

    # Filter out items already on a shared filesystem (visible to all nodes).
    items_to_copy = []
    for local_path, is_dir, label in items:
        if is_path_shared(local_path, cfg.shared_fs):
            log("  Skipping {} ({} is on a shared filesystem)".format(
                label, local_path,
            ))
        else:
            items_to_copy.append((local_path, is_dir, label))

    if not items_to_copy:
        log("  All paths are on a shared filesystem; no rsync needed")
        log("")
        return

    if shutil.which("rsync") is None:
        log("")
        log("ERROR: rsync is required for multi-node deployment")
        log("  Install it:  apt-get install rsync  /  yum install rsync")
        log("  Or place the items below on a shared filesystem:")
        for _p, _d, label in items_to_copy:
            log("    - {}".format(label))
        sys.exit(1)

    # rsync --rsh expects a single string; build it from our canonical
    # SSH options. ConnectTimeout is omitted to match the original behavior.
    rsh_parts = ["ssh"] + ssh_opts(
        cfg.host_ssh_port,
        identity=host_key if os.path.isfile(host_key) else None,
        connect_timeout=None,
    )
    rsh = " ".join(rsh_parts)

    items = items_to_copy

    # Phase 1: create parent directories on remote hosts (parallel)
    remote_dirs = set()  # type: set
    for local_path, is_dir, _label in items:
        if is_dir:
            remote_dirs.add(local_path)
        else:
            remote_dirs.add(os.path.dirname(local_path))

    mkdir_jobs = {
        host: host_ssh_cmd(cfg, host) + ["mkdir", "-p"] + sorted(remote_dirs)
        for host in remote_hosts
    }
    for host, r in run_parallel(mkdir_jobs).items():
        if not r.ok and cfg.verbose:
            log_verbose(
                "mkdir on {}: {}".format(host, r.stderr_text.strip()[:200])
            )

    # Phase 2: rsync each item to each remote host (parallel)
    rsync_jobs = {}  # type: Dict[Tuple[str, str], List[str]]
    for host in remote_hosts:
        for local_path, is_dir, label in items:
            if is_dir:
                src = local_path.rstrip("/") + "/"
                dest = "{}:{}".format(host, local_path.rstrip("/") + "/")
            else:
                src = local_path
                dest = "{}:{}".format(host, local_path)
            rsync_jobs[(host, label)] = [
                "rsync", "-a", "--exclude", "__pycache__",
                "--rsh", rsh,
                src, dest,
            ]
    rsync_results = run_parallel(rsync_jobs)

    # Collect results
    failed = []  # type: List[Tuple[str, str]]
    for key in sorted(rsync_results.keys()):
        r = rsync_results[key]
        host, label = key
        if r.ok:
            log_verbose("  [OK] {} -> {}".format(label, host))
        else:
            log("  [FAIL] {} -> {}: {}".format(
                label, host, r.stderr_text.strip()[:200],
            ))
            failed.append(key)

    if failed:
        log("")
        log("ERROR: file distribution failed for {} item(s)".format(
            len(failed),
        ))
        for host, label in failed:
            log("  - {} on {}".format(label, host))
        log("")
        log("  Ensure remote hosts are reachable via SSH (port {})".format(
            cfg.host_ssh_port,
        ))
        sys.exit(1)

    log("  Distributed to {} node(s)".format(len(remote_hosts)))
    log("")
