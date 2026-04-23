"""Multi-node orchestration: host setup, launch-all, stop-all.

The host-setup phase creates shared directories and installs SSH keys.
launch_all / stop_all fan out to every node in the hostfile by spawning
all SSH commands as concurrent subprocesses (Popen).  Reader threads
stream output line-by-line from each node in real time.
"""

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from typing import Dict, List, Tuple

from .config import Config
from .ssh import install_ssh_keys
from .utils import (
    log, log_verbose, parse_hostfile, get_local_hostnames, Timer,
    ssh_opts, host_ssh_cmd, run_parallel,
)


# ---------------------------------------------------------------------------
# Host setup (shared dirs + SSH keys) — idempotent
# ---------------------------------------------------------------------------
def setup_host(cfg):
    # type: (Config) -> None
    """Create shared directories and install SSH keys."""
    with Timer("Host setup"):
        log("=== Host setup ===")

        for d in (cfg.shared_dir, cfg.builds_dir):
            if not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
                try:
                    os.chmod(d, 0o777)
                except OSError:
                    try:
                        os.chmod(d, 0o755)
                    except OSError:
                        pass
                log_verbose("Created {}".format(d))
            else:
                log_verbose("Exists  {}".format(d))

        key_dir = cfg.ssh.key_dir
        if not os.path.isdir(key_dir):
            os.makedirs(key_dir, exist_ok=True)
            os.chmod(key_dir, 0o700)
            log_verbose("Created {} (mode 700)".format(key_dir))
        else:
            log_verbose("Exists  {}".format(key_dir))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ssh_base_cmd(cfg, host):
    # type: (Config, str) -> List[str]
    """Build the SSH prefix for reaching *host* (host-level SSH).

    Thin wrapper around :func:`utils.host_ssh_cmd` that returns the
    command list without a remote command appended.
    """
    return host_ssh_cmd(cfg, host)


# ---------------------------------------------------------------------------
# SSH key bootstrap for remote hosts
# ---------------------------------------------------------------------------
def _push_pubkey_to_remotes(cfg, remote_hosts, pub_key_path):
    # type: (Config, List[str], str) -> None
    """Append our public key to ~/.ssh/authorized_keys on each remote host.

    Uses ssh-copy-id when available; falls back to a manual mkdir+append.
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
            cmd = _ssh_base_cmd(cfg, host) + [remote_cmd]
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
                    host, r.returncode, r.stderr_text().strip()[:200]
                )
            )

    if ok_count:
        log("  SSH key bootstrapped on {}/{} remote node(s)".format(
            ok_count, len(remote_hosts),
        ))


# ---------------------------------------------------------------------------
# File distribution to remote nodes (no shared FS required)
# ---------------------------------------------------------------------------
def _distribute_files(cfg, remote_hosts):
    # type: (Config, List[str]) -> None
    """Distribute SSH keys, tool code, hostfile, and post-setup to remotes.

    Uses rsync over the host SSH port.  All items are small so this
    completes quickly even for many nodes.
    """
    if not remote_hosts:
        return

    if shutil.which("rsync") is None:
        log("")
        log("ERROR: rsync is required for multi-node deployment")
        log("  Install it:  apt-get install rsync  /  yum install rsync")
        sys.exit(1)

    log("=== Distributing files to {} remote node(s) ===".format(
        len(remote_hosts),
    ))

    host_key = os.path.join(cfg.ssh.key_dir, "id_rsa")
    pub_key = host_key + ".pub"

    # Bootstrap: install our public key into each remote host's
    # ~/.ssh/authorized_keys so subsequent SSH/rsync operations work.
    if os.path.isfile(pub_key):
        _push_pubkey_to_remotes(cfg, remote_hosts, pub_key)

    # rsync --rsh expects a single string; build it from our canonical
    # SSH options. ConnectTimeout is omitted to match the original behavior.
    rsh_parts = ["ssh"] + ssh_opts(
        cfg.host_ssh_port,
        identity=host_key if os.path.isfile(host_key) else None,
        connect_timeout=None,
    )
    rsh = " ".join(rsh_parts)

    # (local_path, is_dir, label)
    items = [
        (cfg.ssh.key_dir, True, "SSH keys"),
        (cfg.script_dir, True, "tool code"),
    ]
    if cfg.hostfile and os.path.isfile(cfg.hostfile):
        items.append((cfg.hostfile, False, "hostfile"))
    if cfg.post_setup_dir and os.path.isdir(cfg.post_setup_dir):
        items.append((cfg.post_setup_dir, True, "post-setup"))

    # Phase 1: create parent directories on remote hosts (parallel)
    remote_dirs = set()  # type: set
    for local_path, is_dir, _label in items:
        if is_dir:
            remote_dirs.add(local_path)
        else:
            remote_dirs.add(os.path.dirname(local_path))

    mkdir_jobs = {
        host: _ssh_base_cmd(cfg, host) + ["mkdir", "-p"] + sorted(remote_dirs)
        for host in remote_hosts
    }
    for host, r in run_parallel(mkdir_jobs).items():
        if not r.ok and cfg.verbose:
            log_verbose(
                "mkdir on {}: {}".format(host, r.stderr_text().strip()[:200])
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
                label, host, r.stderr_text().strip()[:200],
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


# ---------------------------------------------------------------------------
# Forward-argument assembly for remote invocations
# ---------------------------------------------------------------------------
def _build_forward_args(cfg, action="--run",
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
    if cfg.dockerfile != "Dockerfile.Multinode.Ubuntu":
        args += ["--dockerfile", cfg.dockerfile]
    if force_rebuild:
        args.append("--rebuild")
    elif force_replace:
        args.append("--replace")
    if cfg.verbose:
        args.append("--verbose")
    if cfg.post_setup_dir:
        args += ["--post-setup", cfg.post_setup_dir]
    if cfg.ssh.key:
        args += ["--ssh", cfg.ssh.key]
    elif cfg.ssh.keygen:
        args.append("--ssh")
    for vol in cfg.extra_volumes:
        args += ["--volume", vol]
    if cfg.nic_type != "mellanox":
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


# ---------------------------------------------------------------------------
# Launch containers on all nodes (parallel via Popen, streamed output)
# ---------------------------------------------------------------------------
def _make_host_label(host, max_len):
    # type: (str, int) -> str
    """Right-pad *host* so streaming prefixes align across nodes."""
    return host.ljust(max_len)


def launch_all(cfg):
    # type: (Config) -> None
    """Build + launch a container on every node in the hostfile.

    1. Distribute SSH keys, tool code, hostfile, and post-setup to remotes
    2. Spawn deps-build + container-launch on every node concurrently
    3. Stream output line-by-line from each node as it arrives
    """
    with Timer("Launch all nodes"):
        log("=== Launching containers on all nodes ===")
        log("")

        hosts = parse_hostfile(cfg.hostfile)
        local_names = get_local_hostnames()
        script = os.path.join(cfg.script_dir, "run_mnctl.py")

        log("  Hostfile  : {} ({} nodes)".format(cfg.hostfile, len(hosts)))
        log("  Image     : {}".format(cfg.image_tag))
        log("  Container : {}".format(cfg.container_name))
        log_verbose("Script    : {}".format(script))
        log("")

        # Distribute files to remote hosts (handles non-shared FS)
        remote_hosts = [h for h in hosts if h not in local_names]
        _distribute_files(cfg, remote_hosts)

        # Build compound command: setup-deps (idempotent) then run.
        # python3 -u disables output buffering so lines stream in real time.
        # --setup-deps gets --rebuild (image build happens here).
        # --run reuses the image that --setup-deps just built, so we
        # downgrade any --rebuild request to a container --replace to
        # avoid a redundant full image rebuild on every node.
        deps_args = _build_forward_args(cfg, action="--setup-deps")
        run_args = _build_forward_args(
            cfg, action="--run",
            force_rebuild=False,
            force_replace=cfg.force_rebuild or cfg.force_replace,
        )

        def _quote_cmd(args):
            # type: (List[str]) -> str
            return " ".join(shlex.quote(a) for a in args)

        deps_cmd = "python3 -u {} {}".format(
            shlex.quote(script), _quote_cmd(deps_args),
        )
        run_cmd = "python3 -u {} {}".format(
            shlex.quote(script), _quote_cmd(run_args),
        )
        compound = "{} && {}".format(deps_cmd, run_cmd)
        log_verbose("Per-node command: {}".format(compound))

        # Spawn all nodes at once
        procs = {}  # type: Dict[str, subprocess.Popen]
        for host in hosts:
            if host in local_names:
                procs[host] = subprocess.Popen(
                    ["bash", "-c", compound],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
            else:
                procs[host] = subprocess.Popen(
                    _ssh_base_cmd(cfg, host) + [compound],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )

        # Reader threads stream output line-by-line from each node
        label_len = max(len(h) for h in hosts)
        output_lines = {}   # type: Dict[str, List[str]]
        print_lock = threading.Lock()

        def _reader(host, proc):
            # type: (str, subprocess.Popen) -> None
            label = _make_host_label(host, label_len)
            lines = []  # type: List[str]
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
                lines.append(line)
                with print_lock:
                    log("  [{}] {}".format(label, line))
            output_lines[host] = lines

        threads = {}  # type: Dict[str, threading.Thread]
        for host, proc in procs.items():
            t = threading.Thread(target=_reader, args=(host, proc))
            t.daemon = True
            t.start()
            threads[host] = t

        # Wait for all processes, printing a status line as each finishes
        results = {}  # type: Dict[str, Tuple[int, bytes]]
        done = 0
        while done < len(procs):
            for host, proc in procs.items():
                if host in results:
                    continue
                rc = proc.poll()
                if rc is not None:
                    threads[host].join(timeout=5)
                    text = "\n".join(output_lines.get(host, []))
                    results[host] = (rc, text.encode("utf-8"))
                    done += 1
                    status = "[OK]  " if rc == 0 else "[FAIL]"
                    with print_lock:
                        log("  {} {:<40} ({}/{})".format(
                            status, host, done, len(hosts),
                        ))
            if done < len(procs):
                time.sleep(0.5)

        failed = [h for h in hosts if results.get(h, (1, b""))[0] != 0]
        succeeded = [h for h in hosts if results.get(h, (1, b""))[0] == 0]

        if failed:
            _report_launch_failures(cfg, hosts, results, failed, succeeded)
            sys.exit(1)

    log("")
    log("=== All {} containers launched ===".format(len(hosts)))
    log("")
    log("  Verify container SSH:")
    log("    python3 -m mnctl --verify")


# ---------------------------------------------------------------------------
# Partial-failure reporting for launch_all
# ---------------------------------------------------------------------------
def _report_launch_failures(cfg, hosts, results, failed, succeeded):
    # type: (Config, List[str], Dict[str, Tuple[int, bytes]], List[str], List[str]) -> None
    """Print detailed failure info, partial-deployment summary, and next steps."""
    log("")
    log("=== Failed nodes ({}/{}) ===".format(len(failed), len(hosts)))
    for host in failed:
        rc, output = results[host]
        log("")
        log("--- {} (exit {}) ---".format(host, rc))
        text = output.decode("utf-8", errors="replace")
        for line in text.splitlines()[-30:]:
            log("  {}".format(line))

    log("")
    log("=== Partial deployment summary ===")
    log("  Succeeded : {}/{} nodes".format(len(succeeded), len(hosts)))
    log("  Failed    : {}/{} nodes".format(len(failed), len(hosts)))
    if len(failed) <= 20:
        log("  Failed on : {}".format(", ".join(failed)))
    log("")
    log("  Next steps:")
    log("    1. Retry (idempotent — skips already-running containers):")
    log("       python3 -m mnctl --launch-all")
    log("    2. Clean up ALL nodes and start fresh:")
    log("       python3 -m mnctl --stop-all")


# ---------------------------------------------------------------------------
# Stop containers on all nodes (parallel via Popen)
# ---------------------------------------------------------------------------
def stop_all(cfg):
    # type: (Config) -> None
    """Stop and remove containers on every node in the hostfile."""
    log("=== Stopping containers on all nodes ===")
    log("")

    hosts = parse_hostfile(cfg.hostfile)
    local_names = get_local_hostnames()
    stop_cmd = cfg.runtime.get_stop_cmd()

    # Spawn all stop commands at once
    jobs = {}  # type: Dict[str, List[str]]
    for host in hosts:
        if host in local_names:
            jobs[host] = ["sh", "-c", stop_cmd]
        else:
            jobs[host] = _ssh_base_cmd(cfg, host) + [stop_cmd]
    results = run_parallel(jobs)

    # Collect results (all procs ran concurrently)
    unreachable = []
    for host in hosts:
        r = results[host]
        output = r.stdout_text().strip()
        if not r.ok:
            err = r.stderr_text().strip()
            output = "[UNREACHABLE] {}".format(
                err[:200] if err else "exit {}".format(r.returncode)
            )
            unreachable.append(host)
        log("  {:<20} {}".format(host, output))

    if unreachable:
        log("")
        log("WARNING: {} node(s) could not be reached for cleanup:".format(
            len(unreachable)
        ))
        for h in unreachable:
            log("  - {}".format(h))
        log("")
        log("  Manual cleanup on unreachable nodes:")
        log("    ssh <node> docker rm -f {}".format(cfg.container_name))

    log("")
    log("=== Done ===")
