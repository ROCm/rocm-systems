"""Dry-run pre-flight checks.

Validates that all prerequisites are met for the configured action
without actually building, launching, or modifying anything.
Checks are context-dependent: multi-node actions verify remote SSH
and Docker availability; build actions verify Dockerfile and base image.
"""

import os
import shutil
import subprocess
import sys
from typing import List

from .config import Action, Config
from .utils import log, log_verbose, parse_hostfile, get_local_hostnames


_passed = 0
_failed = 0
_warned = 0


def _ok(msg):
    # type: (str) -> None
    global _passed
    _passed += 1
    log("  [OK]   {}".format(msg))


def _fail(msg):
    # type: (str) -> None
    global _failed
    _failed += 1
    log("  [FAIL] {}".format(msg))


def _warn(msg):
    # type: (str) -> None
    global _warned
    _warned += 1
    log("  [WARN] {}".format(msg))


def run_preflight(cfg):
    # type: (Config) -> None
    """Run all pre-flight checks for the configured action, then exit."""
    global _passed, _failed, _warned
    _passed = _failed = _warned = 0

    action = cfg.action
    log("=== Dry run: pre-flight checks for --{} ===".format(action.value))
    log("")

    _check_docker_local()
    _check_gpus(cfg)

    needs_build = action in (
        Action.BUILD, Action.RUN, Action.LAUNCH_ALL, Action.SETUP_DEPS,
    )
    if needs_build:
        _check_dockerfile(cfg)
        _check_base_image(cfg)

    _check_ssh_keys(cfg)

    if action in (Action.LAUNCH_ALL, Action.STOP_ALL):
        hosts = _check_hostfile(cfg)
        if hosts:
            _check_ssh_hosts(cfg, hosts)
            if action == Action.LAUNCH_ALL:
                _check_rsync()
                _check_docker_remote(cfg, hosts)
    elif action == Action.VERIFY:
        _check_hostfile(cfg)

    if cfg.post_setup_dir:
        _check_post_setup(cfg)

    log("")
    log("=== Dry run summary ===")
    log("  Action    : --{}".format(action.value))
    if needs_build:
        log("  Image     : {}".format(cfg.image_tag))
        log("  Dockerfile: {}".format(cfg.dockerfile))
    if action in (Action.RUN, Action.LAUNCH_ALL):
        log("  Container : {}".format(cfg.container_name))
        log("  GPUs      : {}".format(cfg.gpus))
    if action in (Action.LAUNCH_ALL, Action.STOP_ALL, Action.VERIFY):
        if os.path.isfile(cfg.hostfile):
            n = len(parse_hostfile(cfg.hostfile))
            log("  Nodes     : {}".format(n))
    log("")
    log("  Passed    : {}".format(_passed))
    if _warned:
        log("  Warnings  : {}".format(_warned))
    if _failed:
        log("  Failed    : {}".format(_failed))
    log("")

    if _failed:
        log("  Fix the issues above before running without --dry-run.")
        sys.exit(1)
    else:
        log("  All checks passed. Remove --dry-run to execute.")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_docker_local():
    # type: () -> None
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        _ok("Docker daemon running locally")
    else:
        _fail("Docker daemon not running locally")


def _check_rsync():
    # type: () -> None
    if shutil.which("rsync"):
        _ok("rsync available (needed for file distribution)")
    else:
        _fail("rsync not found (install: apt-get install rsync / yum install rsync)")


def _check_gpus(cfg):
    # type: (Config) -> None
    count = int(cfg.gpus) if cfg.gpus else 0
    if count > 0:
        _ok("GPUs: {} detected".format(count))
    else:
        _warn("No GPUs detected (set GPUS env var if needed)")


def _check_dockerfile(cfg):
    # type: (Config) -> None
    if os.path.isabs(cfg.dockerfile):
        df_path = cfg.dockerfile
    else:
        df_path = os.path.join(cfg.script_dir, cfg.dockerfile)
    if os.path.isfile(df_path):
        _ok("Dockerfile: {}".format(cfg.dockerfile))
    else:
        _fail("Dockerfile not found: {}".format(df_path))


def _check_base_image(cfg):
    # type: (Config) -> None
    result = subprocess.run(
        ["docker", "image", "inspect", cfg.rocm_image],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        _ok("Base image '{}' available locally".format(cfg.rocm_image))
    else:
        _warn(
            "Base image '{}' not found locally (will attempt pull)".format(
                cfg.rocm_image
            )
        )


def _check_ssh_keys(cfg):
    # type: (Config) -> None
    if cfg.ssh.key:
        priv = cfg.ssh.priv_key
        pub = cfg.ssh.pub_key
        if priv and os.path.isfile(priv) and pub and os.path.isfile(pub):
            _ok("SSH key pair: {}".format(cfg.ssh.key))
        else:
            _fail("SSH key not found: {} / {}".format(priv, pub))
    elif cfg.ssh.keygen:
        _ok("SSH keys: will auto-generate (--ssh)")
    else:
        if cfg.action in (Action.LAUNCH_ALL, Action.VERIFY):
            _warn("No SSH keys configured (use --ssh or --ssh KEY_PATH)")
        else:
            log_verbose("SSH keys: not configured (optional for single-node)")


def _check_hostfile(cfg):
    # type: (Config) -> List[str]
    if not os.path.isfile(cfg.hostfile):
        _fail("Hostfile not found: {}".format(cfg.hostfile))
        return []

    hosts = parse_hostfile(cfg.hostfile)
    if hosts:
        _ok("Hostfile: {} ({} nodes)".format(cfg.hostfile, len(hosts)))
        for h in hosts:
            log("           - {}".format(h))
    else:
        _fail("Hostfile is empty: {}".format(cfg.hostfile))
    return hosts


def _check_ssh_hosts(cfg, hosts):
    # type: (Config, List[str]) -> None
    """Check SSH reachability to each host (parallel via Popen)."""
    local_names = get_local_hostnames()

    procs = {}  # type: dict
    for host in hosts:
        if host in local_names:
            continue
        procs[host] = subprocess.Popen(
            [
                "ssh",
                "-p", str(cfg.host_ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                host, "hostname",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    for host in hosts:
        if host in local_names:
            _ok("SSH: {} (local)".format(host))
            continue
        proc = procs[host]
        proc.wait()
        if proc.returncode == 0:
            _ok("SSH: {} reachable".format(host))
        else:
            _fail("SSH: {} unreachable".format(host))


def _check_docker_remote(cfg, hosts):
    # type: (Config, List[str]) -> None
    """Check Docker daemon is running on each remote host (parallel via Popen)."""
    local_names = get_local_hostnames()

    procs = {}  # type: dict
    for host in hosts:
        if host in local_names:
            continue
        procs[host] = subprocess.Popen(
            [
                "ssh",
                "-p", str(cfg.host_ssh_port),
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                host, "docker", "info",
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    for host in hosts:
        if host in local_names:
            continue
        proc = procs[host]
        proc.wait()
        if proc.returncode == 0:
            _ok("Docker: {} running".format(host))
        else:
            _fail("Docker: {} not reachable or daemon not running".format(host))


def _check_post_setup(cfg):
    # type: (Config) -> None
    d = cfg.post_setup_dir
    if not os.path.isdir(d):
        _fail("Post-setup directory not found: {}".format(d))
        return
    has_setup = os.path.isfile(os.path.join(d, "setup.sh"))
    has_env = os.path.isfile(os.path.join(d, "env.sh"))
    if has_setup or has_env:
        files = []
        if has_setup:
            files.append("setup.sh")
        if has_env:
            files.append("env.sh")
        _ok("Post-setup: {} ({})".format(d, ", ".join(files)))
    else:
        _fail("Post-setup dir has no setup.sh or env.sh: {}".format(d))
