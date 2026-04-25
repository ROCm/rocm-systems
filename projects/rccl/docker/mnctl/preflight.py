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

from .checks import (
    SSH_KEY_OK, SSH_KEY_KEYGEN, SSH_KEY_MISSING, SSH_KEY_NONE,
    POST_SETUP_OK, POST_SETUP_MISSING, POST_SETUP_EMPTY,
    hostfile_status, ssh_key_status, post_setup_statuses,
    resolve_dockerfile,
)
from .config import Action, Config
from .utils import (
    log, log_verbose, parse_hostfile, get_local_hostnames,
    host_ssh_cmd, run_parallel,
)


class PreflightReport(object):
    """Accumulates pass/fail/warn results from a single preflight run.

    Each check function receives a ``PreflightReport`` instance and records
    its outcome through the ``ok`` / ``fail`` / ``warn`` methods.  This
    keeps the module reentrant and makes the data flow explicit, replacing
    the previous module-level ``_passed/_failed/_warned`` globals.
    """

    __slots__ = ("passed", "failed", "warned")

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warned = 0

    def ok(self, msg):
        # type: (str) -> None
        self.passed += 1
        log("  [OK]   {}".format(msg))

    def fail(self, msg):
        # type: (str) -> None
        self.failed += 1
        log("  [FAIL] {}".format(msg))

    def warn(self, msg):
        # type: (str) -> None
        self.warned += 1
        log("  [WARN] {}".format(msg))


def run_preflight(cfg):
    # type: (Config) -> None
    """Run all pre-flight checks for the configured action, then exit."""
    report = PreflightReport()

    action = cfg.action
    log("=== Dry run: pre-flight checks for --{} ===".format(action.value))
    log("")

    _check_docker_local(report)
    _check_gpus(cfg, report)

    needs_build = action in (
        Action.BUILD, Action.RUN, Action.LAUNCH_ALL, Action.SETUP_DEPS,
    )
    if needs_build:
        _check_dockerfile(cfg, report)
        _check_base_image(cfg, report)

    _check_ssh_keys(cfg, report)

    if action in (Action.LAUNCH_ALL, Action.STOP_ALL):
        hosts = _check_hostfile(cfg, report)
        if hosts:
            _check_ssh_hosts(cfg, hosts, report)
            if action == Action.LAUNCH_ALL:
                _check_rsync(report)
                _check_docker_remote(cfg, hosts, report)
    elif action == Action.VERIFY:
        _check_hostfile(cfg, report)

    if cfg.post_setup_dirs:
        _check_post_setup(cfg, report)

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
    log("  Passed    : {}".format(report.passed))
    if report.warned:
        log("  Warnings  : {}".format(report.warned))
    if report.failed:
        log("  Failed    : {}".format(report.failed))
    log("")

    if report.failed:
        log("  Fix the issues above before running without --dry-run.")
        sys.exit(1)
    else:
        log("  All checks passed. Remove --dry-run to execute.")
        sys.exit(0)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_docker_local(report):
    # type: (PreflightReport) -> None
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        report.ok("Docker daemon running locally")
    else:
        report.fail("Docker daemon not running locally")


def _check_rsync(report):
    # type: (PreflightReport) -> None
    if shutil.which("rsync"):
        report.ok("rsync available (needed for file distribution)")
    else:
        report.fail("rsync not found (install: apt-get install rsync / yum install rsync)")


def _check_gpus(cfg, report):
    # type: (Config, PreflightReport) -> None
    count = int(cfg.gpus) if cfg.gpus else 0
    if count > 0:
        report.ok("GPUs: {} detected".format(count))
    else:
        report.warn("No GPUs detected (set GPUS env var if needed)")


def _check_dockerfile(cfg, report):
    # type: (Config, PreflightReport) -> None
    df_path = resolve_dockerfile(cfg)
    if os.path.isfile(df_path):
        report.ok("Dockerfile: {}".format(cfg.dockerfile))
    else:
        report.fail("Dockerfile not found: {}".format(df_path))


def _check_base_image(cfg, report):
    # type: (Config, PreflightReport) -> None
    if cfg.runtime.image_exists(cfg.rocm_image):
        report.ok("Base image '{}' available locally".format(cfg.rocm_image))
    else:
        report.warn(
            "Base image '{}' not found locally (will attempt pull)".format(
                cfg.rocm_image
            )
        )


def _check_ssh_keys(cfg, report):
    # type: (Config, PreflightReport) -> None
    status, priv, pub = ssh_key_status(cfg)
    if status == SSH_KEY_OK:
        report.ok("SSH key pair: {}".format(cfg.ssh.key))
    elif status == SSH_KEY_MISSING:
        report.fail("SSH key not found: {} / {}".format(priv, pub))
    elif status == SSH_KEY_KEYGEN:
        report.ok("SSH keys: will auto-generate (--ssh)")
    else:  # SSH_KEY_NONE
        if cfg.action in (Action.LAUNCH_ALL, Action.VERIFY):
            report.warn("No SSH keys configured (use --ssh or --ssh KEY_PATH)")
        else:
            log_verbose("SSH keys: not configured (optional for single-node)")


def _check_hostfile(cfg, report):
    # type: (Config, PreflightReport) -> List[str]
    exists, hosts = hostfile_status(cfg)
    if not exists:
        report.fail("Hostfile not found: {}".format(cfg.hostfile))
        return []
    if hosts:
        report.ok("Hostfile: {} ({} nodes)".format(cfg.hostfile, len(hosts)))
        for h in hosts:
            log("           - {}".format(h))
    else:
        report.fail("Hostfile is empty: {}".format(cfg.hostfile))
    return hosts


def _check_ssh_hosts(cfg, hosts, report):
    # type: (Config, List[str], PreflightReport) -> None
    """Check SSH reachability to each host (parallel via Popen)."""
    local_names = get_local_hostnames()
    jobs = {
        host: host_ssh_cmd(cfg, host, remote="hostname", connect_timeout=5)
        for host in hosts if host not in local_names
    }
    results = run_parallel(jobs)

    for host in hosts:
        if host in local_names:
            report.ok("SSH: {} (local)".format(host))
        elif results[host].ok:
            report.ok("SSH: {} reachable".format(host))
        else:
            report.fail("SSH: {} unreachable".format(host))


def _check_docker_remote(cfg, hosts, report):
    # type: (Config, List[str], PreflightReport) -> None
    """Check Docker daemon is running on each remote host (parallel via Popen)."""
    local_names = get_local_hostnames()
    jobs = {
        host: host_ssh_cmd(
            cfg, host, remote=["docker", "info"], connect_timeout=5,
        )
        for host in hosts if host not in local_names
    }
    results = run_parallel(jobs)

    for host in hosts:
        if host in local_names:
            continue
        if results[host].ok:
            report.ok("Docker: {} running".format(host))
        else:
            report.fail("Docker: {} not reachable or daemon not running".format(host))


def _check_post_setup(cfg, report):
    # type: (Config, PreflightReport) -> None
    """Validate every dir in cfg.post_setup_dirs (CLI order)."""
    for d, status, files in post_setup_statuses(cfg):
        if status == POST_SETUP_MISSING:
            report.fail("Post-setup directory not found: {}".format(d))
        elif status == POST_SETUP_EMPTY:
            report.fail("Post-setup dir has no setup.sh or env.sh: {}".format(d))
        else:  # POST_SETUP_OK
            report.ok("Post-setup: {} ({})".format(d, ", ".join(files)))
