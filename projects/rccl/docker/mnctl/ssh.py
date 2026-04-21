"""SSH key management and connectivity verification.

Handles:
  - Installing user-supplied or auto-generated SSH keys into the shared key dir
  - Writing the SSH client config for container-to-container access
  - Verifying passwordless SSH to every host in the hostfile (parallel via Popen)
"""

import os
import shutil
import subprocess
import sys
from typing import Dict, Optional, Tuple

from .config import Config
from .utils import log, log_verbose, get_local_hostnames, Timer


def write_ssh_config(cfg):
    # type: (Config) -> None
    """Write SSH client config and set key/config permissions."""
    key_dir = cfg.ssh.key_dir
    config_path = os.path.join(key_dir, "config")

    with open(config_path, "w") as f:
        f.write("Host *\n")
        f.write("    StrictHostKeyChecking no\n")
        f.write("    UserKnownHostsFile /dev/null\n")
        f.write("    LogLevel ERROR\n")
        f.write("    Port {}\n".format(cfg.ssh.port))
        f.write("    IdentityFile ~/.ssh/id_rsa\n")

    os.chmod(os.path.join(key_dir, "id_rsa"), 0o600)
    for name in ("id_rsa.pub", "authorized_keys", "config"):
        path = os.path.join(key_dir, name)
        if os.path.isfile(path):
            os.chmod(path, 0o644)


def install_ssh_keys(cfg):
    # type: (Config) -> None
    """Install SSH keys into the shared key directory (idempotent)."""
    key_dir = cfg.ssh.key_dir
    id_rsa = os.path.join(key_dir, "id_rsa")

    if os.path.isfile(id_rsa):
        log("  SSH keys exist at {}".format(key_dir))
        pub_key = id_rsa + ".pub"
        if os.path.isfile(pub_key):
            _add_to_host_authorized_keys(pub_key)
        return

    if cfg.ssh.key:
        _install_from_existing(cfg, key_dir)
    elif cfg.ssh.keygen:
        _generate_new_keys(cfg, key_dir, id_rsa)
    else:
        log("  No SSH keys configured (use --ssh for multi-node)")
        log_verbose("Hint: for multi-node SSH, use one of:")
        log_verbose("  --ssh ~/.ssh/id_rsa   # use your existing key pair")
        log_verbose("  --ssh                 # generate a new shared pair")


def _add_to_host_authorized_keys(pub_key_path):
    # type: (str) -> None
    """Append the public key to the host's ~/.ssh/authorized_keys (idempotent)."""
    host_ssh_dir = os.path.join(os.path.expanduser("~"), ".ssh")
    if not os.path.isdir(host_ssh_dir):
        os.makedirs(host_ssh_dir, mode=0o700, exist_ok=True)

    auth_keys = os.path.join(host_ssh_dir, "authorized_keys")
    with open(pub_key_path, "r") as f:
        pub_data = f.read().strip()

    if os.path.isfile(auth_keys):
        with open(auth_keys, "r") as f:
            existing = f.read()
        if pub_data in existing:
            log_verbose("Public key already in host authorized_keys")
            return

    with open(auth_keys, "a") as f:
        f.write(pub_data + "\n")
    os.chmod(auth_keys, 0o600)
    log_verbose("Added public key to {}".format(auth_keys))


def _install_from_existing(cfg, key_dir):
    # type: (Config, str) -> None
    priv = cfg.ssh.priv_key
    pub = cfg.ssh.pub_key
    log("  Configuring SSH keys from {}...".format(cfg.ssh.key))

    shutil.copy2(priv, os.path.join(key_dir, "id_rsa"))
    shutil.copy2(pub, os.path.join(key_dir, "id_rsa.pub"))
    shutil.copy2(pub, os.path.join(key_dir, "authorized_keys"))

    write_ssh_config(cfg)
    _add_to_host_authorized_keys(os.path.join(key_dir, "id_rsa.pub"))
    log("  SSH keys configured at {}".format(key_dir))


def _generate_new_keys(cfg, key_dir, id_rsa):
    # type: (Config, str, str) -> None
    log("  Generating shared SSH keys...")
    subprocess.run(
        [
            "ssh-keygen", "-t", "rsa", "-b", "4096",
            "-N", "", "-f", id_rsa, "-C", "docker-shared-key", "-q",
        ],
        check=True,
    )

    pub_file = id_rsa + ".pub"
    shutil.copy2(pub_file, os.path.join(key_dir, "authorized_keys"))

    write_ssh_config(cfg)
    _add_to_host_authorized_keys(pub_file)
    log("  Keys generated at {}".format(key_dir))


# ---------------------------------------------------------------------------
# SSH connectivity verification
# ---------------------------------------------------------------------------
def _detect_container_user(cfg):
    # type: (Config) -> Optional[str]
    """Query the running container for its CONTAINER_USER env var."""
    try:
        result = subprocess.run(
            ["docker", "exec", cfg.container_name,
             "bash", "-c", "echo $CONTAINER_USER"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
        )
        if result.returncode == 0:
            user = result.stdout.decode("utf-8", errors="replace").strip()
            if user:
                log_verbose(
                    "Detected container user '{}' from running container"
                    .format(user)
                )
                return user
    except Exception:
        pass
    return None


def verify_ssh(cfg):
    # type: (Config) -> None
    """Verify SSH connectivity to all hosts in the hostfile."""
    from .utils import parse_hostfile

    with Timer("SSH verification"):
        log("=== Verifying SSH connectivity (port {}) ===".format(cfg.ssh.port))

        if not os.path.isfile(cfg.hostfile):
            log("  Hostfile not found: {}".format(cfg.hostfile))
            log(
                "  Create it first:  echo 'hostname slots=8' > {}".format(
                    cfg.hostfile
                )
            )
            sys.exit(1)

        log_verbose("Hostfile: {}".format(cfg.hostfile))
        if cfg.verbose:
            log_verbose("Hostfile contents:")
            with open(cfg.hostfile) as f:
                for line in f:
                    log_verbose("  {}".format(line.rstrip()))

        hosts = parse_hostfile(cfg.hostfile)

        ssh_key = os.path.join(cfg.ssh.key_dir, "id_rsa")
        if not os.path.isfile(ssh_key):
            log("  Shared SSH key not found: {}".format(ssh_key))
            log("")
            log("  Set up SSH keys first:")
            log(
                "    python3 -m mnctl --launch-all "
                "--ssh ~/.ssh/id_rsa   # use your key pair"
            )
            log(
                "    python3 -m mnctl --launch-all "
                "--ssh                 # generate a new pair"
            )
            sys.exit(1)

        log_verbose("Using SSH key: {}".format(ssh_key))

        ssh_opts = [
            "-p", str(cfg.ssh.port),
            "-i", ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=5",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
        ]

        container_user = _detect_container_user(cfg) or cfg.container_user
        test_users = ["root", container_user]

        log_verbose(
            "Verifying {} hosts x {} users ({})".format(
                len(hosts), len(test_users),
                ", ".join(test_users),
            )
        )

        # Spawn all checks at once
        procs = {}  # type: Dict[Tuple[str, str], subprocess.Popen]
        for host in hosts:
            for user in test_users:
                procs[(host, user)] = subprocess.Popen(
                    ["ssh"] + ssh_opts
                    + ["{}@{}".format(user, host), "hostname"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

        # Collect results (all procs already running concurrently)
        results = {}  # type: Dict[Tuple[str, str], bool]
        for key, proc in procs.items():
            proc.wait()
            results[key] = proc.returncode == 0

        # Print in hostfile order
        failed = False
        for host in hosts:
            for user in test_users:
                ok = results[(host, user)]
                if ok:
                    log("  [OK]   {}@{}".format(user, host))
                else:
                    log("  [FAIL] {}@{}".format(user, host))
                    failed = True
                    if cfg.verbose:
                        _log_ssh_debug(ssh_opts, user, host)

        if failed:
            _print_ssh_fix_hints(cfg)
            sys.exit(1)

        # Self-SSH: verify each container can SSH to itself (needed by MPI)
        log("")
        log("=== Verifying container self-SSH (localhost) ===")
        self_failed = _verify_self_ssh(cfg, hosts)
        if self_failed:
            _print_ssh_fix_hints(cfg)
            sys.exit(1)

    log("")
    log(
        "All hosts reachable (as {}). Ready for MPI workloads.".format(
            " ".join(test_users)
        )
    )


def _verify_self_ssh(cfg, hosts):
    # type: (Config, list) -> bool
    """Verify each container can SSH to localhost (parallel via Popen).

    Returns True if any host failed.
    """
    local_names = get_local_hostnames()
    self_ssh_cmd = [
        "ssh",
        "-p", str(cfg.ssh.port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
        "localhost", "hostname",
    ]
    exec_cmd = ["docker", "exec", cfg.container_name] + self_ssh_cmd

    host_ssh_base = [
        "ssh",
        "-p", str(cfg.host_ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
    ]

    procs = {}  # type: dict
    for host in hosts:
        if host in local_names:
            procs[host] = subprocess.Popen(
                exec_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        else:
            procs[host] = subprocess.Popen(
                host_ssh_base + [host] + exec_cmd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )

    failed = False
    for host in hosts:
        proc = procs[host]
        proc.wait()
        if proc.returncode == 0:
            log("  [OK]   {} -> localhost".format(host))
        else:
            log("  [FAIL] {} -> localhost (container cannot SSH to itself)".format(host))
            failed = True
            if cfg.verbose:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                for line in stderr.splitlines()[-10:]:
                    log_verbose("  {}".format(line))
    return failed


def _log_ssh_debug(ssh_opts, user, host):
    # type: (list, str, str) -> None
    log_verbose("SSH debug for {}@{}:".format(user, host))
    debug_result = subprocess.run(
        ["ssh", "-v"] + ssh_opts + ["{}@{}".format(user, host), "hostname"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    stderr = debug_result.stderr.decode("utf-8", errors="replace")
    for line in stderr.splitlines()[-20:]:
        log_verbose("  {}".format(line))


def _print_ssh_fix_hints(cfg):
    # type: (Config) -> None
    log("")
    log("Fix failed hosts:")
    log("  1. Ensure the container is running: docker ps")
    log(
        "  2. Check sshd: docker exec <container> "
        "ss -tlnp | grep {}".format(cfg.ssh.port)
    )
    log(
        "  3. Restart sshd: docker exec <container> "
        "/usr/sbin/sshd -p{}".format(cfg.ssh.port)
    )
    log("")
    log("  If SSH keys are not set up, re-launch with:")
    log(
        "    python3 -m mnctl --launch-all "
        "--ssh ~/.ssh/id_rsa   # use your key pair"
    )
    log(
        "    python3 -m mnctl --launch-all "
        "--ssh                 # generate a new pair"
    )
