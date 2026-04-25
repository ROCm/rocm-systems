"""Common utilities: logging, path expansion, GPU detection, hostfile parsing."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set

# ---------------------------------------------------------------------------
# Module-level verbose flag (set once from main)
# ---------------------------------------------------------------------------
_verbose = False


def set_verbose(flag):
    # type: (bool) -> None
    global _verbose
    _verbose = flag


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log(msg):
    # type: (str) -> None
    print(msg, flush=True)


def log_verbose(msg):
    # type: (str) -> None
    if _verbose:
        print("  [verbose] {}".format(msg), flush=True)


def warn(msg):
    # type: (str) -> None
    print("WARNING: {}".format(msg), file=sys.stderr, flush=True)


def error(msg):
    # type: (str) -> None
    print("ERROR: {}".format(msg), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Path expansion (replaces bash expand_path + envsubst)
# ---------------------------------------------------------------------------
def expand_path(p):
    # type: (str) -> str
    """Expand ``~``, ``$VAR`` / ``${VAR}``, and resolve to absolute path."""
    if not p:
        return p
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


# ---------------------------------------------------------------------------
# Filesystem helper
# ---------------------------------------------------------------------------
def ensure_dir(path, modes=()):
    # type: (str, tuple) -> None
    """Create *path* (idempotent) and best-effort apply the first mode that sticks.

    *modes* is an ordered tuple of permission masks to try.  The first one
    that ``chmod`` accepts wins; later modes are tried only if earlier
    ones raise ``OSError`` (e.g. shared-FS where 0o777 is rejected, fall
    back to 0o755).  Pass an empty tuple to skip chmod entirely.
    """
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    for m in modes:
        try:
            os.chmod(path, m)
            return
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Subprocess helper (single-shot, both streams captured)
# ---------------------------------------------------------------------------
def run_capture(cmd, merge_stderr=False):
    # type: (list, bool) -> ParallelResult
    """Run *cmd* synchronously and return a :class:`ParallelResult`.

    Mirrors the :func:`run_parallel` result shape so single-shot and
    fan-out call sites can share post-processing helpers.  Errors are
    NOT raised; inspect ``.ok`` / ``.returncode`` on the result.
    """
    stderr_arg = subprocess.STDOUT if merge_stderr else subprocess.PIPE
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr_arg)
    return ParallelResult(p.returncode, p.stdout, p.stderr)


# ---------------------------------------------------------------------------
# GPU auto-detection
# ---------------------------------------------------------------------------
def auto_detect_gpus():
    # type: () -> int
    """Detect the number of AMD GPUs on the host via sysfs / /dev/dri."""
    count = 0
    topology = Path("/sys/class/kfd/kfd/topology/nodes")
    if topology.is_dir():
        for gpu_id_file in topology.glob("*/gpu_id"):
            try:
                if gpu_id_file.read_text().strip() != "0":
                    count += 1
            except (IOError, OSError):
                pass

    if count == 0:
        dri = Path("/dev/dri")
        if dri.is_dir():
            count = len(list(dri.glob("renderD*")))

    if count == 0:
        warn("Could not detect GPU count; set --gpus manually")

    return count


# ---------------------------------------------------------------------------
# Hostfile parsing
# ---------------------------------------------------------------------------
def parse_hostfile(path):
    # type: (str) -> List[str]
    """Parse an MPI-style hostfile into an ordered list of unique hostnames."""
    if not os.path.isfile(path):
        error("Hostfile not found: {}".format(path))
        error("  Create it:  echo 'hostname slots=8' > {}".format(path))
        sys.exit(1)

    hosts = []   # type: List[str]
    seen = set()  # type: Set[str]
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            host = line.split()[0]
            if host not in seen:
                hosts.append(host)
                seen.add(host)
    return hosts


# ---------------------------------------------------------------------------
# Local hostname detection
# ---------------------------------------------------------------------------
def get_local_hostnames():
    # type: () -> Set[str]
    """Return a set of names that identify the current host."""
    import socket
    names = {"localhost"}  # type: Set[str]
    try:
        names.add(socket.getfqdn())
    except Exception:
        pass
    try:
        names.add(socket.gethostname())
    except Exception:
        pass
    return names


# ---------------------------------------------------------------------------
# SSH command construction (single source of truth for SSH options)
# ---------------------------------------------------------------------------
def ssh_opts(port, identity=None, connect_timeout=10, batch=True):
    # type: (int, str, int, bool) -> List[str]
    """Canonical SSH option list.

    Does NOT include the ``ssh`` executable, the host, or a remote command.
    Pass ``connect_timeout=None`` to omit the ``ConnectTimeout`` option.
    """
    opts = [
        "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
    ]
    if batch:
        opts += ["-o", "BatchMode=yes"]
    if connect_timeout is not None:
        opts += ["-o", "ConnectTimeout={}".format(connect_timeout)]
    if identity:
        opts += ["-i", identity]
    return opts


def ssh_cmd(port, host, remote=None, identity=None,
            connect_timeout=10, batch=True, verbose=False):
    # type: (int, str, object, str, int, bool, bool) -> List[str]
    """Full ``ssh ...`` command list.

    *remote* may be ``None``, a single string, or an iterable of args.
    """
    cmd = ["ssh"]
    if verbose:
        cmd.append("-v")
    cmd += ssh_opts(
        port, identity=identity,
        connect_timeout=connect_timeout, batch=batch,
    )
    cmd.append(host)
    if remote is not None:
        if isinstance(remote, (list, tuple)):
            cmd += list(remote)
        else:
            cmd.append(remote)
    return cmd


def host_ssh_cmd(cfg, host, remote=None, connect_timeout=10):
    """Build an SSH command targeting a HOST node.

    Uses ``cfg.host_ssh_port`` and the shared host key from
    ``cfg.ssh.key_dir/id_rsa`` when it exists.
    """
    host_key = os.path.join(cfg.ssh.key_dir, "id_rsa")
    identity = host_key if os.path.isfile(host_key) else None
    return ssh_cmd(
        cfg.host_ssh_port, host, remote=remote,
        identity=identity, connect_timeout=connect_timeout,
    )


# ---------------------------------------------------------------------------
# Parallel subprocess fan-out
# ---------------------------------------------------------------------------
class ParallelResult(object):
    """Result of one job from :func:`run_parallel`."""

    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode, stdout, stderr):
        # type: (int, bytes, bytes) -> None
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def ok(self):
        # type: () -> bool
        return self.returncode == 0

    @property
    def stdout_text(self):
        # type: () -> str
        if not self.stdout:
            return ""
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self):
        # type: () -> str
        if not self.stderr:
            return ""
        return self.stderr.decode("utf-8", errors="replace")


def run_parallel(jobs, merge_stderr=False):
    """Spawn a batch of subprocesses concurrently and wait for all of them.

    Parameters
    ----------
    jobs : Mapping[Hashable, list[str]]
        Mapping from arbitrary key (host, (host,user), etc.) to an argv
        list.  Use ``["sh", "-c", "<shell-string>"]`` instead of
        ``shell=True``.
    merge_stderr : bool
        If True, redirect stderr into stdout (one combined byte stream).
        Each result's ``stderr`` will be ``None`` in that case.

    Returns
    -------
    dict[key, ParallelResult]
        One entry per input job, keyed by the original key.
    """
    stderr_arg = subprocess.STDOUT if merge_stderr else subprocess.PIPE
    procs = {}
    for key, cmd in jobs.items():
        procs[key] = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_arg,
        )
    results = {}
    for key, proc in procs.items():
        out, err = proc.communicate()
        results[key] = ParallelResult(proc.returncode, out, err)
    return results


# ---------------------------------------------------------------------------
# Timer context manager (verbose-mode only output)
# ---------------------------------------------------------------------------
class Timer(object):
    """Context manager that logs elapsed wall-clock time in verbose mode."""

    def __init__(self, label):
        # type: (str) -> None
        self.label = label
        self._start = 0.0

    def __enter__(self):
        # type: () -> Timer
        self._start = time.time()
        return self

    def __exit__(self, *exc):
        # type: (...) -> None
        elapsed = int(time.time() - self._start)
        m, s = divmod(elapsed, 60)
        log_verbose("\u23f1 {} completed in {}m {}s".format(self.label, m, s))
