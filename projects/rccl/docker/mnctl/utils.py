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
