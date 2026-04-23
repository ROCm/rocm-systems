"""Shared-filesystem detection and leader-election primitives.

Used by ``deps.py`` so that on a network-shared ``MNCTL_SHARED_DIR``
(NFS, GPFS, Lustre, ...) only one node performs the UCX/OpenMPI build
while the others poll for completion.  On a local-only filesystem the
helpers become no-ops and every node builds independently (the existing
behavior).

The lock is a plain file created with ``O_CREAT | O_EXCL``.  This is
atomic on NFSv3+ for create races and is portable across the shared
filesystems we care about.  No ``flock`` is needed.

Layout under ``shared_dir``:
    .deps_lock      held by the active builder; ``hostname:pid:ts`` body
    .deps_complete  written atomically (rename) when build succeeds
"""

import errno
import os
import socket
import subprocess
import time
from typing import Optional

from .utils import log, log_verbose


# Filesystem type names returned by `stat -f -c %T` that we treat as shared.
_SHARED_FS_TYPES = frozenset({
    "nfs", "nfs4",
    "gpfs",
    "lustre", "lustrefs",
    "ceph",
    "glusterfs", "fuse.glusterfs",
    "cifs", "smb3", "smb2",
    "beegfs", "fuse.beegfs",
    "panfs",
})

LOCK_NAME = ".deps_lock"
COMPLETE_NAME = ".deps_complete"


def detect_shared_fs(path):
    # type: (str) -> bool
    """Return True if *path* lives on a recognized shared filesystem.

    Uses ``stat -f -c %T``; falls back to False on any error so that an
    unknown filesystem is treated as local (i.e. safer fallback).
    """
    if not os.path.isdir(path):
        return False
    try:
        out = subprocess.check_output(
            ["stat", "-f", "-c", "%T", path],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", "replace").strip().lower()
    except (subprocess.CalledProcessError, OSError):
        return False
    return out in _SHARED_FS_TYPES


def resolve_shared_fs(path, override):
    # type: (str, str) -> bool
    """Apply the ``MNCTL_SHARED_FS`` override on top of auto-detection.

    ``override`` is one of ``"auto"`` (detect), ``"yes"`` (force shared),
    or ``"no"`` (force local).  Anything else falls back to ``"auto"``.
    """
    o = (override or "auto").strip().lower()
    if o in ("yes", "true", "1", "shared"):
        return True
    if o in ("no", "false", "0", "local"):
        return False
    return detect_shared_fs(path)


def _lock_body():
    # type: () -> str
    """Identifying contents written into the lock file."""
    return "{}:{}:{:.0f}\n".format(
        socket.gethostname(), os.getpid(), time.time(),
    )


def _read_lock(lock_path):
    # type: (str) -> Optional[str]
    """Return the lock file's first line, or None if unreadable."""
    try:
        with open(lock_path, "r") as f:
            return f.readline().strip()
    except (IOError, OSError):
        return None


def _lock_age_seconds(lock_path):
    # type: (str) -> Optional[float]
    """Age of the lock file in seconds, or None if unreadable."""
    try:
        return time.time() - os.path.getmtime(lock_path)
    except (IOError, OSError):
        return None


def claim_leader_or_wait(shared_dir, ttl_seconds):
    # type: (str, float) -> str
    """Try to become the build leader; return ``"leader"`` or ``"follower"``.

    Atomically creates ``<shared_dir>/.deps_lock``.  If the file already
    exists and is older than *ttl_seconds*, the stale lock is removed
    and a fresh attempt is made (caller may still race and lose, in
    which case it becomes a follower).
    """
    lock_path = os.path.join(shared_dir, LOCK_NAME)
    body = _lock_body().encode("utf-8")

    for attempt in range(2):
        try:
            fd = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            # Lock present; check whether it is stale.
            age = _lock_age_seconds(lock_path)
            holder = _read_lock(lock_path) or "<unknown>"
            if age is not None and age > ttl_seconds and attempt == 0:
                log_verbose(
                    "Stealing stale deps lock (age={:.0f}s, holder={})"
                    .format(age, holder)
                )
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            log_verbose(
                "Deps lock held by {} (age={}s); becoming follower"
                .format(
                    holder,
                    "{:.0f}".format(age) if age is not None else "?",
                )
            )
            return "follower"
        else:
            try:
                os.write(fd, body)
            finally:
                os.close(fd)
            return "leader"

    return "follower"


def release_lock(shared_dir):
    # type: (str) -> None
    """Best-effort removal of the lock file (no-op if missing)."""
    lock_path = os.path.join(shared_dir, LOCK_NAME)
    try:
        os.remove(lock_path)
    except OSError:
        pass


def write_completion(shared_dir):
    # type: (str) -> None
    """Atomically write the completion marker (rename within shared_dir)."""
    final = os.path.join(shared_dir, COMPLETE_NAME)
    tmp = final + ".tmp.{}".format(os.getpid())
    body = "{}:{:.0f}\n".format(socket.gethostname(), time.time())
    with open(tmp, "w") as f:
        f.write(body)
    # rename within the same directory is atomic on POSIX and NFS.
    os.rename(tmp, final)


def remove_completion(shared_dir):
    # type: (str) -> None
    """Drop a stale completion marker (used at start of a forced rebuild)."""
    try:
        os.remove(os.path.join(shared_dir, COMPLETE_NAME))
    except OSError:
        pass


def wait_for_completion(shared_dir, timeout_seconds, poll_seconds=5.0,
                        progress_seconds=30.0):
    # type: (str, float, float, float) -> bool
    """Poll for ``.deps_complete``; return True on success, False on timeout.

    Returns False if the leader's lock disappears without a completion
    marker appearing -- that signals the leader gave up (build failed
    or process killed) and the caller should abort.
    """
    complete = os.path.join(shared_dir, COMPLETE_NAME)
    lock = os.path.join(shared_dir, LOCK_NAME)
    deadline = time.time() + timeout_seconds
    last_progress = time.time()
    saw_lock = os.path.exists(lock)

    log("  Waiting for build leader to finish (polling every {}s)..."
        .format(max(1, int(round(poll_seconds)))))

    while time.time() < deadline:
        if os.path.exists(complete):
            return True
        # If we ever observed the lock and it's now gone without a marker,
        # the leader bailed out.  No point waiting further.
        if saw_lock and not os.path.exists(lock):
            log("  [FAIL] Build leader released the lock without finishing.")
            log("         Check {}/logs/ on the leader for details."
                .format(shared_dir))
            return False
        if not saw_lock and os.path.exists(lock):
            saw_lock = True

        now = time.time()
        if now - last_progress >= progress_seconds:
            elapsed = int(now - (deadline - timeout_seconds))
            holder = _read_lock(lock) or "<unknown>"
            log("  ... still waiting (elapsed {}s, holder {})"
                .format(elapsed, holder))
            last_progress = now

        time.sleep(poll_seconds)

    log("  [FAIL] Timed out after {}s waiting for leader."
        .format(int(timeout_seconds)))
    return False
