"""Shared dependency builds (UCX + OpenMPI) into the shared directory.

The builds run inside a throwaway container so the host only needs
a container runtime.  Results are cached in the shared dir; re-run with
``--rebuild`` to force a fresh build.

When ``cfg.shared_dir`` lives on a network filesystem (NFS / GPFS /
Lustre / ...), the first node to invoke ``setup_shared_deps`` claims a
lock and performs the build; the others poll a completion marker so we
don't burn N redundant builds.  On local filesystems the coordination
is skipped and every node builds independently (the original behavior).

See :mod:`mnctl.shared_fs` for the detection and lock primitives.
"""

import os

from .config import Config
from .utils import ensure_dir, log, log_verbose, Timer
from .versions import (
    UCX_VERSION, OMPI_VERSION,
    UCX_TARBALL_URL, OMPI_TARBALL_URL,
)
from . import shared_fs


# UCX and OpenMPI build scripts executed inside the container.
# Version strings and tarball URLs come from versions.env via versions.py
# so there is exactly ONE place to bump a version.
_UCX_SCRIPT = (
    "set -e "
    "&& cd /tmp "
    "&& wget -q {url} "
    "&& mkdir -p ucx && tar -zxf ucx-{ver}.tar.gz "
    "-C ucx --strip-components=1 "
    "&& cd ucx && mkdir build && cd build "
    "&& ../configure --prefix=/opt/shared/ucx --with-rocm=/opt/rocm "
    "   --with-verbs --with-rdmacm --enable-mt "
    "   --disable-examples --silent "
    "&& make -j$(nproc) install "
    "&& echo '>>> UCX installed successfully'"
).format(url=UCX_TARBALL_URL, ver=UCX_VERSION)

_OMPI_SCRIPT = (
    "set -e "
    "&& cd /tmp "
    "&& wget -q {url} "
    "&& mkdir -p ompi4 && tar -zxf openmpi-{ver}.tar.gz "
    "-C ompi4 --strip-components=1 "
    "&& cd ompi4 && mkdir build && cd build "
    "&& ../configure --prefix=/opt/shared/ompi "
    "--with-ucx=/opt/shared/ucx "
    "   --disable-oshmem --disable-mpi-fortran --disable-mpi-cxx "
    "   --enable-orterun-prefix-by-default --silent "
    "&& make -j$(nproc) install "
    "&& echo '>>> OpenMPI installed successfully'"
).format(url=OMPI_TARBALL_URL, ver=OMPI_VERSION)


def setup_shared_deps(cfg, runtime):
    # type: (Config, object) -> None
    """Build UCX and OpenMPI into ``cfg.shared_dir`` (runs once).

    *runtime* is a :class:`mnctl.runtime.ContainerRuntime` used to launch
    the throwaway build container; it is passed in explicitly so this
    module never reaches into ``cfg`` for the runtime instance.

    On a shared filesystem only one node ("leader") performs the build;
    the others ("followers") wait for a completion marker.  On a local
    filesystem the coordination is bypassed and every node builds
    independently -- so callers always get a working install regardless
    of the underlying storage.
    """
    ucx_dir = os.path.join(cfg.shared_dir, "ucx")
    ompi_dir = os.path.join(cfg.shared_dir, "ompi")
    mpirun = os.path.join(ompi_dir, "bin", "mpirun")

    if (
        os.path.isfile(mpirun)
        and os.access(mpirun, os.X_OK)
        and not cfg.force_rebuild
    ):
        log("=== Shared deps already installed (use --rebuild to force) ===")
        log("")
        return

    is_shared = shared_fs.resolve_shared_fs(cfg.shared_dir, cfg.shared_fs)
    log_verbose("shared_dir={} is_shared_fs={}".format(
        cfg.shared_dir, is_shared,
    ))

    # Coordinated path: try to become leader, otherwise poll for completion.
    if is_shared:
        # On a forced rebuild, drop any stale completion marker before
        # claiming so followers entering after us don't fast-skip on
        # half-overwritten install dirs.
        if cfg.force_rebuild:
            shared_fs.remove_completion(cfg.shared_dir)

        role = shared_fs.claim_leader_or_wait(
            cfg.shared_dir, cfg.deps_lock_ttl_sec,
        )
        if role == "follower":
            log("=== Shared deps build in progress on another node ===")
            log("  Shared dir: {}".format(cfg.shared_dir))
            ok = shared_fs.wait_for_completion(
                cfg.shared_dir, cfg.deps_wait_timeout_sec,
            )
            if not ok:
                raise SystemExit(1)
            log("=== Shared deps installed by build leader ===")
            log("")
            return

    # We are either the leader (shared FS) or running standalone (local FS).
    log_dir = os.path.join(cfg.shared_dir, "logs")
    ensure_dir(log_dir)
    ucx_log = os.path.join(log_dir, "ucx-build.log")
    ompi_log = os.path.join(log_dir, "ompi-build.log")

    log("=== Building shared dependencies ===")
    log("  UCX     \u2192 {}".format(ucx_dir))
    log("  OpenMPI \u2192 {}".format(ompi_dir))
    log("  Logs    \u2192 {}/".format(log_dir))
    if is_shared:
        log("  Mode    \u2192 shared FS, this node is the build leader")
    log("")

    try:
        # Ensure the image exists first
        if not runtime.image_exists():
            log(
                "  Image {} not found; building first...".format(cfg.image_tag)
            )
            runtime.build_image()

        _build_component(
            runtime, "UCX {}".format(UCX_VERSION), _UCX_SCRIPT, ucx_log,
        )
        _build_component(
            runtime, "OpenMPI {}".format(OMPI_VERSION),
            _OMPI_SCRIPT, ompi_log,
        )

        if is_shared:
            shared_fs.write_completion(cfg.shared_dir)
    finally:
        # Always release the lock so followers can either succeed (marker
        # was written) or fail fast (marker missing).
        if is_shared:
            shared_fs.release_lock(cfg.shared_dir)

    log("")
    log("  Shared deps installed:")
    log("    UCX:     {}".format(ucx_dir))
    log("    OpenMPI: {}".format(ompi_dir))
    log("    Logs:    {}/".format(log_dir))
    log("")


def _build_component(runtime, label, script, log_path):
    # type: (object, str, str, str) -> None
    """Run a build script inside a throwaway container; abort on failure."""
    with Timer("{} build".format(label)):
        log("  Building {} (log: {})...".format(label, log_path))
        rc = runtime.run_build_task(script, log_path)
        if rc != 0:
            log("  [FAIL] {} build failed. See {}".format(label, log_path))
            _tail_log(log_path, 20)
            raise SystemExit(1)
    log("  [OK] {} installed".format(label))


def _tail_log(path, n):
    # type: (str, int) -> None
    """Print the last *n* lines of a log file."""
    try:
        with open(path) as f:
            lines = f.readlines()
        for line in lines[-n:]:
            log("    {}".format(line.rstrip()))
    except IOError:
        pass
