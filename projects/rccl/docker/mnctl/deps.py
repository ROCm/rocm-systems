"""Shared dependency builds (UCX + OpenMPI) into the shared directory.

The builds run inside a throwaway container so the host only needs
a container runtime.  Results are cached in the shared dir; re-run with
``--rebuild`` to force a fresh build.
"""

import os

from .config import Config
from .utils import log, log_verbose, Timer
from .versions import (
    UCX_VERSION, OMPI_VERSION,
    UCX_TARBALL_URL, OMPI_TARBALL_URL,
)


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


def setup_shared_deps(cfg):
    # type: (Config) -> None
    """Build UCX and OpenMPI into ``cfg.shared_dir`` (runs once)."""
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

    log_dir = os.path.join(cfg.shared_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ucx_log = os.path.join(log_dir, "ucx-build.log")
    ompi_log = os.path.join(log_dir, "ompi-build.log")

    log("=== Building shared dependencies ===")
    log("  UCX     \u2192 {}".format(ucx_dir))
    log("  OpenMPI \u2192 {}".format(ompi_dir))
    log("  Logs    \u2192 {}/".format(log_dir))
    log("")

    # Ensure the image exists first
    if not cfg.runtime.image_exists():
        log(
            "  Image {} not found; building first...".format(cfg.image_tag)
        )
        cfg.runtime.build_image()

    _build_component(
        cfg, "UCX {}".format(UCX_VERSION), _UCX_SCRIPT, ucx_log,
    )
    _build_component(
        cfg, "OpenMPI {}".format(OMPI_VERSION), _OMPI_SCRIPT, ompi_log,
    )

    log("")
    log("  Shared deps installed:")
    log("    UCX:     {}".format(ucx_dir))
    log("    OpenMPI: {}".format(ompi_dir))
    log("    Logs:    {}/".format(log_dir))
    log("")


def _build_component(cfg, label, script, log_path):
    # type: (Config, str, str, str) -> None
    """Run a build script inside a throwaway container; abort on failure."""
    with Timer("{} build".format(label)):
        log("  Building {} (log: {})...".format(label, log_path))
        rc = cfg.runtime.run_build_task(script, log_path)
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
