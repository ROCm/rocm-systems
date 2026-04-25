"""Self-contained helpers used across the docker_ops submodules.

These have no dependency on Config and so can be imported freely without
risking cycles.  Keep this module tiny; anything that needs ``cfg`` belongs
in :mod:`mnctl.docker_ops.env_map`, :mod:`build`, :mod:`launch`, or
:mod:`wait`.
"""

import glob
import os
from typing import List

from ..utils import log_verbose, run_capture


def get_render_gid():
    # type: () -> str
    """Return the host's ``render`` group GID (fallback ``109`` if missing)."""
    try:
        import grp
        return str(grp.getgrnam("render").gr_gid)
    except KeyError:
        return "109"


def docker_output(cmd):
    # type: (List[str]) -> str
    """Run a docker command and return stripped stdout (best-effort)."""
    return run_capture(cmd).stdout_text.strip()


def bind_mount_rdma_libs(args):
    # type: (List[str]) -> None
    """Bind-mount host MLNX_OFED / rdma-core libraries into the container."""
    ib_lib_dir = "/usr/lib/x86_64-linux-gnu"
    for lib in ("libibverbs", "libmlx5", "libmlx4", "libefa"):
        for f in sorted(glob.glob("{}/{}.so*".format(ib_lib_dir, lib))):
            if os.path.exists(f):
                args += ["-v", "{}:{}:ro".format(f, f)]

    ib_provider = "{}/libibverbs".format(ib_lib_dir)
    if os.path.isdir(ib_provider):
        args += ["-v", "{}:{}:ro".format(ib_provider, ib_provider)]
    if os.path.isdir("/etc/libibverbs.d"):
        args += ["-v", "/etc/libibverbs.d:/etc/libibverbs.d:ro"]

    log_verbose("RDMA libs bind-mounted from host")
