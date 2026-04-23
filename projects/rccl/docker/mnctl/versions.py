"""Parse the shared ``versions.env`` file (single source of truth).

The file lives next to the Dockerfiles (``docker/versions.env``) so that
both shell consumers (``source versions.env``) and Python consumers (this
module) read the exact same values.  Version literals must NOT be
duplicated anywhere else in the tree.
"""

import os
import re
from typing import Dict


# Path is resolved relative to the docker/ directory (the parent of this
# package), matching the convention used by Config.script_dir.
_VERSIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "versions.env",
)


def _parse(path):
    # type: (str) -> Dict[str, str]
    """Parse a simple ``KEY=VALUE`` shell-style file.

    Blank lines and ``#``-comments are ignored.  Values may be unquoted,
    single-quoted, or double-quoted.  Anything more exotic (command
    substitution, parameter expansion) is rejected so that the file
    stays trivially shell-sourceable.
    """
    result = {}
    pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=(.*)$')
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = pattern.match(line)
            if not m:
                raise ValueError(
                    "{}: cannot parse line {!r}".format(path, raw.rstrip())
                )
            key, value = m.group(1), m.group(2).strip()
            # Strip a single layer of matched quotes.
            if (len(value) >= 2 and value[0] == value[-1]
                    and value[0] in ("'", '"')):
                value = value[1:-1]
            result[key] = value
    return result


_VALUES = _parse(_VERSIONS_FILE)

UCX_VERSION = _VALUES["UCX_VERSION"]
OMPI_VERSION = _VALUES["OMPI_VERSION"]

# OpenMPI's release URLs are grouped by major.minor series, e.g.
# https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
# Derive the series from the full version so callers don't have to.
OMPI_SERIES = OMPI_VERSION.rsplit(".", 1)[0]

# Convenience tarball URLs (kept here so consumers don't reinvent them).
UCX_TARBALL_URL = (
    "https://github.com/openucx/ucx/releases/download/"
    "v{v}/ucx-{v}.tar.gz".format(v=UCX_VERSION)
)
OMPI_TARBALL_URL = (
    "https://download.open-mpi.org/release/open-mpi/"
    "v{s}/openmpi-{v}.tar.gz".format(s=OMPI_SERIES, v=OMPI_VERSION)
)
