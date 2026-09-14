#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the GPU Agent compiler wrapper used by ``amdsmi-dme-ci.yml``.

GPU Agent's Makefile compiles generated ``.pb.cc`` files with bare
``g++`` (no ``-I`` flags), so it picks up older system protobuf headers
instead of those produced by ``make build-libs``. Separately, the
hard-coded LDFLAGS omit ``-lunwind`` even though ``libzmq.a`` references
libunwind symbols.

This script writes a tiny shell wrapper that appends ``-lunwind`` to
every g++ invocation. Centralising it here documents the workaround
and gives us a single place to delete it once upstream gpu-agent is
fixed.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_WRAPPER_TEMPLATE = '#!/bin/sh\nexec g++ "$@" -lunwind\n'


def write_wrapper(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to follow an attacker-planted symlink on world-writable tmp.
    if destination.is_symlink():
        destination.unlink()
    destination.write_text(_WRAPPER_TEMPLATE)
    destination.chmod(0o755)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "g++-wrap",
        help="Path to write the wrapper script.",
    )
    args = parser.parse_args(argv)
    write_wrapper(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
