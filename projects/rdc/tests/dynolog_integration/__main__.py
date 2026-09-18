#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single CLI entrypoint dispatching to ``dynolog_integration`` subcommands.

Usage from a workflow step::

    python3 -m dynolog_integration clone --ref <sha> --dest /tmp/dynolog
    python3 -m dynolog_integration build-wrapper --dynolog-dir /tmp/dynolog \
        --rocm-dir /opt/rocm
    python3 -m dynolog_integration run-example --binary <path> --duration 20

Each subcommand simply forwards to the matching module's ``main``.
"""

from __future__ import annotations

import sys

from . import build, clone, run_example

_SUBCOMMANDS = {"clone": clone.main, "build-wrapper": build.main, "run-example": run_example.main}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        print("Available subcommands:")
        for name in _SUBCOMMANDS:
            print(f"  {name}")
        return 0 if argv else 2

    sub = argv[0]
    handler = _SUBCOMMANDS.get(sub)
    if handler is None:
        print(f"Unknown subcommand: {sub}", file=sys.stderr)
        print(f"Available: {', '.join(_SUBCOMMANDS)}", file=sys.stderr)
        return 2
    return handler(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
