#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Single CLI entrypoint dispatching to ``dme_integration`` subcommands.

Usage from a workflow step::

    python3 -m dme_integration prepare-submodules --dme-repo ... --dme-dir ...
    python3 -m dme_integration prepare-build-env --gpuagent-src ...
    python3 -m dme_integration write-gpp-wrapper --output /tmp/g++-wrap
    python3 -m dme_integration start-service --name gpuagent --binary ...
    python3 -m dme_integration stop-service --name gpuagent --pid-file ...
    python3 -m dme_integration verify-metrics --url http://localhost:5000/metrics

Each subcommand simply forwards to the matching module's ``main``.
"""

from __future__ import annotations

import sys

from . import build_env, gpp_wrapper, metrics, services, submodules


def _start_service(argv: list[str]) -> int:
    return services.main(["start", *argv])


def _stop_service(argv: list[str]) -> int:
    return services.main(["stop", *argv])


def _check_alive(argv: list[str]) -> int:
    return services.main(["check-alive", *argv])


_SUBCOMMANDS = {
    "prepare-submodules": submodules.main,
    "prepare-build-env": build_env.main,
    "write-gpp-wrapper": gpp_wrapper.main,
    "start-service": _start_service,
    "stop-service": _stop_service,
    "check-alive": _check_alive,
    "verify-metrics": metrics.main,
}


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
