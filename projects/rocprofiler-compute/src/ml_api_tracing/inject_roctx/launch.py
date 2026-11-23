# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

Invoked by absolute path as python <path>/launch.py --frameworks <name>
[<name> ...] -- <target.py> [args...].
"""

import runpy
import sys
from pathlib import Path
from typing import List, Tuple

# Make ML API tracing importable when run by absolute path.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from ml_api_tracing.inject_roctx.installer import install_global_wraps  # noqa: E402


def parse_launch_arguments(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Split argv into the framework names and the workload command.

    A leading "--frameworks <name> [<name> ...]" option and an optional "--"
    separator are consumed. Framework names are all tokens until "--". Exits
    with status 2 when no workload command is left.
    """
    args = list(argv)
    frameworks: List[str] = []
    if args and args[0] == "--frameworks":
        args = args[1:]
        while args and args[0] != "--":
            frameworks.append(args[0])
            args = args[1:]
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        print(
            "usage: python <path>/launch.py [--frameworks <name> ...] -- "
            "<target.py> [args...]",
            file=sys.stderr,
        )
        sys.exit(2)
    return frameworks, args


def main() -> None:
    """Install the selected backends and run the workload script."""
    frameworks, command = parse_launch_arguments(sys.argv[1:])
    target_script = command[0]
    install_global_wraps(frameworks)
    sys.argv = [target_script] + command[1:]
    runpy.run_path(target_script, run_name="__main__")


if __name__ == "__main__":
    main()
