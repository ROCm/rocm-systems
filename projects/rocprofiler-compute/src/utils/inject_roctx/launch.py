# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

Invoked by absolute path as ``python <path>/launch.py --frameworks <name>
[<name> ...] -- <target.py> [args...]``.
"""

import runpy
import sys
from pathlib import Path
from typing import List

# Make the inject_roctx package importable when run by absolute path.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from utils.inject_roctx.core import install_global_wraps  # noqa: E402


# Consume a leading "--frameworks <name> [<name> ...]" option and an optional
# "--" separator. Framework names are all tokens until "--".
args = sys.argv[1:]
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

target_script = args[0]
script_args = args[1:]

install_global_wraps(frameworks)

sys.argv = [target_script] + script_args
runpy.run_path(target_script, run_name="__main__")
