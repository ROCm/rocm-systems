# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

Invoked by absolute path as ``python <path>/launch.py --frameworks <name>
[<name> ...] -- <target.py> [args...]``. When ``--frameworks`` is omitted the
workload runs uninstrumented.
"""

import runpy
import sys
from pathlib import Path

# Make the inject_roctx package importable when run by absolute path.
_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

from utils.inject_roctx.core import install_global_wraps  # noqa: E402


def _report_torch_trace_callback_errors() -> None:
    """Warn when the collector reports callback errors."""
    torch_backend = sys.modules.get("utils.inject_roctx._backends.torch")
    if torch_backend is None:
        return
    stats = torch_backend.dump_torch_trace_stats()
    if not stats:
        return
    callback_errors = int(stats.get("callback_errors", 0) or 0)
    if callback_errors <= 0:
        return
    from utils.logger import console_warning

    console_warning(
        "ml api trace",
        f"torch_trace_collector reported {callback_errors} callback error(s). "
        f"Some ROCTX markers may be missing or misattributed. Stats: {dict(stats)}",
    )


# Consume a leading "--frameworks <name> [<name> ...]" option and an optional
# "--" separator. Framework names are all tokens until "--".
args = sys.argv[1:]
frameworks: list[str] = []
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
# Execute the workload as the top-level program (__name__ == "__main__").
try:
    runpy.run_path(target_script, run_name="__main__")
finally:
    _report_torch_trace_callback_errors()
