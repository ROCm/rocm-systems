# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

Invoked by absolute path as ``python <path>/launch.py --frameworks <name>
[<name> ...] [--capture-args <0|1>] [--capture-arg-values <0|1>] --
<target.py> [args...]``. The capture options configure operator-argument capture.
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

_LAUNCHER_OPTIONS = ("--frameworks", "--capture-args", "--capture-arg-values")


def _flag(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_launcher_options(
    argv: list[str],
) -> tuple[list[str], bool, bool, list[str]]:
    """Parse leading launcher options and the ``--`` separator.

    Returns ``(frameworks, capture_args, capture_arg_values, remaining)`` where
    ``remaining`` is the workload command following ``--``.
    """
    args = list(argv)
    frameworks: list[str] = []
    capture_args = True
    capture_arg_values = False
    while args:
        if args[0] == "--":
            args = args[1:]
            break
        if args[0] == "--frameworks":
            args = args[1:]
            while args and args[0] not in (
                "--",
                "--capture-args",
                "--capture-arg-values",
            ):
                frameworks.append(args[0])
                args = args[1:]
            continue
        if args[0] == "--capture-args" and len(args) > 1:
            capture_args = _flag(args[1])
            args = args[2:]
            continue
        if args[0] == "--capture-arg-values" and len(args) > 1:
            capture_arg_values = _flag(args[1])
            args = args[2:]
            continue
        break
    return frameworks, capture_args, capture_arg_values, args


def main(argv: list[str]) -> None:
    frameworks, capture_args, capture_arg_values, args = parse_launcher_options(argv)

    if not args:
        print(
            "usage: python <path>/launch.py [--frameworks <name> ...] "
            "[--capture-args <0|1>] [--capture-arg-values <0|1>] -- "
            "<target.py> [args...]",
            file=sys.stderr,
        )
        sys.exit(2)

    target_script = args[0]
    script_args = args[1:]

    install_global_wraps(
        frameworks,
        capture_args=capture_args,
        capture_arg_values=capture_arg_values,
    )

    sys.argv = [target_script] + script_args
    try:
        runpy.run_path(target_script, run_name="__main__")
    finally:
        # Never let reporting replace the workload's exception.
        try:
            _report_torch_trace_callback_errors()
        except Exception:
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
