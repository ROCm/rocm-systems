# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

Invoked by absolute path as ``python <path>/launch.py --frameworks <names>
[--capture-args <0|1>] [--capture-arg-values <0|1>] -- <target.py> [args...]``.
Backends are given as a comma-separated list; when omitted the workload runs
uninstrumented. The capture options configure operator-argument capture.
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
    torch_backend = sys.modules.get("utils.inject_roctx.backends.torch")
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
) -> tuple[str, bool, bool, list[str]]:
    """Parse leading launcher options and the ``--`` separator.

    Returns ``(frameworks, capture_args, capture_arg_values, remaining)`` where
    ``remaining`` is the workload command following ``--``.
    """
    args = list(argv)
    frameworks = ""
    capture_args = True
    capture_arg_values = False
    while args and args[0] in _LAUNCHER_OPTIONS:
        option = args[0]
        value = args[1] if len(args) > 1 else ""
        if option == "--frameworks":
            frameworks = value
        elif option == "--capture-args":
            capture_args = _flag(value)
        elif option == "--capture-arg-values":
            capture_arg_values = _flag(value)
        args = args[2:]
    if args and args[0] == "--":
        args = args[1:]
    return frameworks, capture_args, capture_arg_values, args


def main(argv: list[str]) -> None:
    frameworks, capture_args, capture_arg_values, args = parse_launcher_options(argv)

    if not args:
        print(
            "usage: python <path>/launch.py [--frameworks <names>] "
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
    # Execute the workload as the top-level program (__name__ == "__main__").
    try:
        runpy.run_path(target_script, run_name="__main__")
    finally:
        _report_torch_trace_callback_errors()


if __name__ == "__main__":
    main(sys.argv[1:])
