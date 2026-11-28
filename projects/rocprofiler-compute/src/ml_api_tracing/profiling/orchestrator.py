# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Check the ML API tracing options and add the launcher to the workload command."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from utils.logger import console_error, console_warning
from utils.utils_exceptions import PythonScriptNotFoundError


def validate_ml_api_trace_pc_sampling(
    frameworks: Optional[List[str]], only_pc_sampling: bool
) -> None:
    """Reject tracing when the generic profile has no counter collection."""
    if frameworks and only_pc_sampling:
        console_error(
            "--trace-ml-api cannot be used with PC-sampling-only profiling, "
            "which does not collect counters. Remove --trace-ml-api or add "
            "a counter block."
        )


def validate_ml_api_trace_attach(
    frameworks: Optional[List[str]], args: argparse.Namespace
) -> None:
    """Reject attach options that cannot inject into the running workload."""
    if not frameworks:
        return
    if args.attach_pid:
        console_error(
            "ML API tracing cannot be used with --attach-pid. "
            "ROCTX injection requires launching the workload; "
            "already-running processes cannot be instrumented. "
            "Please remove one of these options."
        )
    if args.attach_duration_msec:
        console_error(
            "ML API tracing cannot be used with --attach-duration-msec. "
            "--attach-duration-msec only applies to --attach-pid, which "
            "is incompatible with ML API tracing. Please remove one of "
            "these options."
        )


def prepare_ml_api_trace_injection(
    remaining: List[str],
    resolved_exec_path: Path,
    is_python: bool,
    script_index: Optional[int],
    skip_flag: Optional[str],
    frameworks: List[str],
) -> None:
    """Insert the inject_roctx launcher into the workload command.

    Modifies the remaining command list in place. The launcher is run by
    absolute path, with the selected frameworks passed as --frameworks
    followed by each framework name, then -- and the workload command. The
    rewrite depends on the workload type:
      1. Python interpreter - insert the launcher before the script.
      2. Direct .py script  - prepend sys.executable and the launcher.
      3. Other executables  - leave the command unchanged and emit a warning.
    """
    launch_script = Path(__file__).parent.parent / "inject_roctx" / "launch.py"
    if not launch_script.is_file():
        console_error(
            f"Cannot find inject_roctx launcher at {launch_script}. "
            "Please verify your installation."
        )

    launcher = [
        str(launch_script),
        "--frameworks",
        *sorted(frameworks),
        "--",
    ]

    if is_python:
        if skip_flag:
            console_warning(
                f"Cannot inject ROCTX markers into 'python {skip_flag}' "
                "invocations. Launching workload as-is; "
                "ML API tracing may have no effect."
            )
        elif not Path(remaining[script_index]).is_file():
            raise PythonScriptNotFoundError(remaining[script_index])
        else:
            remaining[script_index:script_index] = launcher
    elif resolved_exec_path.suffix in (".py", ".pyw", ".pyc", ".pyo"):
        remaining[0:0] = [sys.executable, *launcher]
    else:
        console_warning(
            "Command does not look like a Python entry point, "
            "skipping ROCTX auto-injection and launching workload as-is. "
            "Ensure the binary already initializes ROCTX markers, "
            "otherwise ML API tracing will have no effect."
        )

    if (resolved_exec_path.parent / "_internal").is_dir():
        console_warning(
            "Workload appears to be a self-contained binary. "
            "Such bundles typically ship private ROCm/HSA libraries, which "
            "prevents ML API tracing from collecting data. "
            "Rebuild without packaging libhsa/libhip or "
            "adjust LD_LIBRARY_PATH to /opt/rocm before profiling."
        )
