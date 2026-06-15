# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Internal entry point used by rocprof-compute to launch a workload under
ROCTX injection.

The backend selection is read from the ROCPROFCOMPUTE_ROCTX_FRAMEWORKS
environment variable (comma-separated). When unset or empty, no backend is
installed and the workload runs uninstrumented.
"""

import importlib
import os
import runpy
import sys

_FRAMEWORK_ENV_VAR = "ROCPROFCOMPUTE_ROCTX_FRAMEWORKS"


def _report_recordfn_callback_errors() -> None:
    """Warn if the C++ RecordFunction tier swallowed callback exceptions."""
    torch_backend = sys.modules.get("utils.inject_roctx._backends._torch")
    if torch_backend is None:
        return
    stats = torch_backend.dump_recordfn_stats()
    if not stats:
        return
    callback_errors = int(stats.get("callback_errors", 0) or 0)
    if callback_errors <= 0:
        return
    from utils.logger import console_warning

    console_warning(
        "ml api trace",
        f"roctx_recordfn observed {callback_errors} swallowed callback "
        "exception(s) during the workload; some ROCTX markers may be missing "
        f"or misattributed. Stats: {dict(stats)}",
    )


if len(sys.argv) < 2:
    print(
        "usage: python -m utils.inject_roctx <target.py> [args...]",
        file=sys.stderr,
    )
    sys.exit(2)

target_script = sys.argv[1]
script_args = sys.argv[2:]

importlib.import_module("utils.inject_roctx").install_global_wraps(
    os.environ.get(_FRAMEWORK_ENV_VAR, "")
)

sys.argv = [target_script] + script_args
# Execute the workload as the top-level program (__name__ == "__main__").
try:
    runpy.run_path(target_script, run_name="__main__")
finally:
    _report_recordfn_callback_errors()
