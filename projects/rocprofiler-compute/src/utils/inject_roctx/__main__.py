# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Entry point used by rocprof-compute to launch a workload under ROCTX injection.

The backend selection is read from the ROCPROFCOMPUTE_ROCTX_FRAMEWORKS
environment variable (comma-separated). When unset or empty, no backend is
installed and the workload runs uninstrumented.
"""

import importlib
import importlib.util
import os
import sys

_ENV_VAR = "ROCPROFCOMPUTE_ROCTX_FRAMEWORKS"

if len(sys.argv) < 2:
    print(
        "usage: python -m utils.inject_roctx <target.py> [args...]",
        file=sys.stderr,
    )
    sys.exit(2)

target_script = sys.argv[1]
script_args = sys.argv[2:]

importlib.import_module("utils.inject_roctx").install_global_wraps(
    os.environ.get(_ENV_VAR, "")
)

sys.argv = [target_script] + script_args
spec = importlib.util.spec_from_file_location("__main__", target_script)
module = importlib.util.module_from_spec(spec)
sys.modules["__main__"] = module
spec.loader.exec_module(module)
