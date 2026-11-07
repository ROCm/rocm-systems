#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run rocprofiler-systems tests against a ROCm nightly tarball.

The workflow lives in the ``rocprof_sys_nightly`` package next to this file.
This launcher stays thin so the script can still be copied onto a cluster and
invoked as ``python3 run-nightly-tarball-tests.py``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# The package sits beside this file and is not installed.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from rocprof_sys_nightly.cli import main  # noqa: E402
from rocprof_sys_nightly.command import die, emit  # noqa: E402

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        die("interrupted", 130)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        emit(traceback.format_exc(), stream=sys.stderr)
        die(f"unexpected error: {exc}")
