#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Thin wrapper so the viewer runs as a plain script.

The implementation lives in the ``openspec_viewer`` package next to this file;
see ``openspec_viewer/__init__.py``. Standard library only, no build step::

    python3 tools/openspec_viewer.py                       # default project
    python3 tools/openspec_viewer.py A/openspec B/openspec  # several projects
    python3 tools/openspec_viewer.py --check                # CI structure gate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openspec_viewer.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
