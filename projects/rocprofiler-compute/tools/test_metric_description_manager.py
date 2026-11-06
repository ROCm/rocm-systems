# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for tools/config_management/metric_description_manager.py.

Run from the project directory:
    python3 -m pytest tools/test_metric_description_manager.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config_management.metric_description_manager import order_like_existing


def test_sync_keeps_the_order_of_the_existing_file():
    """The views give a panel's descriptions in row order; a sync must not reorder
    the committed per-arch file, and appends what is new."""
    existing = {
        "Memory Chart": {"Read Req": {}, "Write Req": {}},
        "Wavefront": {"VGPR": {}, "SGPR": {}},
    }
    synced = {
        "Wavefront": {"SGPR": {}, "LDS Allocation": {}, "VGPR": {}},
        "Memory Chart": {"Write Req": {}, "Read Req": {}},
        "Instruction Cache": {"Hit Rate": {}},
    }

    ordered = order_like_existing(synced, existing)

    assert list(ordered) == ["Memory Chart", "Wavefront", "Instruction Cache"]
    assert list(ordered["Memory Chart"]) == ["Read Req", "Write Req"]
    assert list(ordered["Wavefront"]) == ["VGPR", "SGPR", "LDS Allocation"]
