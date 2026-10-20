#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the profiled counter snapshot check."""

from config_management.counter_snapshot import diff_snapshots, panel_counters


def test_panel_counters_resolves_each_panel_on_its_own():
    """Denominators appear only under "always", built-in variables resolve to
    their counters, TCC templates stay templates, and panels without counters
    are left out."""
    counters = panel_counters(
        {
            "0200": ["SUM(SQ_INSTS) / $denom", "SUM(TCC_HIT[::_1])"],
            "0300": ["$numActiveCUs"],
            "0400": ["$cu_per_gpu * 64"],
        },
        "MI300",
    )

    assert counters == {
        "always": ["GRBM_GUI_ACTIVE", "SQ_WAVES"],
        "0200": ["SQ_INSTS", "TCC_HIT["],
        "0300": ["GRBM_GUI_ACTIVE", "SQ_BUSY_CU_CYCLES"],
    }


def test_diff_reports_added_and_removed_counters_per_panel():
    recorded = {"gfx950": {"0200": ["SQ_A", "SQ_B"], "always": ["SQ_WAVES"]}}
    current = {"gfx950": {"0200": ["SQ_B", "SQ_C"], "always": ["SQ_WAVES"]}}

    assert diff_snapshots(recorded, current) == [
        "gfx950 0200: added SQ_C",
        "gfx950 0200: removed SQ_A",
    ]
