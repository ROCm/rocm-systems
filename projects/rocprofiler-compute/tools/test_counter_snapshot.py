#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the profiled counter snapshot check."""

from config_management.counter_snapshot import diff_snapshots


def test_diff_reports_added_and_removed_counters_per_panel():
    recorded = {"gfx950": {"0200": ["SQ_A", "SQ_B"], "always": ["SQ_WAVES"]}}
    current = {"gfx950": {"0200": ["SQ_B", "SQ_C"], "always": ["SQ_WAVES"]}}

    assert diff_snapshots(recorded, current) == [
        "gfx950 0200: added SQ_C",
        "gfx950 0200: removed SQ_A",
    ]
