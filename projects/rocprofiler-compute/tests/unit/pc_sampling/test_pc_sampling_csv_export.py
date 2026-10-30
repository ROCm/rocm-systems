# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for pc_sampling_csv_export.py."""

from pc_sampling.pc_sampling_csv_export import build_header


def test_build_header_places_stall_reasons_between_leading_and_trailing_columns():
    """Each stall reason becomes a prefixed column of its own."""
    header = build_header(["Count"], ["NONE", "WAITCNT"], ["Source"])

    assert header == ["Count", "Stall NONE", "Stall WAITCNT", "Source"]


def test_build_header_carries_no_stall_columns_without_reasons():
    """A host_trap workload records no stall reason, so it gets no stall column."""
    assert build_header(["Count"], [], ["Source"]) == ["Count", "Source"]
