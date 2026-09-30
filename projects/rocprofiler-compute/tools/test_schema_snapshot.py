#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the version bump rules of the analysis database schema snapshot."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from schema_snapshot import (  # noqa: E402
    PREFIX,
    build_schema_snapshot,
    check_version_bump,
    required_bump,
)

ARROW_TABLE = f"{PREFIX}memchart_arrow"
DIRECTION_VALUES = f"{PREFIX}memchart_arrow.direction"


def rename_table(snapshot):
    snapshot["tables"]["compute_renamed"] = snapshot["tables"].pop(ARROW_TABLE)


def remove_column(snapshot):
    del snapshot["tables"][ARROW_TABLE]["columns"]["group_label"]


def remove_view_column(snapshot):
    snapshot["views"][f"{PREFIX}kernel_view"].pop()


def add_value(snapshot):
    snapshot["value_sets"][DIRECTION_VALUES].append("none")


def add_column(snapshot):
    snapshot["tables"][ARROW_TABLE]["columns"]["width"] = {
        "type": "INTEGER",
        "not_null": False,
        "primary_key": False,
    }


def remove_value(snapshot):
    snapshot["value_sets"][DIRECTION_VALUES].remove("both")


def no_change(snapshot):
    pass


def edited_snapshot(edit, schema_version):
    """Return the current schema snapshot with one edit and a new version."""
    snapshot = build_schema_snapshot()
    edit(snapshot)
    return {**snapshot, "schema_version": schema_version}


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (rename_table, "major"),
        (remove_column, "major"),
        (remove_view_column, "major"),
        (add_value, "major"),
        (add_column, "minor"),
        # A reader never meets the removed value, so nothing breaks.
        (remove_value, "minor"),
        (no_change, "none"),
    ],
)
def test_required_bump_follows_the_version_rules(edit, expected):
    old = build_schema_snapshot()

    assert required_bump(old, edited_snapshot(edit, "0.0.0")) == expected


@pytest.mark.parametrize(
    ("edit", "new_version", "allowed"),
    [
        (remove_column, "3.0.0", True),
        (remove_column, "2.5.0", False),
        (add_column, "2.5.0", True),
        (add_column, "2.4.1", False),
        (add_column, "2.4.0", False),
        (no_change, "2.4.1", True),
    ],
)
def test_check_version_bump_rejects_a_bump_too_small_for_the_change(
    edit, new_version, allowed
):
    old = {**build_schema_snapshot(), "schema_version": "2.4.0"}

    assert (check_version_bump(old, edited_snapshot(edit, new_version)) == "") is (
        allowed
    )
