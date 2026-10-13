#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the analysis database schema visualizer.

The renderer is exercised against a metadata object built here, so these tests
stay independent of whatever the analysis schema happens to look like.
"""

import sys
from pathlib import Path

from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Table

sys.path.insert(0, str(Path(__file__).parent))

from schema_visualizer import (  # noqa: E402
    PREFIX,
    _view_source_tables,
    build_schema_dot,
)

PARENT_TABLE = f"{PREFIX}parent"
CHILD_TABLE = f"{PREFIX}child"


def make_metadata():
    """Build a two-table schema with one required and one nullable foreign key."""
    metadata = MetaData()
    Table(PARENT_TABLE, metadata, Column("parent_uuid", Integer, primary_key=True))
    Table(
        CHILD_TABLE,
        metadata,
        Column("child_uuid", Integer, primary_key=True),
        Column(
            "parent_uuid",
            Integer,
            ForeignKey(f"{PARENT_TABLE}.parent_uuid"),
            nullable=False,
        ),
        Column(
            "optional_parent_uuid",
            Integer,
            ForeignKey(f"{PARENT_TABLE}.parent_uuid"),
            nullable=True,
        ),
        Column("label", String),
    )
    return metadata


def test_schema_dot_renders_every_table_and_column():
    """No table or column is dropped from the rendered diagram."""
    metadata = make_metadata()

    dot = build_schema_dot(metadata)

    for table in metadata.sorted_tables:
        assert table.name in dot
        for column in table.columns:
            assert column.name in dot


def test_schema_dot_dashes_only_nullable_foreign_keys():
    """A required foreign key edge is solid; a nullable one is dashed."""
    dot = build_schema_dot(make_metadata())

    assert (
        f'"{PARENT_TABLE}":"parent_uuid" -> "{CHILD_TABLE}":"parent_uuid" [style=solid'
        in dot
    )
    assert (
        f'"{PARENT_TABLE}":"parent_uuid" -> '
        f'"{CHILD_TABLE}":"optional_parent_uuid" [style=dashed' in dot
    )


def test_view_source_tables_keeps_only_known_tables():
    """A view's sources are the prefixed names that name a real table."""
    view_sql = (
        f"SELECT {CHILD_TABLE}.label FROM {CHILD_TABLE} "
        f"JOIN {PARENT_TABLE} ON {CHILD_TABLE}.parent_uuid = "
        f"{PARENT_TABLE}.parent_uuid AS {PREFIX}absent_table"
    )

    sources = _view_source_tables(view_sql, {PARENT_TABLE, CHILD_TABLE})

    assert sources == [CHILD_TABLE, PARENT_TABLE]
