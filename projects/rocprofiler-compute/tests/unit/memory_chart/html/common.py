# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared synthetic layout helpers for HTML memory chart unit tests."""

from dataclasses import replace
from pathlib import Path

from memory_chart.loader import Layout, LayoutBlock, LayoutItem


def make_block(block_id="source", **changes):
    """Create a synthetic block with one metric and explicit overrides."""
    return replace(
        LayoutBlock(
            id=block_id,
            title=block_id.title(),
            column=0,
            position="grid",
            style="panel",
            parent=None,
            host=None,
            stall_level=None,
            items=(LayoutItem(block_id, block_id.title(), "hit", False),),
            children=(),
            above=(),
            below=(),
        ),
        **changes,
    )


def make_layout(columns=None, arrows=(), **changes):
    """Create an in-memory layout without loading architecture data."""
    if columns is None:
        columns = ((make_block(),), (make_block("target", column=1),))

    split = next(block for column in columns for block in column)

    return replace(
        Layout(
            path=Path("synthetic.json"),
            archs=(),
            description="Synthetic diagram layout",
            scope_labels=("Left", "Right"),
            scope_split=split.id,
            columns=columns,
            arrows=arrows,
        ),
        **changes,
    )
