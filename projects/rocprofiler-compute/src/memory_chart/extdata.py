# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""The memory chart layout as stored in the analysis database for ROCm Optiq.

The format is a superset of Optiq's memory chart layout: metrics are referenced
by metric_id, and description, scope, note, host, and group are extra keys.
"""

from collections.abc import Mapping
from typing import Any

from memory_chart.loader import Layout, LayoutArrow, LayoutBlock

# Version of the stored layout format
FORMAT_VERSION = 1


def layout_extdata(layout: Layout, metric_ids: Mapping[str, str]) -> dict[str, Any]:
    """*layout* in stored form.

    Each metric name becomes its id in *metric_ids*, or None when it has none.
    """
    hosts = [block for column in layout.columns for block in column]
    blocks = [
        _block(block, order, 0, metric_ids)
        for column in layout.columns
        for order, block in enumerate(column)
    ]
    blocks += [b for host in hosts for b in _attached_blocks(host, metric_ids)]
    return {
        "version": FORMAT_VERSION,
        "description": layout.description,
        "scope": {"labels": list(layout.scope_labels), "split": layout.scope_split},
        "blocks": blocks,
        "arrows": [_arrow(arrow, metric_ids) for arrow in layout.arrows],
    }


def _attached_blocks(
    host: LayoutBlock, metric_ids: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Blocks above and below *host*, as rows of its column (-1 and 1 are nearest)."""
    above = [
        _block(block, 0, i - len(host.above), metric_ids)
        for i, block in enumerate(host.above)
    ]
    below = [_block(block, 0, i + 1, metric_ids) for i, block in enumerate(host.below)]
    return above + below


def _block(
    block: LayoutBlock, order: int, row: int, metric_ids: Mapping[str, str]
) -> dict[str, Any]:
    stored: dict[str, Any] = {
        "id": block.id,
        "title": block.title,
        "column": block.column,
        "row": row,
        "order": order,
    }
    if block.host:
        stored["host"] = block.host
    if block.note:
        stored["note"] = block.note
    if block.items:
        stored["content"] = [
            {
                "metric": metric_ids.get(item.metric),
                "title": item.title,
                "category": item.category,
            }
            for item in block.items
        ]
    if block.children:
        stored["children"] = [
            _block(child, i, row, metric_ids) for i, child in enumerate(block.children)
        ]
    return stored


def _arrow(arrow: LayoutArrow, metric_ids: Mapping[str, str]) -> dict[str, Any]:
    stored: dict[str, Any] = {
        "from": arrow.source,
        "to": arrow.target,
        "direction": arrow.direction,
        "metric": metric_ids.get(arrow.metric),
        "title": arrow.title,
        "category": arrow.category,
    }
    if arrow.group:
        stored["group"] = arrow.group
    return stored
