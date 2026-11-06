# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Adapt resolved memory chart layouts to the HTML diagram model."""

from dataclasses import dataclass
from typing import Mapping, Optional, TypedDict

from ..loader import Layout, LayoutArrow, LayoutBlock
from ..units import PLAIN, DisplayUnit, display_unit

ArrowGroup = TypedDict(
    "ArrowGroup",
    {"from": str, "to": str, "lanes": list[dict[str, object]]},
)


@dataclass(frozen=True)
class SlotSpec:
    """Describe one metric value displayed in the diagram."""

    slot_id: str
    metric: str
    unit: DisplayUnit
    category: str
    on_arrow: bool
    cu_block: bool
    bar: bool


def diagram_payload(
    layout: Layout, units: Optional[Mapping[str, str]] = None
) -> dict[str, object]:
    """Copy resolved topology and layout metadata into a page model."""
    specs = {spec.slot_id: spec for spec in slot_specs(layout, units or {})}

    grid_blocks = [
        _block_payload(block, order, specs)
        for column in layout.columns
        for order, block in enumerate(column)
    ]

    io_blocks = [
        _block_payload(block, order, specs)
        for host in layout.blocks()
        for attached_blocks in (host.above, host.below)
        for order, block in enumerate(attached_blocks)
    ]

    split = next(block for block in layout.blocks() if block.id == layout.scope_split)

    return {
        "arch": layout.path.stem,
        "gridBlocks": grid_blocks,
        "ioBlocks": io_blocks,
        "arrows": _arrow_groups(layout.arrows),
        "scope": {
            "splitColumn": split.column,
            "splitBlock": split.id,
            "gpuLabel": layout.scope_labels[0],
            "memoryLabel": layout.scope_labels[1],
        },
    }


def slot_specs(layout: Layout, units: Mapping[str, str]) -> tuple[SlotSpec, ...]:
    """Describe block slots in declared order, followed by arrow slots."""
    specs: list[SlotSpec] = []

    for block in layout.blocks():
        for index, item in enumerate(block.items):
            unit = display_unit(units.get(item.metric)) or PLAIN
            specs.append(
                SlotSpec(
                    slot_id=_item_slot_id(block.id, index),
                    metric=item.metric,
                    unit=unit,
                    category=item.category,
                    on_arrow=False,
                    cu_block=block.column == 0,
                    bar=item.bar,
                )
            )

    specs.extend(
        SlotSpec(
            slot_id=_arrow_slot_id(index),
            metric=arrow.metric,
            unit=display_unit(units.get(arrow.metric)) or PLAIN,
            category=arrow.category,
            on_arrow=True,
            cu_block=False,
            bar=False,
        )
        for index, arrow in enumerate(layout.arrows)
    )

    return tuple(specs)


def diagram_metrics(layout: Layout) -> frozenset[str]:
    """Return the metric set already resolved by the loader."""
    return frozenset(layout.metrics())


def _block_payload(
    block: LayoutBlock, order: int, specs: Mapping[str, SlotSpec]
) -> dict[str, object]:
    """Preserve block display fields, nesting, and attachment metadata."""
    return {
        "id": block.id,
        "title": block.title,
        "column": block.column,
        "order": order,
        "position": block.position,
        "style": block.style,
        "parent": block.parent,
        "host": block.host,
        "stallLevel": block.stall_level,
        "content": [
            {
                "metric": item.metric,
                "title": item.title,
                "category": item.category,
                "slotId": slot_id,
                "bar": specs[slot_id].bar,
            }
            for index, item in enumerate(block.items)
            for slot_id in (_item_slot_id(block.id, index),)
        ],
        "children": [
            _block_payload(child, index, specs)
            for index, child in enumerate(block.children)
        ],
    }


def _arrow_groups(arrows: tuple[LayoutArrow, ...]) -> list[ArrowGroup]:
    """Group lanes by endpoint pair without changing their declared order."""
    groups: dict[tuple[str, str], ArrowGroup] = {}

    for index, arrow in enumerate(arrows):
        pair = (arrow.source, arrow.target)

        if pair not in groups:
            groups[pair] = {"from": pair[0], "to": pair[1], "lanes": []}

        lanes = groups[pair]["lanes"]
        previous_group = lanes[-1].get("group") if lanes else None

        lanes.append({
            "from": arrow.source,
            "to": arrow.target,
            "metric": arrow.metric,
            "title": arrow.title,
            "category": arrow.category,
            "style": arrow.style,
            "direction": arrow.direction,
            "group": arrow.group,
            "slotId": _arrow_slot_id(index),
            "groupHeader": arrow.group if arrow.group != previous_group else None,
        })

    return list(groups.values())


def _item_slot_id(block_id: str, index: int) -> str:
    """Return the stable slot identifier for an item in a block."""
    return f"{block_id}.{index}"


def _arrow_slot_id(index: int) -> str:
    """Return the stable slot identifier for an arrow."""
    return f"arrow.{index}"
